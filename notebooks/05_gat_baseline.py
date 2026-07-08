#!/usr/bin/env python3
"""Multi-network GAT baseline for TNBC tissue classification.

Every sample is evaluated against all three tissue-specific networks.  This
avoids the previous label leakage where the true tissue label selected the graph
used for inference, while still allowing the model to learn how the normal,
primary, and metastatic network topologies differ.
"""

import copy
import json
import random
import sys
from itertools import product
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

_NOTEBOOK_DIR = Path(__file__).resolve().parent
if str(_NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOK_DIR))

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from cv_utils import load_patient_groups, patient_holdout_val_split
from evaluation_metrics import (
    compute_classification_metrics,
    print_test_metrics,
    save_test_metrics,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv


PROCESSED_DIR = Path('data/processed')
RESULTS_DIR = Path('results')
TABLES_DIR = RESULTS_DIR / 'tables'
FIGURES_DIR = RESULTS_DIR / 'figures'
MODELS_DIR = Path('models')

GRAPH_FILES = {
    'normal': RESULTS_DIR / 'normal_network.graphml',
    'primary': RESULTS_DIR / 'primary_network.graphml',
    'metastatic': RESULTS_DIR / 'metastatic_network.graphml',
}

LABEL_MAP = {'normal': 0, 'primary': 1, 'metastatic': 2}
CLASS_NAMES = [k for k, _ in sorted(LABEL_MAP.items(), key=lambda x: x[1])]
NETWORK_NAMES = list(GRAPH_FILES.keys())
RANDOM_STATE = 42
MAX_EPOCHS = 250
PATIENCE = 35
BATCH_SIZE = 16
BRANCH_LOSS_WEIGHT = 0.25

PARAM_GRID = {
    'num_heads': [2, 4],
    'hidden_size': [32, 64],
    'dropout': [0.1, 0.2],
    'lr': [0.001],
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if DEVICE.type == 'cuda':
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_split_indices(sample_ids: List[str]):
    candidates = [
        PROCESSED_DIR / 'split_indices.json',
        PROCESSED_DIR / 'saved_split_indices.json',
        PROCESSED_DIR / 'train_test_split.json',
        PROCESSED_DIR / 'split_indices.npz',
        PROCESSED_DIR / 'train_indices.npy',
        PROCESSED_DIR / 'test_indices.npy',
    ]

    for path in candidates:
        if not path.exists():
            continue

        if path.suffix == '.json':
            with path.open('r', encoding='utf-8') as f:
                payload = json.load(f)
            train_raw = payload.get('train_sample_ids') or payload.get('train_indices') or payload.get('train')
            test_raw = payload.get('test_sample_ids') or payload.get('test_indices') or payload.get('test')
            train_idx = _coerce_indices(train_raw, sample_ids)
            test_idx = _coerce_indices(test_raw, sample_ids)
            if train_idx is not None and test_idx is not None:
                return train_idx, test_idx

        if path.suffix == '.npz':
            payload = np.load(path, allow_pickle=True)
            train_raw = payload.get('train_indices', None) or payload.get('train', None)
            test_raw = payload.get('test_indices', None) or payload.get('test', None)
            train_idx = _coerce_indices(train_raw, sample_ids)
            test_idx = _coerce_indices(test_raw, sample_ids)
            if train_idx is not None and test_idx is not None:
                return train_idx, test_idx

        if path.suffix == '.npy':
            loaded = np.load(path, allow_pickle=True)
            train_idx = _coerce_indices(loaded, sample_ids)
            if train_idx is not None:
                sample_set = set(sample_ids)
                test_idx = [x for x in sample_ids if x not in set(train_idx)]
                if len(test_idx) > 0:
                    return train_idx, test_idx

    return None


def _coerce_indices(raw, sample_ids: List[str]):
    if raw is None:
        return None

    arr = np.asarray(raw, dtype=object)
    if arr.size == 0:
        return []

    if np.issubdtype(arr.dtype, np.integer):
        if arr.max() < len(sample_ids) and arr.min() >= 0:
            return [sample_ids[int(i)] for i in arr]
        return None

    if np.issubdtype(arr.dtype, np.bool_):
        if arr.size != len(sample_ids):
            return None
        return [sample_ids[i] for i, flag in enumerate(arr) if bool(flag)]

    values = [str(v) for v in arr.tolist()]
    if all(v in sample_ids for v in values):
        return values
    return None


def load_data() -> Tuple[pd.DataFrame, pd.Series, pd.Index, pd.Index]:
    train_mirna = pd.read_csv(PROCESSED_DIR / 'train_mirna.csv', index_col=0)
    train_mrna = pd.read_csv(PROCESSED_DIR / 'train_mrna.csv', index_col=0)
    test_mirna = pd.read_csv(PROCESSED_DIR / 'test_mirna.csv', index_col=0)
    test_mrna = pd.read_csv(PROCESSED_DIR / 'test_mrna.csv', index_col=0)

    train_labels = pd.read_csv(PROCESSED_DIR / 'train_labels.csv', index_col='sample_id')['tissue_group']
    test_labels = pd.read_csv(PROCESSED_DIR / 'test_labels.csv', index_col='sample_id')['tissue_group']

    train_mirna = train_mirna.loc[train_labels.index]
    train_mrna = train_mrna.loc[train_labels.index]
    test_mirna = test_mirna.loc[test_labels.index]
    test_mrna = test_mrna.loc[test_labels.index]

    x_train = pd.concat([train_mirna, train_mrna], axis=1)
    x_test = pd.concat([test_mirna, test_mrna], axis=1)

    common_cols = x_train.columns.intersection(x_test.columns)
    x_train = x_train[common_cols]
    x_test = x_test[common_cols]

    x_all = pd.concat([x_train, x_test], axis=0)
    y_all = pd.concat([train_labels, test_labels], axis=0)

    median_vals = x_all.loc[x_train.index].median(axis=0)
    x_all = x_all.fillna(median_vals)

    saved_split = load_split_indices(list(x_all.index))
    if saved_split is not None:
        train_ids, test_ids = saved_split
        train_ids = pd.Index(train_ids).intersection(x_all.index)
        test_ids = pd.Index(test_ids).intersection(x_all.index)
        if len(train_ids) == 0 or len(test_ids) == 0:
            train_ids = x_train.index
            test_ids = x_test.index
    else:
        train_ids = x_train.index
        test_ids = x_test.index

    return x_all, y_all, pd.Index(train_ids), pd.Index(test_ids)


def _parse_edge_weight(value) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(parsed):
        return 0.0
    return parsed


def load_graphs(shared_nodes: Sequence[str]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    node_to_idx = {name: idx for idx, name in enumerate(shared_nodes)}
    edge_index_by_network: Dict[str, torch.Tensor] = {}
    edge_attr_by_network: Dict[str, torch.Tensor] = {}

    for network, path in GRAPH_FILES.items():
        graph = nx.read_graphml(path)
        edge_src = []
        edge_tgt = []
        edge_attr = []

        for src, tgt, attrs in graph.edges(data=True):
            if src not in node_to_idx or tgt not in node_to_idx:
                continue
            src_idx = node_to_idx[src]
            tgt_idx = node_to_idx[tgt]
            weight = _parse_edge_weight(attrs.get('weight', attrs.get('pearson_r', 0.0)))
            edge_src.extend([src_idx, tgt_idx])
            edge_tgt.extend([tgt_idx, src_idx])
            edge_attr.extend([[weight], [weight]])

        if len(edge_src) == 0:
            raise ValueError(f'No usable edges found in {network} graph: {path}')

        edge_index_by_network[network] = torch.tensor([edge_src, edge_tgt], dtype=torch.long)
        edge_attr_by_network[network] = torch.tensor(edge_attr, dtype=torch.float32)

    return edge_index_by_network, edge_attr_by_network


def infer_node_types(nodes: Sequence[str], x_mirna_columns: Sequence[str]) -> torch.Tensor:
    mirna_set = set(x_mirna_columns)
    node_types = [0 if node in mirna_set else 1 for node in nodes]
    return torch.tensor(node_types, dtype=torch.long)


def make_graph_data(sample_id: str, x_row: pd.Series, label: str, nodes: Sequence[str]) -> Data:
    feature_vector = x_row.reindex(nodes)
    if feature_vector.isna().any():
        missing = feature_vector[feature_vector.isna()].index.tolist()
        raise ValueError(f'{sample_id}: missing features for {len(missing)} graph nodes')

    x_tensor = torch.from_numpy(feature_vector.to_numpy(dtype=float, copy=True)).view(-1, 1).float()
    y_tensor = torch.tensor([LABEL_MAP[label]], dtype=torch.long)
    return Data(x=x_tensor, y=y_tensor, sample_id=sample_id)


def build_dataset(
    x_all: pd.DataFrame,
    y_all: pd.Series,
    train_ids: pd.Index,
    test_ids: pd.Index,
    nodes: Sequence[str],
) -> Tuple[Dict[str, Data], List[str], List[str], Dict[str, int]]:
    dataset_by_id = {}
    y_numeric = {}

    for sample_id in x_all.index:
        label = y_all.loc[sample_id]
        if label not in LABEL_MAP:
            continue
        dataset_by_id[sample_id] = make_graph_data(sample_id, x_all.loc[sample_id], label, nodes)
        y_numeric[sample_id] = LABEL_MAP[label]

    train_ids = [sid for sid in train_ids if sid in dataset_by_id]
    test_ids = [sid for sid in test_ids if sid in dataset_by_id]
    return dataset_by_id, train_ids, test_ids, y_numeric


class MultiNetworkGAT(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        node_types: torch.Tensor,
        edge_index_by_network: Dict[str, torch.Tensor],
        edge_attr_by_network: Dict[str, torch.Tensor],
        num_heads: int = 4,
        hidden_size: int = 64,
        dropout: float = 0.2,
        num_layers: int = 2,
        node_embedding_size: int = 16,
        modality_embedding_size: int = 4,
        expression_hidden_size: int = 64,
    ):
        super().__init__()
        self.network_names = list(edge_index_by_network.keys())
        self.num_nodes = num_nodes
        self.num_layers = int(num_layers)
        self._dropout = dropout
        self._edge_cache = {}

        if self.num_layers < 1:
            raise ValueError('num_layers must be at least 1')

        self.register_buffer('node_ids', torch.arange(num_nodes, dtype=torch.long))
        self.register_buffer('node_types', node_types.clone().detach().long())
        for network, edge_index in edge_index_by_network.items():
            self.register_buffer(f'{network}_edge_index', edge_index.clone().detach())
        for network, edge_attr in edge_attr_by_network.items():
            self.register_buffer(f'{network}_edge_attr', edge_attr.clone().detach())

        in_channels = 1 + node_embedding_size + modality_embedding_size
        self.node_embedding = nn.Embedding(num_nodes, node_embedding_size)
        self.modality_embedding = nn.Embedding(2, modality_embedding_size)
        self.conv1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_size,
            heads=num_heads,
            concat=self.num_layers > 1,
            dropout=dropout,
            edge_dim=1,
        )
        if self.num_layers > 1:
            self.conv2 = GATConv(
                in_channels=hidden_size * num_heads,
                out_channels=hidden_size,
                heads=1,
                concat=False,
                dropout=dropout,
                edge_dim=1,
            )
        else:
            self.conv2 = None
        self.extra_convs = nn.ModuleList()
        prev_channels = hidden_size
        for _ in range(2, self.num_layers):
            self.extra_convs.append(GATConv(
                in_channels=prev_channels,
                out_channels=hidden_size,
                heads=1,
                concat=False,
                dropout=dropout,
                edge_dim=1,
            ))
        self.branch_classifier = nn.Linear(hidden_size * 2, len(CLASS_NAMES))
        self.expression_head = nn.Sequential(
            nn.LayerNorm(num_nodes),
            nn.Linear(num_nodes, expression_hidden_size),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        combined_size = expression_hidden_size + len(self.network_names) * (hidden_size * 2 + len(CLASS_NAMES))
        self.classifier = nn.Sequential(
            nn.Linear(combined_size, 96),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(96, len(CLASS_NAMES)),
        )

    def _node_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        batch_size = x.size(0)
        node_emb = self.node_embedding(self.node_ids)
        modality_emb = self.modality_embedding(self.node_types)
        node_emb = node_emb.unsqueeze(0).expand(batch_size, -1, -1)
        modality_emb = modality_emb.unsqueeze(0).expand(batch_size, -1, -1)
        return torch.cat([x, node_emb, modality_emb], dim=2)

    def _edge_tensors(self, network: str) -> Tuple[torch.Tensor, torch.Tensor]:
        return getattr(self, f'{network}_edge_index'), getattr(self, f'{network}_edge_attr')

    def _batched_edge_tensors(self, network: str, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_key = (network, batch_size, self.node_ids.device)
        cached = self._edge_cache.get(cache_key)
        if cached is not None:
            return cached

        edge_index, edge_attr = self._edge_tensors(network)
        offsets = torch.arange(batch_size, device=edge_index.device, dtype=edge_index.dtype) * self.num_nodes
        batched_edge_index = edge_index.unsqueeze(0) + offsets.view(-1, 1, 1)
        batched_edge_index = batched_edge_index.permute(1, 0, 2).reshape(2, -1).contiguous()
        batched_edge_attr = edge_attr.repeat(batch_size, 1).contiguous()
        self._edge_cache[cache_key] = (batched_edge_index, batched_edge_attr)
        return batched_edge_index, batched_edge_attr

    def _network_embedding(
        self,
        x_in: torch.Tensor,
        network: str,
        return_attention: bool = False,
    ):
        batch_size = x_in.size(0)
        edge_index, edge_attr = self._batched_edge_tensors(network, batch_size)
        h = x_in.reshape(batch_size * self.num_nodes, -1)
        att_edge_index = None
        alpha = None
        conv_layers = ([self.conv2] if self.conv2 is not None else []) + list(self.extra_convs)
        if return_attention and not conv_layers:
            h, (att_edge_index, alpha) = self.conv1(
                h,
                edge_index,
                edge_attr=edge_attr,
                return_attention_weights=True,
            )
        else:
            h = self.conv1(h, edge_index, edge_attr=edge_attr)

        for layer_i, conv in enumerate(conv_layers):
            h = F.elu(h)
            h = F.dropout(h, p=self._dropout, training=self.training)
            is_last = layer_i == len(conv_layers) - 1
            if return_attention and is_last:
                h, (att_edge_index, alpha) = conv(
                    h,
                    edge_index,
                    edge_attr=edge_attr,
                    return_attention_weights=True,
                )
            else:
                h = conv(h, edge_index, edge_attr=edge_attr)

        h = F.elu(h).reshape(batch_size, self.num_nodes, -1)
        pooled = torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=1)
        branch_logits = self.branch_classifier(pooled)
        if return_attention:
            return pooled, branch_logits, att_edge_index, alpha
        return pooled, branch_logits

    def forward(self, x: torch.Tensor):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x_in = self._node_input(x)
        graph_parts = []
        branch_logits = []
        for network in self.network_names:
            pooled, logits = self._network_embedding(x_in, network)
            graph_parts.append(pooled)
            branch_logits.append(logits)

        expression_features = self.expression_head(x.squeeze(-1))
        combined = torch.cat([expression_features, *graph_parts, *branch_logits], dim=1)
        logits = self.classifier(combined)
        branch_logits_tensor = torch.stack(branch_logits, dim=1)
        return logits, branch_logits_tensor

    def attention_for_network(self, x: torch.Tensor, network: str):
        x_in = self._node_input(x)
        return self._network_embedding(x_in, network, return_attention=True)


def compute_class_weights(tr_ids: Sequence[str], y_numeric: Dict[str, int]) -> torch.Tensor:
    labels = np.array([y_numeric[sid] for sid in tr_ids])
    counts = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(float)
    counts[counts == 0] = 1.0
    weights = len(labels) / (len(CLASS_NAMES) * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def iter_minibatches(ids: Sequence[str], batch_size: int, shuffle: bool = True) -> List[List[str]]:
    ids = list(ids)
    if shuffle:
        random.shuffle(ids)
    return [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]


def batch_logits_and_labels(model: MultiNetworkGAT, dataset_by_id: Dict[str, Data], ids: Sequence[str]):
    x_batch = torch.stack([dataset_by_id[sample_id].x for sample_id in ids], dim=0)
    labels = torch.stack([dataset_by_id[sample_id].y.squeeze(0) for sample_id in ids], dim=0)
    logits, branch_logits = model(x_batch)
    return logits, branch_logits, labels


def evaluate_ids(model: MultiNetworkGAT, dataset_by_id: Dict[str, Data], ids: Sequence[str]) -> Dict:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[np.ndarray] = []

    with torch.no_grad():
        for batch_ids in iter_minibatches(ids, BATCH_SIZE, shuffle=False):
            logits, _, labels = batch_logits_and_labels(model, dataset_by_id, batch_ids)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            y_true.extend(labels.cpu().numpy().astype(int).tolist())
            y_pred.extend(np.argmax(probs, axis=1).astype(int).tolist())
            y_prob.extend(list(probs))

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    y_prob_np = np.vstack(y_prob)
    return compute_classification_metrics(
        y_true=y_true_np,
        y_pred=y_pred_np,
        y_prob=y_prob_np,
        class_names=CLASS_NAMES,
        class_ids=[0, 1, 2],
    ) | {'y_true': y_true_np, 'y_pred': y_pred_np, 'y_prob': y_prob_np}


def train_with_early_stopping(
    model: MultiNetworkGAT,
    dataset_by_id: Dict[str, Data],
    tr_ids: Sequence[str],
    val_ids: Sequence[str],
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    class_weights: torch.Tensor = None,
    verbose: bool = True,
) -> Tuple[MultiNetworkGAT, float, int]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = copy.deepcopy(model.state_dict())
    best_val_macro_f1 = -float('inf')
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses = []
        for batch_ids in iter_minibatches(tr_ids, BATCH_SIZE, shuffle=True):
            optimizer.zero_grad()
            logits, branch_logits, labels = batch_logits_and_labels(model, dataset_by_id, batch_ids)
            final_loss = F.cross_entropy(logits, labels, weight=class_weights)
            branch_labels = labels.unsqueeze(1).expand(-1, len(model.network_names)).reshape(-1)
            branch_loss = F.cross_entropy(
                branch_logits.reshape(-1, len(CLASS_NAMES)),
                branch_labels,
                weight=class_weights,
            )
            loss = final_loss + BRANCH_LOSS_WEIGHT * branch_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_metrics = evaluate_ids(model, dataset_by_id, val_ids)
        val_macro_f1 = float(val_metrics['macro_f1'])
        train_loss = float(np.mean(train_losses)) if train_losses else float('nan')

        if verbose:
            print(
                f'Epoch {epoch:03d} | train_loss={train_loss:.4f} | '
                f'val_acc={val_metrics["accuracy"]:.4f} | val_macro_f1={val_macro_f1:.4f}'
            )

        if val_macro_f1 > best_val_macro_f1 + 1e-8:
            best_val_macro_f1 = val_macro_f1
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                if verbose:
                    print(f'Early stopping at epoch {epoch}')
                break

    model.load_state_dict(best_state)
    if verbose:
        print(f'Validation macro F1 at best epoch {best_epoch}: {best_val_macro_f1:.4f}')
    return model, best_val_macro_f1, best_epoch


def make_model(
    params: Dict[str, float],
    num_nodes: int,
    node_types: torch.Tensor,
    edge_index_by_network: Dict[str, torch.Tensor],
    edge_attr_by_network: Dict[str, torch.Tensor],
) -> MultiNetworkGAT:
    return MultiNetworkGAT(
        num_nodes=num_nodes,
        node_types=node_types,
        edge_index_by_network=edge_index_by_network,
        edge_attr_by_network=edge_attr_by_network,
        num_heads=int(params['num_heads']),
        hidden_size=int(params['hidden_size']),
        dropout=float(params['dropout']),
        num_layers=int(params.get('num_layers', 2)),
    ).to(DEVICE)


def grid_search_cv(
    dataset_by_id: Dict[str, Data],
    train_ids: List[str],
    y_numeric: Dict[str, int],
    train_groups: pd.Series,
    num_nodes: int,
    node_types: torch.Tensor,
    edge_index_by_network: Dict[str, torch.Tensor],
    edge_attr_by_network: Dict[str, torch.Tensor],
    n_folds: int = 5,  # match XGBoost/TabNet 5-fold for a fair head-to-head comparison
) -> Tuple[Dict[str, float], pd.DataFrame]:
    import os as _os
    _fix = _os.environ.get('GAT_FIX_PARAMS')
    if _fix:
        _p = dict(kv.split('=') for kv in _fix.split(','))
        fixed = {'num_heads': int(float(_p['num_heads'])), 'hidden_size': int(float(_p['hidden_size'])),
                 'dropout': float(_p['dropout']), 'lr': float(_p['lr'])}
        print(f'GAT_FIX_PARAMS set -> skipping grid search, using {fixed}')
        return fixed, pd.DataFrame([fixed])
    keys = list(PARAM_GRID.keys())
    combos = list(product(*[PARAM_GRID[k] for k in keys]))
    print(f'Grid search: {len(combos)} combinations x {n_folds}-fold patient-aware CV')

    y_train = pd.Series([y_numeric[sid] for sid in train_ids], index=train_ids)
    group_arr = train_groups.loc[train_ids].values
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    results = []
    for ci, vals in enumerate(combos, 1):
        params = dict(zip(keys, vals))
        print(f'\n[{ci}/{len(combos)}] {params}')

        fold_scores = []
        fold_epochs = []
        for fold_i, (tr_idx, val_idx) in enumerate(sgkf.split(train_ids, y_train.values, group_arr)):
            tr_ids = [train_ids[i] for i in tr_idx]
            vl_ids = [train_ids[i] for i in val_idx]

            set_seed(RANDOM_STATE + fold_i)
            model = make_model(params, num_nodes, node_types, edge_index_by_network, edge_attr_by_network)
            class_weights = compute_class_weights(tr_ids, y_numeric).to(DEVICE)
            model, best_score, best_epoch = train_with_early_stopping(
                model=model,
                dataset_by_id=dataset_by_id,
                tr_ids=tr_ids,
                val_ids=vl_ids,
                lr=float(params['lr']),
                class_weights=class_weights,
                verbose=False,
            )
            fold_scores.append(best_score)
            fold_epochs.append(best_epoch)
            print(f'  fold {fold_i + 1}/{n_folds} val_macro_f1={best_score:.4f} best_epoch={best_epoch}')

        avg_score = float(np.mean(fold_scores))
        std_score = float(np.std(fold_scores))
        row = {
            **params,
            'mean_val_macro_f1': avg_score,
            'std_val_macro_f1': std_score,
            'mean_best_epoch': float(np.mean(fold_epochs)),
        }
        results.append(row)
        print(f'  => avg_val_macro_f1={avg_score:.4f} +/- {std_score:.4f}')

    df = pd.DataFrame(results).sort_values('mean_val_macro_f1', ascending=False).reset_index(drop=True)
    best_row = df.iloc[0]
    best_params = {k: best_row[k] for k in keys}
    print(f'\nBest params: {best_params}')
    print(f'Best mean val macro F1: {best_row["mean_val_macro_f1"]:.4f}')
    return best_params, df


def extract_attention(
    model: MultiNetworkGAT,
    dataset_by_id: Dict[str, Data],
    test_ids: Sequence[str],
    nodes: Sequence[str],
) -> pd.DataFrame:
    rows = []
    model.eval()
    with torch.no_grad():
        for network in model.network_names:
            pair_scores: Dict[Tuple[str, str], List[float]] = {}
            for sample_id in test_ids:
                data = dataset_by_id[sample_id]
                _, _, att_edge_index, alpha = model.attention_for_network(data.x, network)
                att_edge_index = att_edge_index.cpu().numpy()
                alpha = alpha.mean(dim=-1).cpu().numpy()

                for (u_idx, v_idx), weight in zip(att_edge_index.T, alpha):
                    if u_idx == v_idx:
                        continue
                    src = str(nodes[int(u_idx)])
                    dst = str(nodes[int(v_idx)])
                    key = tuple(sorted((src, dst)))
                    pair_scores.setdefault(key, []).append(float(weight))

            for (u, v), weights in pair_scores.items():
                rows.append({
                    'network': network,
                    'node_u': u,
                    'node_v': v,
                    'mean_attention': float(np.mean(weights)),
                })

    att_df = pd.DataFrame(rows)
    if not att_df.empty:
        att_df = (
            att_df.sort_values(['network', 'mean_attention'], ascending=[True, False])
            .groupby('network', group_keys=False)
            .head(30)
            .reset_index(drop=True)
        )
    return att_df


def extract_network_branch_scores(
    model: MultiNetworkGAT,
    dataset_by_id: Dict[str, Data],
    ids: Sequence[str],
) -> pd.DataFrame:
    rows = []
    id_to_label = {v: k for k, v in LABEL_MAP.items()}
    model.eval()
    with torch.no_grad():
        for batch_ids in iter_minibatches(ids, BATCH_SIZE, shuffle=False):
            logits, branch_logits, labels = batch_logits_and_labels(model, dataset_by_id, batch_ids)
            preds = torch.argmax(logits, dim=1).cpu().numpy().astype(int)
            branch_probs = F.softmax(branch_logits, dim=-1).cpu().numpy()
            labels_np = labels.cpu().numpy().astype(int)
            for sample_i, sample_id in enumerate(batch_ids):
                true_label = id_to_label[int(labels_np[sample_i])]
                pred_label = id_to_label[int(preds[sample_i])]
                for network_i, network in enumerate(model.network_names):
                    row = {
                        'sample_id': sample_id,
                        'true_label': true_label,
                        'pred_label': pred_label,
                        'network': network,
                    }
                    for class_i, class_name in enumerate(CLASS_NAMES):
                        row[f'branch_prob_{class_name}'] = float(branch_probs[sample_i, network_i, class_i])
                    rows.append(row)
    return pd.DataFrame(rows)


def summarize_network_branch_scores(branch_scores: pd.DataFrame) -> pd.DataFrame:
    """Summarize how each tissue network branch scores each true class."""
    prob_cols = [f'branch_prob_{class_name}' for class_name in CLASS_NAMES]
    summary = (
        branch_scores
        .groupby(['true_label', 'network'], as_index=False)[prob_cols]
        .mean()
    )
    rows = []
    for _, row in summary.iterrows():
        true_label = row['true_label']
        true_prob_col = f'branch_prob_{true_label}'
        other_cols = [c for c in prob_cols if c != true_prob_col]
        rows.append({
            **row.to_dict(),
            'true_class_branch_prob': float(row[true_prob_col]),
            'margin_vs_best_other_class': float(row[true_prob_col] - row[other_cols].max()),
        })
    return pd.DataFrame(rows)


def update_comparison_csv(
    path: Path,
    accuracy: float,
    macro_f1: float,
    per_class_auc: Dict[str, float],
) -> None:
    row = {
        'model': 'Multi-network GAT',
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'auc_roc_normal': per_class_auc.get('normal', float('nan')),
        'auc_roc_primary': per_class_auc.get('primary', float('nan')),
        'auc_roc_metastatic': per_class_auc.get('metastatic', float('nan')),
    }

    row_df = pd.DataFrame([row])
    if path.exists():
        existing = pd.read_csv(path)
        existing = existing[~existing['model'].isin(['GAT Baseline', 'Multi-network GAT'])]
        updated = pd.concat([existing, row_df], ignore_index=True)
    else:
        updated = row_df

    columns = [
        'model',
        'accuracy',
        'macro_f1',
        'auc_roc_normal',
        'auc_roc_primary',
        'auc_roc_metastatic',
    ]
    updated = updated.reindex(columns=columns)
    updated.to_csv(path, index=False)


def main() -> None:
    set_seed()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f'Using device: {DEVICE}')
    if DEVICE.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        print(f'  GPU: {props.name}')
        print(f'  VRAM: {props.total_memory / 1e9:.1f} GB')

    x_all, y_all, train_ids, test_ids = load_data()
    nodes = x_all.columns.tolist()
    train_mirna_cols = pd.read_csv(PROCESSED_DIR / 'train_mirna.csv', index_col=0).columns.tolist()
    node_types = infer_node_types(nodes, train_mirna_cols)
    edge_index_by_network, edge_attr_by_network = load_graphs(nodes)

    dataset_by_id, train_ids, test_ids, y_numeric = build_dataset(
        x_all=x_all,
        y_all=y_all,
        train_ids=train_ids,
        test_ids=test_ids,
        nodes=nodes,
    )
    dataset_by_id = {sid: data.to(DEVICE) for sid, data in dataset_by_id.items()}
    node_types = node_types.to(DEVICE)
    edge_index_by_network = {k: v.to(DEVICE) for k, v in edge_index_by_network.items()}
    edge_attr_by_network = {k: v.to(DEVICE) for k, v in edge_attr_by_network.items()}

    print(f'Feature nodes: {len(nodes)} ({len(train_mirna_cols)} miRNAs, {len(nodes) - len(train_mirna_cols)} mRNAs)')
    print(f'Train samples: {len(train_ids)}, Test samples: {len(test_ids)}')

    train_groups = load_patient_groups(train_ids)
    best_params, grid_df = grid_search_cv(
        dataset_by_id=dataset_by_id,
        train_ids=train_ids,
        y_numeric=y_numeric,
        train_groups=train_groups,
        num_nodes=len(nodes),
        node_types=node_types,
        edge_index_by_network=edge_index_by_network,
        edge_attr_by_network=edge_attr_by_network,
    )

    grid_path = TABLES_DIR / 'gat_grid_search.csv'
    grid_df.to_csv(grid_path, index=False)
    print(f'Saved grid search results to {grid_path}')

    tr_ids, val_ids = patient_holdout_val_split(
        train_ids,
        y=y_numeric,
        groups=train_groups,
        n_splits=5,
        val_fold=0,
        random_state=RANDOM_STATE,
    )
    print(f'Final train/val split: {len(tr_ids)} train samples, {len(val_ids)} val samples')

    set_seed()
    model = make_model(best_params, len(nodes), node_types, edge_index_by_network, edge_attr_by_network)
    class_weights = compute_class_weights(tr_ids, y_numeric).to(DEVICE)
    model, _, _ = train_with_early_stopping(
        model=model,
        dataset_by_id=dataset_by_id,
        tr_ids=tr_ids,
        val_ids=val_ids,
        lr=float(best_params['lr']),
        class_weights=class_weights,
        verbose=True,
    )

    test_metrics = evaluate_ids(model=model, dataset_by_id=dataset_by_id, ids=test_ids)
    print_test_metrics(test_metrics)
    cm_fig, per_class_path = save_test_metrics(test_metrics, 'gat', FIGURES_DIR, TABLES_DIR)

    attention = extract_attention(model=model, dataset_by_id=dataset_by_id, test_ids=test_ids, nodes=nodes)
    attention_path = TABLES_DIR / 'gat_attention.csv'
    attention.to_csv(attention_path, index=False)

    branch_scores = extract_network_branch_scores(model=model, dataset_by_id=dataset_by_id, ids=test_ids)
    branch_scores_path = TABLES_DIR / 'gat_network_branch_scores.csv'
    branch_scores.to_csv(branch_scores_path, index=False)

    branch_summary = summarize_network_branch_scores(branch_scores)
    branch_summary_path = TABLES_DIR / 'gat_network_branch_summary.csv'
    branch_summary.to_csv(branch_summary_path, index=False)

    model_path = MODELS_DIR / 'gat_baseline.pt'
    torch.save({
        'state_dict': model.state_dict(),
        'nodes': nodes,
        'class_names': CLASS_NAMES,
        'network_names': NETWORK_NAMES,
        'best_params': best_params,
    }, model_path)

    comparison_path = TABLES_DIR / 'model_comparison.csv'
    update_comparison_csv(
        comparison_path,
        test_metrics['accuracy'],
        test_metrics['macro_f1'],
        test_metrics['per_class_auc'],
    )

    print('Saved artifacts:')
    print(f'- {model_path}')
    print(f'- {attention_path}')
    print(f'- {branch_scores_path}')
    print(f'- {branch_summary_path}')
    print(f'- {cm_fig}')
    print(f'- {per_class_path}')
    print(f'- {comparison_path}')
    print(f'- {grid_path}')


if __name__ == '__main__':
    main()
