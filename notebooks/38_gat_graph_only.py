#!/usr/bin/env python3
"""Graph-only GAT control (reviewer request).

The multi-network GAT includes a direct expression-only MLP pathway alongside the
graph branches, so it can partly route around the graph. This control removes that
direct bypass (the expression contribution is zeroed) so the model must classify
from the graph branches, whose node features are still expression values. Node
features therefore still reach the model through the graph; only the *direct*
expression shortcut is removed.

Interpretation:
  - If graph-only performance collapses far below the full GAT, the full GAT's
    accuracy came largely from the expression bypass (edges/graph carry little).
  - If graph-only stays close to the full GAT, the node features carry the signal
    through the graph, consistent with the edge ablation (the specific edges do
    not matter, but expression-as-node-features does).

Run on a GPU runtime, on the 118 build. ~1 hour for 20 seeds.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_ND = Path(__file__).resolve().parent
if str(_ND) not in sys.path:
    sys.path.insert(0, str(_ND))


def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, _ND / fn)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


gat = _load('gat_baseline', '05_gat_baseline.py')
PROCESSED_DIR = gat.PROCESSED_DIR
TABLES_DIR = gat.TABLES_DIR
MODELS_DIR = gat.MODELS_DIR
DEVICE = gat.DEVICE
RANDOM_STATE = gat.RANDOM_STATE
N_SEEDS = 20
BEST_PARAMS_FALLBACK = {'num_heads': 2, 'hidden_size': 32, 'dropout': 0.2, 'lr': 0.001}


class GraphOnlyGAT(gat.MultiNetworkGAT):
    """Same architecture, but the direct expression-MLP contribution is zeroed."""

    def forward(self, x: torch.Tensor):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x_in = self._node_input(x)
        graph_parts, branch_logits = [], []
        for network in self.network_names:
            pooled, logits = self._network_embedding(x_in, network)
            graph_parts.append(pooled)
            branch_logits.append(logits)
        expr_dim = self.expression_head[1].out_features
        expression_features = torch.zeros(x.size(0), expr_dim, device=x.device)  # bypass removed
        combined = torch.cat([expression_features, *graph_parts, *branch_logits], dim=1)
        logits = self.classifier(combined)
        return logits, torch.stack(branch_logits, dim=1)


def best_params() -> dict:
    ckpt = MODELS_DIR / 'gat_baseline.pt'
    if ckpt.exists():
        try:
            c = torch.load(ckpt, map_location='cpu', weights_only=False)
            if c.get('best_params'):
                return dict(c['best_params'])
        except Exception:  # noqa: BLE001
            pass
    grid = TABLES_DIR / 'gat_grid_search.csv'
    if grid.exists():
        df = pd.read_csv(grid)
        if len(df):
            t = df.sort_values('mean_val_macro_f1', ascending=False).iloc[0]
            params = {k: t[k] for k in ('num_heads', 'hidden_size', 'dropout', 'lr') if k in t}
            if params:
                return params
    return dict(BEST_PARAMS_FALLBACK)


def make_graph_only(params, num_nodes, node_types, eidx, eattr) -> GraphOnlyGAT:
    return GraphOnlyGAT(
        num_nodes=num_nodes, node_types=node_types,
        edge_index_by_network=eidx, edge_attr_by_network=eattr,
        num_heads=int(params['num_heads']), hidden_size=int(params['hidden_size']),
        dropout=float(params['dropout']), num_layers=int(params.get('num_layers', 2)),
    ).to(DEVICE)


def main() -> None:
    print(f'Using device: {DEVICE}')
    x_all, y_all, train_ids, test_ids = gat.load_data()
    nodes = x_all.columns.tolist()
    mir = pd.read_csv(PROCESSED_DIR / 'train_mirna.csv', index_col=0).columns.tolist()
    node_types = gat.infer_node_types(nodes, mir).to(DEVICE)
    eidx, eattr = gat.load_graphs(nodes)
    eidx = {k: v.to(DEVICE) for k, v in eidx.items()}
    eattr = {k: v.to(DEVICE) for k, v in eattr.items()}
    ds, train_ids, test_ids, ynum = gat.build_dataset(
        x_all=x_all, y_all=y_all, train_ids=train_ids, test_ids=test_ids, nodes=nodes)
    ds = {s: dd.to(DEVICE) for s, dd in ds.items()}
    assert len(nodes) == 118, f'expected the 118 build, got {len(nodes)} nodes'

    bp = best_params()
    print('best_params:', bp)
    groups = gat.load_patient_groups(train_ids)
    tr, val = gat.patient_holdout_val_split(
        train_ids, y=ynum, groups=groups, n_splits=5, val_fold=0, random_state=RANDOM_STATE)

    scores = []
    for i in range(N_SEEDS):
        gat.set_seed(RANDOM_STATE + i)
        m = make_graph_only(bp, len(nodes), node_types, eidx, eattr)
        cw = gat.compute_class_weights(tr, ynum).to(DEVICE)
        m, _, _ = gat.train_with_early_stopping(
            model=m, dataset_by_id=ds, tr_ids=tr, val_ids=val,
            lr=float(bp['lr']), class_weights=cw, verbose=False)
        met = gat.evaluate_ids(model=m, dataset_by_id=ds, ids=test_ids)
        scores.append(float(met['macro_f1']))
        print(f'  seed {i + 1}/{N_SEEDS} macro_f1={met["macro_f1"]:.4f}', flush=True)

    a = np.array(scores)
    print(f'\nGRAPH-ONLY GAT: macro F1 {a.mean():.4f} +/- {a.std():.4f} '
          f'(min {a.min():.3f}, max {a.max():.3f})')
    print('Compare: full GAT 0.929 +/- 0.024 ; XGBoost 0.939')
    out = TABLES_DIR / 'gat_graph_only_seed_distribution.csv'
    pd.DataFrame({'seed': list(range(N_SEEDS)), 'macro_f1': scores}).to_csv(out, index=False)
    print('saved', out)


if __name__ == '__main__':
    main()
