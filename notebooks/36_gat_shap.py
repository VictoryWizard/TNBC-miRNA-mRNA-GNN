#!/usr/bin/env python3
"""SHAP feature attribution for the multi-network GAT (seed-averaged).

Motivation (reviewer / professor ask):
  The manuscript compares GAT *attention* rankings against XGBoost *SHAP*
  rankings. Those are two different attribution methods, so any "the models
  disagree on top features" claim is confounded by the method difference. This
  script computes SHAP values for the GAT itself, so the paper can compare:

    * GAT-SHAP    vs  XGBoost-SHAP   (same method -> a fair model-vs-model test)
    * GAT-SHAP    vs  GAT-attention  (same model -> does attention track SHAP?)

Seed treatment (to match the paper):
  The manuscript reports GAT attention averaged across 20 retraining seeds "for
  stability." To keep the two attributions on equal footing, GAT-SHAP is computed
  the same way: the GAT is retrained across ``N_SEEDS`` seeds and per-feature
  mean|SHAP| is averaged across seeds (mean and SD reported).

Method:
  Primary  = shap.GradientExplainer (expected gradients; principled for a
             differentiable network, fast on GPU).
  Fallback = shap.KernelExplainer (model-agnostic) if GradientExplainer raises;
             the explainer actually used is recorded in the output CSV
             (``shap_method`` column) so the choice is auditable.

Outputs (results/tables/):
  gat_shap_importance.csv        per-node mean|SHAP| (mean +/- SD across seeds)
  gat_shap_comparison.csv        top-25 Jaccard overlaps between the three rankings
  gat_shap_report.txt            plain-text summary for the manuscript

Run AFTER a GPU runtime is selected. Requires: shap, torch, torch_geometric.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

_NOTEBOOK_DIR = Path(__file__).resolve().parent
if str(_NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOK_DIR))


def _load_module(name: str, filename: str):
    path = _NOTEBOOK_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gat = _load_module('gat_baseline', '05_gat_baseline.py')

PROCESSED_DIR = gat.PROCESSED_DIR
TABLES_DIR = gat.TABLES_DIR
MODELS_DIR = gat.MODELS_DIR
DEVICE = gat.DEVICE
CLASS_NAMES = gat.CLASS_NAMES
RANDOM_STATE = gat.RANDOM_STATE

# --- configuration -----------------------------------------------------------
N_SEEDS = 20            # match the paper's 20-seed attention averaging
N_BACKGROUND = 64       # training samples used as the SHAP reference distribution
GRADIENT_NSAMPLES = 100  # expected-gradient samples per explained instance
KERNEL_NSAMPLES = 200   # only used if GradientExplainer fails
TOP_K = 25              # top-K used for ranking-overlap (Jaccard) comparisons
XGB_IMPORTANCE_PATH = TABLES_DIR / 'xgboost_importance.csv'      # feature, mean_abs_shap
GAT_ATTENTION_NODES_PATH = TABLES_DIR / 'gat_attention_stability_graphml.csv'  # nb29 20-seed (node/molecule, mean_attention)
BEST_PARAMS_FALLBACK = {'num_heads': 2, 'hidden_size': 32, 'dropout': 0.2, 'lr': 0.001}


class GATProbWrapper(torch.nn.Module):
    """Expose the GAT as f(x) -> class probabilities for SHAP.

    SHAP passes a 2-D tensor (batch, num_nodes); the GAT expects one expression
    value per node, so the input is reshaped to (batch, num_nodes, 1). Only the
    final classifier head (not the auxiliary branch logits) is returned.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x3 = x.reshape(x.size(0), -1, 1)
        logits, _ = self.model(x3)
        return F.softmax(logits, dim=-1)


def load_best_params() -> Dict[str, float]:
    """Best GAT hyperparameters: prefer the saved checkpoint, then grid CSV, then paper defaults."""
    ckpt_path = MODELS_DIR / 'gat_baseline.pt'
    if ckpt_path.exists():
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            params = ckpt.get('best_params')
            if params:
                print(f'Loaded best_params from {ckpt_path}: {params}')
                return dict(params)
        except Exception as exc:  # noqa: BLE001
            print(f'Could not read best_params from checkpoint ({exc}); trying grid CSV.')

    grid_path = TABLES_DIR / 'gat_grid_search.csv'
    if grid_path.exists():
        grid = pd.read_csv(grid_path)
        if not grid.empty:
            top = grid.sort_values('mean_val_macro_f1', ascending=False).iloc[0]
            params = {k: top[k] for k in ('num_heads', 'hidden_size', 'dropout', 'lr') if k in top}
            if params:
                print(f'Loaded best_params from {grid_path}: {params}')
                return params

    print(f'Falling back to paper default params: {BEST_PARAMS_FALLBACK}')
    return dict(BEST_PARAMS_FALLBACK)


def build_context() -> Dict:
    x_all, y_all, train_ids, test_ids = gat.load_data()
    nodes = x_all.columns.tolist()
    train_mirna_cols = pd.read_csv(PROCESSED_DIR / 'train_mirna.csv', index_col=0).columns.tolist()
    node_types = gat.infer_node_types(nodes, train_mirna_cols)
    edge_index_by_network, edge_attr_by_network = gat.load_graphs(nodes)

    dataset_by_id, train_ids, test_ids, y_numeric = gat.build_dataset(
        x_all=x_all, y_all=y_all, train_ids=train_ids, test_ids=test_ids, nodes=nodes,
    )
    dataset_by_id = {sid: data.to(DEVICE) for sid, data in dataset_by_id.items()}
    node_types = node_types.to(DEVICE)
    edge_index_by_network = {k: v.to(DEVICE) for k, v in edge_index_by_network.items()}
    edge_attr_by_network = {k: v.to(DEVICE) for k, v in edge_attr_by_network.items()}

    mirna_set = set(train_mirna_cols)
    modality = ['miRNA' if n in mirna_set else 'mRNA' for n in nodes]

    return {
        'nodes': nodes,
        'modality': modality,
        'node_types': node_types,
        'edge_index_by_network': edge_index_by_network,
        'edge_attr_by_network': edge_attr_by_network,
        'dataset_by_id': dataset_by_id,
        'train_ids': train_ids,
        'test_ids': test_ids,
        'y_numeric': y_numeric,
    }


def stack_x(dataset_by_id: Dict, ids: Sequence[str]) -> torch.Tensor:
    """(n, num_nodes) expression matrix for the given sample ids."""
    return torch.stack([dataset_by_id[sid].x.reshape(-1) for sid in ids], dim=0).to(DEVICE)


def train_one_seed(ctx: Dict, best_params: Dict, seed: int) -> torch.nn.Module:
    tr_ids, val_ids = gat.patient_holdout_val_split(
        ctx['train_ids'], y=ctx['y_numeric'],
        groups=gat.load_patient_groups(ctx['train_ids']),
        n_splits=5, val_fold=0, random_state=RANDOM_STATE,
    )
    gat.set_seed(seed)
    model = gat.make_model(
        best_params, len(ctx['nodes']), ctx['node_types'],
        ctx['edge_index_by_network'], ctx['edge_attr_by_network'],
    )
    class_weights = gat.compute_class_weights(tr_ids, ctx['y_numeric']).to(DEVICE)
    model, _, _ = gat.train_with_early_stopping(
        model=model, dataset_by_id=ctx['dataset_by_id'],
        tr_ids=tr_ids, val_ids=val_ids,
        lr=float(best_params['lr']), class_weights=class_weights, verbose=False,
    )
    model.eval()
    return model


def _mean_abs_over_classes(shap_values, n_nodes: int) -> np.ndarray:
    """Reduce a SHAP result to per-node mean|SHAP| averaged over classes and samples.

    Handles both the list-of-arrays layout (older SHAP: one (n, nodes) array per
    class) and the stacked-array layout (newer SHAP: (n, nodes, n_classes)).
    """
    if isinstance(shap_values, list):
        stacked = np.stack([np.asarray(sv).reshape(-1, n_nodes) for sv in shap_values], axis=0)
        # stacked: (n_classes, n_samples, n_nodes)
        return np.abs(stacked).mean(axis=(0, 1))
    arr = np.asarray(shap_values)
    if arr.ndim == 3:  # (n_samples, n_nodes, n_classes)
        return np.abs(arr).mean(axis=(0, 2))
    if arr.ndim == 2:  # (n_samples, n_nodes)
        return np.abs(arr).mean(axis=0)
    raise ValueError(f'Unexpected SHAP output shape: {arr.shape}')


def shap_for_model(
    wrapper: torch.nn.Module,
    background: torch.Tensor,
    test_x: torch.Tensor,
    n_nodes: int,
    method: str,
) -> Tuple[np.ndarray, str]:
    """Per-node mean|SHAP| for one trained model. Returns (values, method_used)."""
    import shap

    if method != 'kernel':
        try:
            explainer = shap.GradientExplainer(wrapper, background)
            shap_values = explainer.shap_values(test_x, nsamples=GRADIENT_NSAMPLES)
            return _mean_abs_over_classes(shap_values, n_nodes), 'gradient'
        except Exception as exc:  # noqa: BLE001
            print(f'  GradientExplainer failed ({exc}); switching to KernelExplainer.')
            method = 'kernel'

    # Model-agnostic fallback.
    def predict_fn(x_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            xt = torch.tensor(x_np, dtype=torch.float32, device=DEVICE)
            return wrapper(xt).cpu().numpy()

    bg_np = shap.kmeans(background.cpu().numpy(), min(25, background.shape[0]))
    explainer = shap.KernelExplainer(predict_fn, bg_np)
    shap_values = explainer.shap_values(test_x.cpu().numpy(), nsamples=KERNEL_NSAMPLES)
    return _mean_abs_over_classes(shap_values, n_nodes), 'kernel'


def compute_seed_averaged_shap(ctx: Dict, best_params: Dict) -> Tuple[pd.DataFrame, str]:
    nodes = ctx['nodes']
    n_nodes = len(nodes)

    rng = np.random.default_rng(RANDOM_STATE)
    bg_ids = list(ctx['train_ids'])
    if len(bg_ids) > N_BACKGROUND:
        bg_ids = list(rng.choice(bg_ids, size=N_BACKGROUND, replace=False))
    background = stack_x(ctx['dataset_by_id'], bg_ids)
    test_x = stack_x(ctx['dataset_by_id'], ctx['test_ids'])

    per_seed = np.zeros((N_SEEDS, n_nodes), dtype=float)
    method_used = 'gradient'
    for i in range(N_SEEDS):
        seed = RANDOM_STATE + i
        print(f'[SHAP] seed {i + 1}/{N_SEEDS} (retraining GAT + explaining)...')
        model = train_one_seed(ctx, best_params, seed)
        wrapper = GATProbWrapper(model).to(DEVICE).eval()
        values, method_used = shap_for_model(wrapper, background, test_x, n_nodes, method_used)
        per_seed[i] = values
        del model, wrapper
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    mean_abs = per_seed.mean(axis=0)
    sd_abs = per_seed.std(axis=0)
    df = pd.DataFrame({
        'feature': nodes,
        'modality': ctx['modality'],
        'mean_abs_shap': mean_abs,
        'sd_abs_shap': sd_abs,
        'n_seeds': N_SEEDS,
        'shap_method': method_used,
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    df['gat_shap_rank'] = np.arange(1, len(df) + 1)
    return df, method_used


# --- ranking comparison ------------------------------------------------------
def _is_mirna(name: str) -> bool:
    return str(name).lower().startswith('hsa-')


def _top_set(series: pd.Series, k: int) -> set:
    return set(series.head(k).tolist())


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return float('nan')
    return len(a & b) / len(a | b)


def build_comparison(gat_shap: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Top-K Jaccard overlaps: GAT-SHAP vs XGBoost-SHAP vs GAT-attention."""
    lines: List[str] = []

    def ranked_features(df: pd.DataFrame, score_col: str) -> pd.Series:
        return df.sort_values(score_col, ascending=False)['feature'].reset_index(drop=True)

    gat_shap = gat_shap.copy()
    rankings: Dict[str, pd.Series] = {'gat_shap': ranked_features(gat_shap, 'mean_abs_shap')}

    if XGB_IMPORTANCE_PATH.exists():
        xgb = pd.read_csv(XGB_IMPORTANCE_PATH).rename(columns={'feature': 'feature'})
        rankings['xgb_shap'] = ranked_features(xgb, 'mean_abs_shap')
    else:
        lines.append(f'WARNING: {XGB_IMPORTANCE_PATH} not found; XGBoost-SHAP comparison skipped.')

    if GAT_ATTENTION_NODES_PATH.exists():
        att = pd.read_csv(GAT_ATTENTION_NODES_PATH).rename(columns={'molecule': 'feature'})
        score_col = 'mean_attention' if 'mean_attention' in att.columns else ('gat_attention_score' if 'gat_attention_score' in att.columns else att.columns[2])
        rankings['gat_attention'] = ranked_features(att, score_col)
    else:
        lines.append(f'WARNING: {GAT_ATTENTION_NODES_PATH} not found; attention comparison skipped.')

    pairs = [('gat_shap', 'xgb_shap'), ('gat_shap', 'gat_attention'), ('xgb_shap', 'gat_attention')]
    rows = []
    for a, b in pairs:
        if a not in rankings or b not in rankings:
            continue
        sa, sb = rankings[a], rankings[b]
        a_mir = sa[sa.map(_is_mirna)].reset_index(drop=True)
        b_mir = sb[sb.map(_is_mirna)].reset_index(drop=True)
        a_mr = sa[~sa.map(_is_mirna)].reset_index(drop=True)
        b_mr = sb[~sb.map(_is_mirna)].reset_index(drop=True)
        rows.append({
            'comparison': f'{a}_vs_{b}',
            'top_k': TOP_K,
            'jaccard_overall': jaccard(_top_set(sa, TOP_K), _top_set(sb, TOP_K)),
            'n_shared_overall': len(_top_set(sa, TOP_K) & _top_set(sb, TOP_K)),
            'jaccard_mirna': jaccard(_top_set(a_mir, TOP_K), _top_set(b_mir, TOP_K)),
            'jaccard_mrna': jaccard(_top_set(a_mr, TOP_K), _top_set(b_mr, TOP_K)),
        })
    comparison = pd.DataFrame(rows)

    lines.append('GAT-SHAP vs XGBoost-SHAP vs GAT-attention -- top-{} ranking overlap'.format(TOP_K))
    lines.append('')
    for _, r in comparison.iterrows():
        lines.append(
            f"{r['comparison']:<28} Jaccard overall={r['jaccard_overall']:.3f} "
            f"(shared {int(r['n_shared_overall'])})  miRNA={r['jaccard_mirna']:.3f}  "
            f"mRNA={r['jaccard_mrna']:.3f}"
        )
    lines.append('')
    lines.append('Top-10 GAT-SHAP features:')
    for _, r in gat_shap.head(10).iterrows():
        lines.append(f"  {r['gat_shap_rank']:>2}. {r['feature']:<18} ({r['modality']})  "
                     f"mean|SHAP|={r['mean_abs_shap']:.5f} +/- {r['sd_abs_shap']:.5f}")
    return comparison, '\n'.join(lines)


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    print(f'Using device: {DEVICE}')
    if DEVICE.type != 'cuda':
        print('WARNING: no GPU detected -- 20-seed SHAP will be slow. '
              'Set the Colab runtime to GPU (Runtime -> Change runtime type -> GPU).')

    best_params = load_best_params()
    ctx = build_context()
    print(f"Nodes: {len(ctx['nodes'])} | train: {len(ctx['train_ids'])} | test: {len(ctx['test_ids'])}")

    gat_shap, method_used = compute_seed_averaged_shap(ctx, best_params)
    importance_path = TABLES_DIR / 'gat_shap_importance.csv'
    gat_shap.to_csv(importance_path, index=False)
    print(f'Saved {importance_path} (method={method_used})')

    comparison, report = build_comparison(gat_shap)
    comparison_path = TABLES_DIR / 'gat_shap_comparison.csv'
    comparison.to_csv(comparison_path, index=False)
    report_path = TABLES_DIR / 'gat_shap_report.txt'
    report_path.write_text(report + '\n', encoding='utf-8')

    print('\n' + report)
    print('\nSaved artifacts:')
    print(f'- {importance_path}')
    print(f'- {comparison_path}')
    print(f'- {report_path}')


if __name__ == '__main__':
    main()
