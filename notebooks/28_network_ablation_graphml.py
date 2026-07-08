#!/usr/bin/env python3
"""Network-structure ablation on the top-150 graphml networks (no kNN).

significance.py. Tests whether the GAT's graph EDGES carry information beyond the
node expression values, using the canonical top-150 networks from
notebooks/03_network_construction.py and the baseline GAT from
notebooks/05_gat_baseline.py.

Conditions (all share one locked hyperparameter set, selected once on the real
network via the baseline's patient-grouped CV):
  * real        : the true tissue networks
  * no_edges    : every edge removed (self-loops only) -> expression-only GAT
  * permuted    : degree-preserving random rewiring of every network (N_PERM
                  independent permutations)

Statistics:
  * The real condition is trained over N_REAL_SEEDS seeds; mean/SD reported so
    its run-to-run variance is not assumed to be zero.
  * Each permutation is a single training run (matching the permuted null's
    procedure). Empirical p-value uses the unbiased estimator
    p = (1 + #{permuted macro_F1 >= real_mean}) / (1 + N_PERM)   (Phipson &
    Smyth 2010); this can never be exactly 0.

Outputs:
  results/tables/network_ablation_graphml.csv
  results/tables/network_ablation_graphml_permutations.csv
  results/figures/network_ablation_graphml.png

Run from repo root (GPU recommended; N_PERM full runs are slow):
  python notebooks/28_network_ablation_graphml.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

_NOTEBOOK_DIR = Path(__file__).resolve().parent
if str(_NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOK_DIR))

import importlib.util as _ilu  # load 05_gat_baseline.py (name can't start with a digit)
_spec = _ilu.spec_from_file_location('gat_baseline', str(_NOTEBOOK_DIR / '05_gat_baseline.py'))
gb = _ilu.module_from_spec(_spec)
sys.modules['gat_baseline'] = gb
_spec.loader.exec_module(gb)

# --- Tunables ---------------------------------------------------------------
N_REAL_SEEDS = 5       # seeds for the real-condition mean/SD
N_PERM = 1000          # permutations for the null (reviewer asked for >=1000).
                       # Each is a full GAT training (~6-10 h total on a T4), so the
                       # permuted loop is CHECKPOINTED per-permutation and resumes
                       # automatically if the run is interrupted (see main()).
MASTER_SEED = gb.RANDOM_STATE
TABLES_DIR = gb.TABLES_DIR
FIGURES_DIR = gb.FIGURES_DIR


def self_loop_edges(num_nodes, device):
    """Edge tensors with only self-loops (used for the no-edges condition)."""
    idx = torch.arange(num_nodes, device=device)
    edge_index = torch.stack([idx, idx], dim=0)
    edge_attr = torch.ones((num_nodes, 1), dtype=torch.float32, device=device)
    return edge_index, edge_attr


def permute_edges_degree_preserving(edge_index, num_nodes, rng):
    """Degree-preserving rewiring via double-edge swaps on the undirected graph.

    edge_index is the symmetric (both-direction) tensor used by the model. We
    rebuild an undirected edge list, swap endpoints, and re-symmetrize, keeping
    each node's degree fixed (so only topology, not degree, changes).
    """
    ei = edge_index.cpu().numpy()
    # collapse to undirected unique pairs
    pairs = set()
    for s, t in zip(ei[0], ei[1]):
        if s != t:
            pairs.add((min(int(s), int(t)), max(int(s), int(t))))
    edges = [list(p) for p in pairs]
    m = len(edges)
    if m < 2:
        return edge_index
    n_swaps = 10 * m
    edge_set = set(map(tuple, edges))
    for _ in range(n_swaps):
        i, j = rng.integers(0, m), rng.integers(0, m)
        if i == j:
            continue
        a, b = edges[i]
        c, d = edges[j]
        nodes = {a, b, c, d}
        if len(nodes) < 4:
            continue
        new1, new2 = (min(a, d), max(a, d)), (min(c, b), max(c, b))
        if new1 in edge_set or new2 in edge_set:
            continue
        edge_set.discard((a, b)); edge_set.discard((c, d))
        edge_set.add(new1); edge_set.add(new2)
        edges[i] = list(new1); edges[j] = list(new2)
    # re-symmetrize
    src, tgt, attr = [], [], []
    for a, b in edges:
        src.extend([a, b]); tgt.extend([b, a]); attr.extend([[1.0], [1.0]])
    device = edge_index.device
    return (torch.tensor([src, tgt], dtype=torch.long, device=device),
            torch.tensor(attr, dtype=torch.float32, device=device))


def train_eval(best_params, num_nodes, node_types, edge_index_by_network,
               edge_attr_by_network, dataset_by_id, tr_ids, val_ids, test_ids,
               y_numeric, seed):
    gb.set_seed(seed)
    model = gb.make_model(best_params, num_nodes, node_types,
                          edge_index_by_network, edge_attr_by_network)
    class_weights = gb.compute_class_weights(tr_ids, y_numeric).to(gb.DEVICE)
    model, _, _ = gb.train_with_early_stopping(
        model=model, dataset_by_id=dataset_by_id, tr_ids=tr_ids, val_ids=val_ids,
        lr=float(best_params['lr']), class_weights=class_weights, verbose=False,
    )
    m = gb.evaluate_ids(model=model, dataset_by_id=dataset_by_id, ids=test_ids)
    return float(m['accuracy']), float(m['macro_f1'])


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f'Using device: {gb.DEVICE}')

    x_all, y_all, train_ids, test_ids = gb.load_data()
    nodes = x_all.columns.tolist()
    mirna_cols = pd.read_csv(gb.PROCESSED_DIR / 'train_mirna.csv', index_col=0).columns.tolist()
    node_types = gb.infer_node_types(nodes, mirna_cols).to(gb.DEVICE)

    real_eix, real_eattr = gb.load_graphs(nodes)
    real_eix = {k: v.to(gb.DEVICE) for k, v in real_eix.items()}
    real_eattr = {k: v.to(gb.DEVICE) for k, v in real_eattr.items()}

    dataset_by_id, train_ids, test_ids, y_numeric = gb.build_dataset(
        x_all=x_all, y_all=y_all, train_ids=train_ids, test_ids=test_ids, nodes=nodes)
    dataset_by_id = {sid: d.to(gb.DEVICE) for sid, d in dataset_by_id.items()}

    train_groups = gb.load_patient_groups(train_ids)
    # Lock hyperparameters once on the real network (paper's locked-HP design).
    # cache best_params so a resumed run skips the (slow) grid search
    params_path = TABLES_DIR / 'network_ablation_best_params.json'
    if params_path.exists():
        best_params = json.load(open(params_path)); print('Loaded cached best_params')
    else:
        best_params, _ = gb.grid_search_cv(
            dataset_by_id=dataset_by_id, train_ids=train_ids, y_numeric=y_numeric,
            train_groups=train_groups, num_nodes=len(nodes), node_types=node_types,
            edge_index_by_network=real_eix, edge_attr_by_network=real_eattr)
        json.dump({k: float(v) for k, v in best_params.items()}, open(params_path, 'w'))
    tr_ids, val_ids = gb.patient_holdout_val_split(
        train_ids, y=y_numeric, groups=train_groups, n_splits=5, val_fold=0,
        random_state=MASTER_SEED)

    nnodes = len(nodes)

    # --- real (multi-seed) ---
    real_runs = [train_eval(best_params, nnodes, node_types, real_eix, real_eattr,
                            dataset_by_id, tr_ids, val_ids, test_ids, y_numeric,
                            MASTER_SEED + s) for s in range(N_REAL_SEEDS)]
    real_acc = np.array([r[0] for r in real_runs])
    real_f1 = np.array([r[1] for r in real_runs])
    real_f1_mean = float(real_f1.mean())
    print(f'real: macro_f1 {real_f1_mean:.4f} +/- {real_f1.std():.4f} (n={N_REAL_SEEDS})')

    # --- no_edges (self-loops) ---
    sl = {k: self_loop_edges(nnodes, gb.DEVICE) for k in real_eix}
    ne_eix = {k: v[0] for k, v in sl.items()}
    ne_eattr = {k: v[1] for k, v in sl.items()}
    ne_acc, ne_f1 = train_eval(best_params, nnodes, node_types, ne_eix, ne_eattr,
                               dataset_by_id, tr_ids, val_ids, test_ids, y_numeric,
                               MASTER_SEED)
    print(f'no_edges: macro_f1 {ne_f1:.4f}')

    # --- permuted null (RESUMABLE: each permutation is checkpointed to disk, so an
    #     interrupted/recycled run resumes where it left off instead of restarting) ---
    perm_path = TABLES_DIR / 'network_ablation_graphml_permutations.csv'
    # If Google Drive is mounted, mirror the checkpoint there so the 1000-perm run
    # resumes even across a full runtime recycle (not just within a session).
    drive_dir = Path('/content/drive/MyDrive/TNBC_results')
    drive_perm = drive_dir / 'network_ablation_graphml_permutations.csv' if drive_dir.exists() else None
    if not perm_path.exists() and drive_perm is not None and drive_perm.exists():
        pd.read_csv(drive_perm).to_csv(perm_path, index=False)  # restore from Drive
        print('Restored permutation checkpoint from Drive')
    if perm_path.exists():
        perm_df = pd.read_csv(perm_path)
        done = set(int(p) for p in perm_df['permutation'].tolist())
        print(f'Resuming permuted null: {len(done)}/{N_PERM} already done')
    else:
        perm_df = pd.DataFrame(columns=['permutation', 'accuracy', 'macro_f1'])
        done = set()
    for p in range(N_PERM):
        if p in done:
            continue
        rng = np.random.default_rng(MASTER_SEED + 1000 + p)
        peix, peattr = {}, {}
        for k in real_eix:
            ei, ea = permute_edges_degree_preserving(real_eix[k], nnodes, rng)
            peix[k] = ei.to(gb.DEVICE); peattr[k] = ea.to(gb.DEVICE)
        acc, f1 = train_eval(best_params, nnodes, node_types, peix, peattr,
                             dataset_by_id, tr_ids, val_ids, test_ids, y_numeric,
                             MASTER_SEED + 1000 + p)
        perm_df = pd.concat([perm_df, pd.DataFrame(
            [{'permutation': p, 'accuracy': acc, 'macro_f1': f1}])], ignore_index=True)
        perm_df.to_csv(perm_path, index=False)   # checkpoint after every permutation
        if drive_perm is not None:
            perm_df.to_csv(drive_perm, index=False)  # mirror to Drive (recycle-proof)
        if (p + 1) % 25 == 0:
            print(f'  permutation {p + 1}/{N_PERM}')
    perm_df = perm_df.sort_values('permutation').reset_index(drop=True)

    n_ge = int((perm_df['macro_f1'] >= real_f1_mean).sum())
    p_value = (1 + n_ge) / (1 + N_PERM)   # unbiased; never 0

    summary = pd.DataFrame([
        {'condition': 'real', 'accuracy_mean': float(real_acc.mean()),
         'accuracy_std': float(real_acc.std()), 'macro_f1_mean': real_f1_mean,
         'macro_f1_std': float(real_f1.std()), 'n_runs': N_REAL_SEEDS},
        {'condition': 'no_edges', 'accuracy_mean': ne_acc, 'accuracy_std': 0.0,
         'macro_f1_mean': ne_f1, 'macro_f1_std': 0.0, 'n_runs': 1},
        {'condition': 'permuted', 'accuracy_mean': float(perm_df['accuracy'].mean()),
         'accuracy_std': float(perm_df['accuracy'].std()),
         'macro_f1_mean': float(perm_df['macro_f1'].mean()),
         'macro_f1_std': float(perm_df['macro_f1'].std()), 'n_runs': N_PERM},
    ])
    summary.to_csv(TABLES_DIR / 'network_ablation_graphml.csv', index=False)

    print('\nNetwork ablation (graphml top-150):')
    print(summary.to_string(index=False))
    print(f'\nEmpirical p (real vs permuted null): {p_value:.4f} '
          f'({n_ge} of {N_PERM} permutations >= real mean)')

    # figure
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(perm_df['macro_f1'], bins=30, color='lightgray', edgecolor='gray',
            label=f'permuted null (n={N_PERM})')
    ax.axvline(real_f1_mean, color='crimson', lw=2,
               label=f'real (mean {real_f1_mean:.3f})')
    ax.axvline(ne_f1, color='steelblue', lw=2, ls='--',
               label=f'no edges ({ne_f1:.3f})')
    ax.set_xlabel('Held-out macro F1'); ax.set_ylabel('Permutations')
    ax.set_title(f'Network ablation (top-150 graphml); empirical p = {p_value:.3f}')
    ax.legend()
    fig.tight_layout()
    out = FIGURES_DIR / 'network_ablation_graphml.png'
    fig.savefig(out, dpi=300); plt.close(fig)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
