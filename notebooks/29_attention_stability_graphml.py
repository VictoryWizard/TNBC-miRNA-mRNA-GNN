#!/usr/bin/env python3
"""Multi-seed GAT attention stability on the top-150 graphml networks (no kNN).

Retrains the
baseline multi-network GAT (notebooks/05_gat_baseline.py) over N_RESEEDS random
seeds on the canonical top-150 networks (notebooks/03_network_construction.py),
extracts test-set attention each time, and reports how STABLE each node's and
each edge's attention is across seeds (mean, SD, coefficient of variation).

Why this matters: a feature whose attention rank swings between seeds is not a
trustworthy biomarker readout. CV = SD / mean; high-CV entities are flagged.
NOTE: CV is inherently inflated for near-zero-mean attention, so the report also
keeps the raw mean and SD — rank instability among LOW-attention nodes is less
meaningful than among high-attention ones.

Hyperparameters are locked once (selected on the real network via the baseline's
patient-grouped CV) so only the random seed varies across retrains; the
train/val split is held fixed (seed-independent) so only initialization and
optimization stochasticity move.

Outputs:
  results/tables/gat_attention_stability_graphml.csv        (per-node)
  results/tables/gat_attention_stability_graphml_edges.csv  (per-edge)

Run from repo root (GPU recommended; N_RESEEDS full GAT trainings):
  python notebooks/29_attention_stability_graphml.py
"""

import sys
from collections import defaultdict
from pathlib import Path

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

N_RESEEDS = 20
TOP_UNSTABLE = 15
MASTER_SEED = gb.RANDOM_STATE
TABLES_DIR = gb.TABLES_DIR


def full_attention(model, dataset_by_id, test_ids, nodes):
    """Per-edge mean attention over test samples, UNTRUNCATED.

    Returns {(node_u, node_v): mean_attention} aggregated across networks and
    test samples (undirected pairs averaged), plus a per-node aggregate
    {node: mean incident-edge attention}.
    """
    pair_scores = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for network in model.network_names:
            for sample_id in test_ids:
                data = dataset_by_id[sample_id]
                _, _, att_edge_index, alpha = model.attention_for_network(data.x, network)
                att_edge_index = att_edge_index.cpu().numpy()
                alpha = alpha.mean(dim=-1).cpu().numpy()
                for (u_idx, v_idx), w in zip(att_edge_index.T, alpha):
                    if u_idx == v_idx:
                        continue
                    u = str(nodes[int(u_idx)]); v = str(nodes[int(v_idx)])
                    pair_scores[tuple(sorted((u, v)))].append(float(w))

    edge_attn = {k: float(np.mean(v)) for k, v in pair_scores.items()}
    node_inc = defaultdict(list)
    for (u, v), w in edge_attn.items():
        node_inc[u].append(w); node_inc[v].append(w)
    node_attn = {n: float(np.mean(w)) for n, w in node_inc.items()}
    return edge_attn, node_attn


def cv_table(per_seed_values, key_name):
    """Build a mean/SD/CV table across seeds for a dict-of-dicts {seed: {key: val}}."""
    all_keys = set()
    for d in per_seed_values:
        all_keys.update(d.keys())
    rows = []
    for k in all_keys:
        vals = np.array([d.get(k, 0.0) for d in per_seed_values], dtype=float)
        mean = float(vals.mean())
        sd = float(vals.std())
        cv = float(sd / mean) if mean > 1e-12 else float('nan')
        row = {key_name: k if isinstance(k, str) else k,
               'mean_attention': mean, 'sd_attention': sd,
               'cv_attention': cv, 'n_seeds_present': int((vals > 0).sum())}
        rows.append(row)
    df = pd.DataFrame(rows).sort_values('mean_attention', ascending=False).reset_index(drop=True)
    return df


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    print(f'Using device: {gb.DEVICE}; {N_RESEEDS} retrain seeds')

    x_all, y_all, train_ids, test_ids = gb.load_data()
    nodes = x_all.columns.tolist()
    mirna_cols = pd.read_csv(gb.PROCESSED_DIR / 'train_mirna.csv', index_col=0).columns.tolist()
    mirna_set = set(mirna_cols)
    node_types = gb.infer_node_types(nodes, mirna_cols).to(gb.DEVICE)

    eix, eattr = gb.load_graphs(nodes)
    eix = {k: v.to(gb.DEVICE) for k, v in eix.items()}
    eattr = {k: v.to(gb.DEVICE) for k, v in eattr.items()}

    dataset_by_id, train_ids, test_ids, y_numeric = gb.build_dataset(
        x_all=x_all, y_all=y_all, train_ids=train_ids, test_ids=test_ids, nodes=nodes)
    dataset_by_id = {sid: d.to(gb.DEVICE) for sid, d in dataset_by_id.items()}

    train_groups = gb.load_patient_groups(train_ids)
    best_params, _ = gb.grid_search_cv(
        dataset_by_id=dataset_by_id, train_ids=train_ids, y_numeric=y_numeric,
        train_groups=train_groups, num_nodes=len(nodes), node_types=node_types,
        edge_index_by_network=eix, edge_attr_by_network=eattr)
    # Fixed (seed-independent) train/val split so only the retrain seed varies.
    tr_ids, val_ids = gb.patient_holdout_val_split(
        train_ids, y=y_numeric, groups=train_groups, n_splits=5, val_fold=0,
        random_state=MASTER_SEED)

    edge_seed_vals, node_seed_vals = [], []
    for s in range(N_RESEEDS):
        seed = MASTER_SEED + s
        gb.set_seed(seed)
        model = gb.make_model(best_params, len(nodes), node_types, eix, eattr)
        class_weights = gb.compute_class_weights(tr_ids, y_numeric).to(gb.DEVICE)
        model, _, _ = gb.train_with_early_stopping(
            model=model, dataset_by_id=dataset_by_id, tr_ids=tr_ids, val_ids=val_ids,
            lr=float(best_params['lr']), class_weights=class_weights, verbose=False)
        e_attn, n_attn = full_attention(model, dataset_by_id, test_ids, nodes)
        edge_seed_vals.append(e_attn); node_seed_vals.append(n_attn)
        print(f'  seed {s + 1}/{N_RESEEDS} done')

    # Per-node table
    node_df = cv_table(node_seed_vals, 'node')
    node_df['modality'] = node_df['node'].apply(lambda n: 'miRNA' if n in mirna_set else 'mRNA')
    # Columns for compatibility with 24_figure2_feature_importance.py
    node_df['entity_type'] = 'node'
    node_df['molecule'] = node_df['node']
    node_path = TABLES_DIR / 'gat_attention_stability_graphml.csv'
    node_df.to_csv(node_path, index=False)

    # Per-edge table (store node_u/node_v separately)
    edge_df = cv_table(edge_seed_vals, 'edge')
    edge_df['node_u'] = edge_df['edge'].apply(lambda t: t[0])
    edge_df['node_v'] = edge_df['edge'].apply(lambda t: t[1])
    edge_df = edge_df.drop(columns=['edge'])
    edge_path = TABLES_DIR / 'gat_attention_stability_graphml_edges.csv'
    edge_df.to_csv(edge_path, index=False)

    # Report: steadiest and least-stable among the top-attention nodes
    leading = node_df.head(30).dropna(subset=['cv_attention'])
    steadiest = leading.nsmallest(TOP_UNSTABLE, 'cv_attention')
    unstable = leading.nlargest(TOP_UNSTABLE, 'cv_attention')
    print('\nSteadiest leading nodes (low CV):')
    print(steadiest[['node', 'modality', 'mean_attention', 'cv_attention']].to_string(index=False))
    print('\nLeast stable leading nodes (high CV):')
    print(unstable[['node', 'modality', 'mean_attention', 'cv_attention']].to_string(index=False))
    print(f'\nSaved {node_path}\nSaved {edge_path}')


if __name__ == '__main__':
    main()
