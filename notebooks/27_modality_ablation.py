#!/usr/bin/env python3
"""Modality ablation for the multi-network GAT (miRNA-only / mRNA-only / both).

Question: does the GAT classify TNBC tissue better with only miRNA nodes, only
mRNA nodes, or the full bimodal network (both)?

Method: reuse the exact baseline GAT pipeline (notebooks/05_gat_baseline.py) but
restrict the graph NODE SET to a single modality for two of the three variants.
Because ``load_graphs`` only keeps an edge when BOTH endpoints are in the node
set, subsetting nodes to one modality automatically:
  * miRNA-only  -> keeps only miRNA-miRNA edges, miRNA node features
  * mRNA-only   -> keeps only mRNA-mRNA edges, mRNA node features
  * both        -> full network (miRNA-miRNA + mRNA-mRNA + miRNA-mRNA edges)

Everything else is held identical across variants: train/test split, patient-
grouped CV hyperparameter search, final patient-holdout val split, early
stopping on val macro-F1, and single held-out test evaluation. This isolates the
effect of node modality on performance.

Inputs : data/processed/* and results/{tissue}_network.graphml (from
         notebooks/03_network_construction.py).
Outputs: results/tables/modality_ablation.csv
         results/figures/modality_ablation.png

Run from repo root:  python notebooks/27_modality_ablation.py
"""

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


MODALITIES = ('both', 'mirna', 'mrna')
RESULTS_DIR = Path('results')
TABLES_DIR = RESULTS_DIR / 'tables'
FIGURES_DIR = RESULTS_DIR / 'figures'


def select_nodes(all_nodes, mirna_cols, modality):
    """Return the node subset for a modality variant, preserving order."""
    mirna_set = set(mirna_cols)
    if modality == 'mirna':
        return [n for n in all_nodes if n in mirna_set]
    if modality == 'mrna':
        return [n for n in all_nodes if n not in mirna_set]
    return list(all_nodes)  # 'both'


def run_variant(modality, x_all, y_all, train_ids_in, test_ids_in, mirna_cols):
    """Train + evaluate the GAT for one modality variant; return a metrics dict."""
    print('\n' + '=' * 60)
    print(f'MODALITY VARIANT: {modality}')
    print('=' * 60)
    gb.set_seed()

    nodes = select_nodes(x_all.columns.tolist(), mirna_cols, modality)
    if len(nodes) < 2:
        raise ValueError(f'{modality}: fewer than 2 nodes available.')

    node_types = gb.infer_node_types(nodes, mirna_cols)

    # Restrict feature matrix to the selected nodes so the expression head and
    # per-sample graph tensors match the node set.
    x_sub = x_all[nodes]

    # load_graphs keeps only edges whose endpoints are both in `nodes`.
    edge_index_by_network, edge_attr_by_network = gb.load_graphs(nodes)

    dataset_by_id, train_ids, test_ids, y_numeric = gb.build_dataset(
        x_all=x_sub, y_all=y_all,
        train_ids=train_ids_in, test_ids=test_ids_in, nodes=nodes,
    )
    dataset_by_id = {sid: d.to(gb.DEVICE) for sid, d in dataset_by_id.items()}
    node_types = node_types.to(gb.DEVICE)
    edge_index_by_network = {k: v.to(gb.DEVICE) for k, v in edge_index_by_network.items()}
    edge_attr_by_network = {k: v.to(gb.DEVICE) for k, v in edge_attr_by_network.items()}

    n_edges = {k: v.shape[1] // 2 for k, v in edge_index_by_network.items()}
    print(f'{modality}: {len(nodes)} nodes; edges/network (undirected): {n_edges}')

    train_groups = gb.load_patient_groups(train_ids)

    # Same patient-grouped CV hyperparameter search as the baseline.
    best_params, grid_df = gb.grid_search_cv(
        dataset_by_id=dataset_by_id,
        train_ids=train_ids,
        y_numeric=y_numeric,
        train_groups=train_groups,
        num_nodes=len(nodes),
        node_types=node_types,
        edge_index_by_network=edge_index_by_network,
        edge_attr_by_network=edge_attr_by_network,
    )

    tr_ids, val_ids = gb.patient_holdout_val_split(
        train_ids, y=y_numeric, groups=train_groups,
        n_splits=5, val_fold=0, random_state=gb.RANDOM_STATE,
    )

    gb.set_seed()
    model = gb.make_model(best_params, len(nodes), node_types,
                          edge_index_by_network, edge_attr_by_network)
    class_weights = gb.compute_class_weights(tr_ids, y_numeric).to(gb.DEVICE)
    model, val_macro_f1, _ = gb.train_with_early_stopping(
        model=model, dataset_by_id=dataset_by_id,
        tr_ids=tr_ids, val_ids=val_ids,
        lr=float(best_params['lr']), class_weights=class_weights, verbose=False,
    )

    test_metrics = gb.evaluate_ids(model=model, dataset_by_id=dataset_by_id, ids=test_ids)
    gb.print_test_metrics(test_metrics, title=f'Test set — modality={modality}')

    row = {
        'modality': modality,
        'n_nodes': len(nodes),
        'n_mirna_nodes': int((node_types == 0).sum().item()),
        'n_mrna_nodes': int((node_types == 1).sum().item()),
        'edges_per_network': int(sum(n_edges.values()) / max(len(n_edges), 1)),
        'cv_val_macro_f1': float(val_macro_f1),
        'test_accuracy': float(test_metrics['accuracy']),
        'test_macro_f1': float(test_metrics['macro_f1']),
    }
    for cname, auc in test_metrics['per_class_auc'].items():
        row[f'auroc_{cname}'] = float(auc)
    return row


def plot_results(df):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    labels = df['modality'].tolist()
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, df['test_accuracy'], width, label='Test accuracy')
    ax.bar(x + width / 2, df['test_macro_f1'], width, label='Test macro F1')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Score')
    ax.set_title('GAT modality ablation (held-out test)')
    for i, (a, f) in enumerate(zip(df['test_accuracy'], df['test_macro_f1'])):
        ax.text(i - width / 2, a + 0.01, f'{a:.3f}', ha='center', fontsize=8)
        ax.text(i + width / 2, f + 0.01, f'{f:.3f}', ha='center', fontsize=8)
    ax.legend()
    fig.tight_layout()
    out = FIGURES_DIR / 'modality_ablation.png'
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f'Saved {out}')


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    print(f'Using device: {gb.DEVICE}')

    x_all, y_all, train_ids, test_ids = gb.load_data()
    mirna_cols = pd.read_csv(
        gb.PROCESSED_DIR / 'train_mirna.csv', index_col=0
    ).columns.tolist()

    rows = []
    for modality in MODALITIES:
        rows.append(run_variant(modality, x_all, y_all, train_ids, test_ids, mirna_cols))

    df = pd.DataFrame(rows)
    out_csv = TABLES_DIR / 'modality_ablation.csv'
    df.to_csv(out_csv, index=False)
    print('\n' + '=' * 60)
    print('MODALITY ABLATION SUMMARY')
    print('=' * 60)
    print(df.to_string(index=False))
    print(f'\nSaved {out_csv}')
    plot_results(df)


if __name__ == '__main__':
    main()
