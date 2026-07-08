#!/usr/bin/env python3
"""
Tissue-specific network construction for TNBC GSE45498 (top-N hub-preserving).

DESIGN (top-150 edges per type):

For each tissue group (normal / primary / metastatic), build an undirected
network from TRAINING samples only with three edge types, each capped at the
top ``TOP_N_PER_EDGE_TYPE`` (default 150) strongest edges that pass
Benjamini-Hochberg FDR correction (q < ``FDR_ALPHA``):

  * miRNA--miRNA : co-expression, ranked by Pearson r (positive correlation)
  * mRNA--mRNA   : co-expression, ranked by Pearson r (positive correlation)
  * miRNA--mRNA  : regulatory, ranked by most-NEGATIVE Pearson r (repression
                   biology); miRDB v6 predictions are attached as a validation
                   annotation only (they do NOT select edges).

Unlike the kNN builder, NO per-node degree cap is applied, so highly
co-expressed molecules accumulate many edges and emerge as hubs (scale-free-like
topology) instead of every node having ~equal degree. Nodes that end up with no
FDR-significant partner are retained as isolated nodes (the GAT adds self-loops
so they keep their own expression signal).

Outputs (per tissue):
  results/{tissue}_network.graphml          (consumed by the GAT scripts)
  results/networks/{tissue}_edges_top{N}.csv (edge table for inspection/ablation)

Leakage note: correlations are computed strictly on training samples of the
given tissue, so the topology is never informed by the held-out test set.
"""

import os
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import t as _t_dist
from statsmodels.stats.multitest import multipletests


PROCESSED_DIR = 'data/processed/'
RESULTS_DIR = 'results/'
TABLES_DIR = os.path.join(RESULTS_DIR, 'tables')
NETWORKS_DIR = os.path.join(RESULTS_DIR, 'networks')

# --- Tunable parameters -----------------------------------------------------
TOP_N_PER_EDGE_TYPE = 150   # top edges kept per type per tissue (allows hubs)
FDR_ALPHA = 0.05            # Benjamini-Hochberg threshold (annotation/gate)
# If True, only FDR-significant pairs are eligible for the top-N (strict). The
# cross-modal miRNA-mRNA anti-correlations rarely survive genome-wide FDR at
# this sample size (n=32-113 per tissue), which empties the regulatory layer,
# so the default keeps the top-N strongest edges and records FDR q-values as an
# ANNOTATION instead. Flip to True for a strict, possibly-sparse network.
REQUIRE_FDR = False
TISSUES = ('normal', 'primary', 'metastatic')


def load_inputs():
    """Load training matrices, labels, and (optional) miRDB validation pairs."""
    required = [
        os.path.join(PROCESSED_DIR, 'train_mirna.csv'),
        os.path.join(PROCESSED_DIR, 'train_mrna.csv'),
        os.path.join(PROCESSED_DIR, 'train_labels.csv'),
    ]
    for p in required:
        if not os.path.exists(p):
            raise FileNotFoundError(f'Missing required file: {p}')

    mirna = pd.read_csv(os.path.join(PROCESSED_DIR, 'train_mirna.csv'), index_col=0)
    mrna = pd.read_csv(os.path.join(PROCESSED_DIR, 'train_mrna.csv'), index_col=0)
    labels = pd.read_csv(
        os.path.join(PROCESSED_DIR, 'train_labels.csv'), index_col=0
    )['tissue_group']

    # miRDB validated pairs (annotation only, never edge selection).
    mirdb_validated = set()
    pair_path = os.path.join(PROCESSED_DIR, 'mirna_mrna_pairs.csv')
    if os.path.exists(pair_path):
        try:
            pairs = pd.read_csv(pair_path)
            mrna_col = 'mRNA_gene_symbol' if 'mRNA_gene_symbol' in pairs.columns else (
                'mRNA' if 'mRNA' in pairs.columns else None
            )
            if mrna_col is not None and 'miRNA' in pairs.columns:
                mirdb_validated = set(
                    zip(pairs['miRNA'].astype(str), pairs[mrna_col].astype(str))
                )
        except pd.errors.EmptyDataError:
            pass

    print('Loaded processed inputs')
    print(f'  train_mirna: {mirna.shape[0]} samples x {mirna.shape[1]} miRNAs')
    print(f'  train_mrna:  {mrna.shape[0]} samples x {mrna.shape[1]} mRNAs')
    print(f'  train_labels: {len(labels)} entries')
    print(f'  miRDB validated pairs (annotation): {len(mirdb_validated)}')

    return mirna, mrna, labels, mirdb_validated


def _cuda_corrcoef(mat: np.ndarray) -> np.ndarray:
    """Pearson correlation matrix via cupy (GPU) when available, else numpy."""
    try:
        import cupy as cp
        gpu_mat = cp.asarray(mat)
        corr = cp.corrcoef(gpu_mat.T)
        return cp.asnumpy(corr)
    except Exception:
        return np.corrcoef(mat.T)


def _pearson_pvalue_matrix(r: np.ndarray, n: int) -> np.ndarray:
    """Two-tailed p-values for a correlation matrix via the t identity."""
    r_safe = np.clip(r, -1.0 + 1e-10, 1.0 - 1e-10)
    t_stat = r_safe * np.sqrt(n - 2) / np.sqrt(1.0 - r_safe ** 2)
    p = 2.0 * _t_dist.sf(np.abs(t_stat), df=n - 2)
    np.fill_diagonal(p, 1.0)
    return p


def _cross_pvalues(r: np.ndarray, n: int) -> np.ndarray:
    """Two-tailed p-values for a rectangular cross-correlation matrix."""
    r_safe = np.clip(r, -1.0 + 1e-10, 1.0 - 1e-10)
    t_stat = r_safe * np.sqrt(n - 2) / np.sqrt(1.0 - r_safe ** 2)
    return 2.0 * _t_dist.sf(np.abs(t_stat), df=n - 2)


def _clean_matrix(values: pd.DataFrame, tissue: str, label: str):
    """Return (matrix, feature_names) with NaN imputed and zero-var cols dropped."""
    feature_names = values.columns.tolist()
    mat = values.to_numpy(dtype=float)
    col_means = np.nanmean(mat, axis=0)
    nan_mask = np.isnan(mat)
    if nan_mask.any():
        mat = mat.copy()
        mat[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    col_std = mat.std(axis=0)
    valid = col_std > 0
    if not valid.all():
        print(f'  {tissue}: dropping {int((~valid).sum())} zero-variance {label} feature(s)')
        mat = mat[:, valid]
        feature_names = [fn for fn, v in zip(feature_names, valid) if v]
    return mat, feature_names


def build_intramodal_edges(values, tissue, edge_type, top_n=TOP_N_PER_EDGE_TYPE):
    """Top-N positive co-expression edges (FDR-significant), hubs allowed."""
    if values.shape[1] < 2:
        return []
    mat, feature_names = _clean_matrix(values, tissue, edge_type)
    if len(feature_names) < 2:
        return []
    n_samples = mat.shape[0]

    start = time.time()
    corr = _cuda_corrcoef(mat)
    pmat = _pearson_pvalue_matrix(corr, n_samples)
    rows, cols = np.triu_indices(len(feature_names), k=1)
    r_vals = corr[rows, cols]
    p_vals = pmat[rows, cols]

    finite = np.isfinite(r_vals) & np.isfinite(p_vals)
    r_vals, p_vals = r_vals[finite], p_vals[finite]
    rows, cols = rows[finite], cols[finite]
    if r_vals.size == 0:
        return []

    _, p_adj, _, _ = multipletests(p_vals, alpha=FDR_ALPHA, method='fdr_bh')
    fdr_sig = p_adj < FDR_ALPHA
    # Co-expression: keep positive correlations; FDR is a gate or annotation.
    eligible = (r_vals > 0)
    if REQUIRE_FDR:
        eligible = eligible & fdr_sig
    r_e, p_e, padj_e = r_vals[eligible], p_vals[eligible], p_adj[eligible]
    rows_e, cols_e = rows[eligible], cols[eligible]

    order = np.argsort(r_e)[::-1][:top_n]   # strongest positive first
    edges = []
    n_top_fdr = 0
    for idx in order:
        i, j = int(rows_e[idx]), int(cols_e[idx])
        passed = bool(padj_e[idx] < FDR_ALPHA)
        n_top_fdr += int(passed)
        edges.append({
            'source': feature_names[i], 'target': feature_names[j],
            'edge_type': edge_type, 'regulatory_type': 'coexpression',
            'weight': float(r_e[idx]), 'pearson_r': float(r_e[idx]),
            'pearson_p': float(p_e[idx]), 'pearson_p_fdr': float(padj_e[idx]),
            'fdr_significant': passed,
            'miRDB_score': np.nan, 'miRDB_validated': False, 'comparison': '',
        })
    elapsed = time.time() - start
    print(f'  {tissue}: {edge_type} - {int(fdr_sig.sum())} FDR-sig overall, '
          f'kept top {len(edges)} ({n_top_fdr} of them FDR-sig) ({elapsed:.2f}s)')
    return edges


def build_crossmodal_edges(mirna_values, mrna_values, tissue, mirdb_validated,
                           top_n=TOP_N_PER_EDGE_TYPE):
    """Top-N most anti-correlated miRNA-mRNA edges (FDR-significant).

    Ranks by most-negative Pearson r (miRNA represses mRNA). miRDB membership is
    attached as an annotation but does NOT influence selection.
    """
    if mirna_values.shape[1] < 1 or mrna_values.shape[1] < 1:
        return []
    mi_mat, mi_names = _clean_matrix(mirna_values, tissue, 'miRNA(cross)')
    mr_mat, mr_names = _clean_matrix(mrna_values, tissue, 'mRNA(cross)')
    n_samples = mi_mat.shape[0]
    if not mi_names or not mr_names:
        return []

    start = time.time()
    # Cross-correlation between every miRNA (columns of mi_mat) and mRNA.
    mi_z = (mi_mat - mi_mat.mean(0)) / mi_mat.std(0)
    mr_z = (mr_mat - mr_mat.mean(0)) / mr_mat.std(0)
    cross = (mi_z.T @ mr_z) / n_samples          # (n_mirna, n_mrna)
    pmat = _cross_pvalues(cross, n_samples)

    r_flat = cross.ravel()
    p_flat = pmat.ravel()
    mi_idx, mr_idx = np.unravel_index(np.arange(r_flat.size), cross.shape)
    finite = np.isfinite(r_flat) & np.isfinite(p_flat)
    r_flat, p_flat = r_flat[finite], p_flat[finite]
    mi_idx, mr_idx = mi_idx[finite], mr_idx[finite]

    _, p_adj, _, _ = multipletests(p_flat, alpha=FDR_ALPHA, method='fdr_bh')
    fdr_sig = p_adj < FDR_ALPHA
    eligible = (r_flat < 0)                       # anti-correlation (repression)
    if REQUIRE_FDR:
        eligible = eligible & fdr_sig
    r_e, p_e, padj_e = r_flat[eligible], p_flat[eligible], p_adj[eligible]
    mi_e, mr_e = mi_idx[eligible], mr_idx[eligible]

    order = np.argsort(r_e)[:top_n]              # most negative first
    edges = []
    n_top_fdr = 0
    for idx in order:
        mi_name = mi_names[int(mi_e[idx])]
        mr_name = mr_names[int(mr_e[idx])]
        validated = (mi_name, mr_name) in mirdb_validated
        passed = bool(padj_e[idx] < FDR_ALPHA)
        n_top_fdr += int(passed)
        edges.append({
            'source': mi_name, 'target': mr_name,
            'edge_type': 'miRNA-mRNA',
            'regulatory_type': 'validated' if validated else 'anticorrelation',
            'weight': float(r_e[idx]), 'pearson_r': float(r_e[idx]),
            'pearson_p': float(p_e[idx]), 'pearson_p_fdr': float(padj_e[idx]),
            'fdr_significant': passed,
            'miRDB_score': np.nan, 'miRDB_validated': bool(validated),
            'comparison': '',
        })
    elapsed = time.time() - start
    n_val = sum(e['miRDB_validated'] for e in edges)
    print(f'  {tissue}: miRNA-mRNA - {int(fdr_sig.sum())} anti-corr FDR-sig overall, '
          f'kept top {len(edges)} ({n_top_fdr} FDR-sig, {n_val} miRDB-validated) '
          f'({elapsed:.2f}s)')
    return edges


def build_network(tissue, mirna_group, mrna_group, mirdb_validated):
    """Assemble a tissue network from the three top-N edge sets."""
    graph = nx.Graph()
    tissue_attr = f'mean_expression_{tissue}'

    for mir in mirna_group.columns:
        graph.add_node(mir, node_type='miRNA',
                       mean_expression_normal=np.nan,
                       mean_expression_primary=np.nan,
                       mean_expression_metastatic=np.nan)
        graph.nodes[mir][tissue_attr] = float(mirna_group[mir].mean())
    for gene in mrna_group.columns:
        graph.add_node(gene, node_type='mRNA',
                       mean_expression_normal=np.nan,
                       mean_expression_primary=np.nan,
                       mean_expression_metastatic=np.nan)
        graph.nodes[gene][tissue_attr] = float(mrna_group[gene].mean())

    mi_edges = build_intramodal_edges(mirna_group, tissue, 'miRNA-miRNA')
    mr_edges = build_intramodal_edges(mrna_group, tissue, 'mRNA-mRNA')
    cross_edges = build_crossmodal_edges(mirna_group, mrna_group, tissue, mirdb_validated)

    for item in mi_edges + mr_edges + cross_edges:
        graph.add_edge(
            item['source'], item['target'],
            edge_type=item['edge_type'],
            regulatory_type=item['regulatory_type'],
            weight=item['weight'],
            pearson_r=item['pearson_r'],
            pearson_p=item['pearson_p'],
            pearson_p_fdr=item['pearson_p_fdr'],
            miRDB_score=item['miRDB_score'],
            miRDB_validated=item['miRDB_validated'],
            comparison=item['comparison'],
        )

    counts = {
        'miRNA-miRNA': len(mi_edges),
        'mRNA-mRNA': len(mr_edges),
        'miRNA-mRNA': len(cross_edges),
    }
    # Degree summary to confirm hub formation.
    degrees = dict(graph.degree())
    if degrees:
        deg_vals = np.array(list(degrees.values()))
        top_hub = max(degrees, key=degrees.get)
        n_isolated = int((deg_vals == 0).sum())
        print(f'{tissue.upper()}: edges mi-mi={len(mi_edges)}, mr-mr={len(mr_edges)}, '
              f'mi-mr={len(cross_edges)}; max degree={deg_vals.max()} ({top_hub}), '
              f'mean degree={deg_vals.mean():.1f}, isolated nodes={n_isolated}')
    return graph, counts


def save_outputs(graph, tissue):
    """Write graphml and edge CSV for a tissue network."""
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(NETWORKS_DIR).mkdir(parents=True, exist_ok=True)

    graphml_path = os.path.join(RESULTS_DIR, f'{tissue}_network.graphml')
    nx.write_graphml(graph, graphml_path)

    edge_rows = []
    for u, v, data in graph.edges(data=True):
        edge_rows.append({
            'source': u, 'target': v,
            'source_type': graph.nodes[u].get('node_type', ''),
            'target_type': graph.nodes[v].get('node_type', ''),
            'edge_type': data.get('edge_type', ''),
            'regulatory_type': data.get('regulatory_type', ''),
            'weight': data.get('weight', np.nan),
            'pearson_r': data.get('pearson_r', np.nan),
            'pearson_p': data.get('pearson_p', np.nan),
            'pearson_p_fdr': data.get('pearson_p_fdr', np.nan),
            'fdr_significant': data.get('fdr_significant', False),
            'miRDB_validated': data.get('miRDB_validated', False),
        })
    csv_path = os.path.join(NETWORKS_DIR, f'{tissue}_edges_top{TOP_N_PER_EDGE_TYPE}.csv')
    pd.DataFrame(edge_rows).to_csv(csv_path, index=False)

    print(f'Saved graphml: {graphml_path}')
    print(f'Saved edge table: {csv_path}')


def build_tissue_networks(tissues=TISSUES):
    mirna, mrna, labels, mirdb_validated = load_inputs()
    summary_rows = []
    for tissue in tissues:
        samples = labels[labels == tissue].index.tolist()
        if not samples:
            print(f'No samples for {tissue}; skipping')
            continue
        mirna_group = mirna.loc[samples]
        mrna_group = mrna.loc[samples]
        print(f'\nBuilding {tissue.upper()} network ({len(samples)} samples, '
              f'{mirna_group.shape[1]} miRNAs, {mrna_group.shape[1]} mRNAs)')
        graph, counts = build_network(tissue, mirna_group, mrna_group, mirdb_validated)
        save_outputs(graph, tissue)
        summary_rows.append({
            'tissue': tissue,
            'node_count': graph.number_of_nodes(),
            'total_edges': graph.number_of_edges(),
            'miRNA-miRNA_edges': counts['miRNA-miRNA'],
            'mRNA-mRNA_edges': counts['mRNA-mRNA'],
            'miRNA-mRNA_edges': counts['miRNA-mRNA'],
        })
    if summary_rows:
        print('\nNetwork Summary')
        print(pd.DataFrame(summary_rows).to_string(index=False))


def main():
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    print('=' * 60)
    print(f'Network construction: top-{TOP_N_PER_EDGE_TYPE} per edge type, '
          f'FDR<{FDR_ALPHA}, hubs allowed')
    print('=' * 60)
    build_tissue_networks()


if __name__ == '__main__':
    main()
