#!/usr/bin/env python3
"""
miRNA-mRNA Pair Construction
Build putative miRNA-mRNA regulatory pairs from DE results and miRDB predictions.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import warnings
import re
warnings.filterwarnings('ignore')


def _batched_pearsonr(x_mat: np.ndarray, y_mat: np.ndarray) -> np.ndarray:
    """Compute column-wise Pearson r between paired columns of x_mat and y_mat.

    Uses cupy (GPU) when CUDA is available; falls back to numpy otherwise.

    Args:
        x_mat: (n_samples, n_pairs)
        y_mat: (n_samples, n_pairs)

    Returns:
        r: (n_pairs,) — nan where standard deviation is zero.
    """
    try:
        import cupy as cp
        x = cp.asarray(x_mat, dtype=cp.float64)
        y = cp.asarray(y_mat, dtype=cp.float64)
        xm = x - x.mean(axis=0)
        ym = y - y.mean(axis=0)
        num = (xm * ym).sum(axis=0)
        denom = cp.sqrt((xm ** 2).sum(axis=0) * (ym ** 2).sum(axis=0))
        r = cp.where(denom > 1e-10, num / denom, cp.nan)
        return cp.asnumpy(r)
    except Exception:
        pass

    # NumPy fallback (BLAS-accelerated via einsum).
    x = x_mat.astype(np.float64)
    y = y_mat.astype(np.float64)
    xm = x - x.mean(axis=0)
    ym = y - y.mean(axis=0)
    num = (xm * ym).sum(axis=0)
    denom = np.sqrt((xm ** 2).sum(axis=0) * (ym ** 2).sum(axis=0))
    return np.where(denom > 1e-10, num / denom, np.nan)

PROCESSED_DIR = 'data/processed/'
RAW_DATA_DIR = 'data/raw/'
PAIR_COLUMNS = [
    'miRNA',
    'mRNA_probe',
    'mRNA_gene_symbol',
    'pearson_r',
    'miRDB_score',
    'comparison',
    'log2FC_miRNA',
    'correlation_direction'
]


def normalize_mirna_name(name):
    """Normalize miRNA IDs to align DE results with training column names."""
    if pd.isna(name):
        return name
    return re.sub(r"\s*\(\+\+\+ See note below\)\s*$", "", str(name)).strip()

def load_data():
    """Load all required data files."""
    print("Loading data files...")
    
    de_mirnas = pd.read_csv(f'{PROCESSED_DIR}de_mirnas.csv')
    de_mirnas['molecule_name'] = de_mirnas['molecule_name'].map(normalize_mirna_name)
    de_mrnas = pd.read_csv(f'{PROCESSED_DIR}de_mrnas.csv')
    
    print(f"DE miRNAs: {len(de_mirnas)} entries")
    print(f"DE mRNAs: {len(de_mrnas)} entries")
    
    mrna_raw = pd.read_csv(f'{RAW_DATA_DIR}GSE45498_mRNA_non-normalized_data.txt.gz', 
                           sep='\t', compression='gzip')
    refseq_to_symbol = dict(zip(mrna_raw['Accession'].str.split('.').str[0], mrna_raw['Name']))
    print(f"Created RefSeq to gene symbol mapping: {len(refseq_to_symbol)} entries")
    
    mirdb_file = f'{RAW_DATA_DIR}miRDB_v6.0_prediction_result.txt.gz'
    mirdb = pd.read_csv(mirdb_file, sep="\t", header=None, compression='gzip',
                        names=["miRNA", "target_refseq", "score"])
    mirdb = mirdb[mirdb['miRNA'].str.startswith('hsa-')].copy()
    mirdb['target_refseq'] = mirdb['target_refseq'].astype(str)
    mirdb['target_probe'] = mirdb['target_refseq'].str.split('.').str[0]
    mirdb['target_gene'] = mirdb['target_probe'].map(refseq_to_symbol)
    mirdb = mirdb.dropna(subset=['target_gene']).copy()
    print(f"miRDB predictions (human only, mapped to symbols): {len(mirdb)} entries")
    
    train_mirna = pd.read_csv(f'{PROCESSED_DIR}train_mirna.csv', index_col=0)
    train_mrna = pd.read_csv(f'{PROCESSED_DIR}train_mrna.csv', index_col=0)
    
    print(f"Training miRNA expression: {train_mirna.shape}")
    print(f"Training mRNA expression: {train_mrna.shape}")
    
    return de_mirnas, de_mrnas, mirdb, train_mirna, train_mrna

def build_pairs(de_mirnas, de_mrnas, mirdb, train_mirna, train_mrna):
    """Build miRNA-mRNA pairs based on DE overlap and miRDB predictions."""
    print("\nBuilding miRNA-mRNA pairs...")
    
    de_mirna_names = set(de_mirnas['molecule_name'].unique())
    measurable_mrna = set(train_mrna.columns)
    
    print(f"Number of DE miRNAs: {len(de_mirna_names)}")
    print(f"Number of DE mRNAs (reference only): {len(set(de_mrnas['molecule_name'].unique()))}")
    print(f"Measured mRNA panel size: {len(measurable_mrna)}")
    
    mirdb_scored = mirdb[mirdb['score'] > 70].copy()
    print(f"Total human miRDB rows after score filter: {len(mirdb_scored)}")

    mirdb_filtered = mirdb_scored[mirdb_scored['target_gene'].isin(measurable_mrna)].copy()
    print(f"Human miRDB rows in measured mRNA panel: {len(mirdb_filtered)}")
    
    mirdb_de = mirdb_filtered[mirdb_filtered['miRNA'].isin(de_mirna_names)].copy()
    print(f"Number of candidate overlaps before correlation filtering: {len(mirdb_de)}")
    
    mirna_comparisons = de_mirnas.groupby('molecule_name')['comparison'].apply(set).to_dict()
    mirna_fc = de_mirnas.set_index(['molecule_name', 'comparison'])['log2FC'].to_dict()
    
    pairs_data = []
    all_candidates = []
    pair_count = 0
    
    mirna_cols = train_mirna.columns.tolist()
    mrna_cols = train_mrna.columns.tolist()
    mirna_indices = {c: i for i, c in enumerate(mirna_cols)}
    mrna_indices = {c: i for i, c in enumerate(mrna_cols)}
    
    mirna_arr = train_mirna.values.astype(float)
    mrna_arr = train_mrna.values.astype(float)
    mirna_std = np.std(mirna_arr, axis=0)
    mrna_std = np.std(mrna_arr, axis=0)
    
    print("Computing Pearson correlations (batched, GPU-accelerated if CUDA available)...")

    # Collect all valid candidate rows for a single batched correlation call.
    batch_meta = []
    for _, row in mirdb_de.iterrows():
        mirna = row['miRNA']
        mrna_probe = row['target_probe']
        mrna = row['target_gene']
        mirdb_score = row['score']

        mirna_idx = mirna_indices.get(mirna)
        mrna_idx = mrna_indices.get(mrna)

        if mirna_idx is None or mrna_idx is None:
            continue
        if mirna_std[mirna_idx] == 0 or mrna_std[mrna_idx] == 0:
            continue

        batch_meta.append((mirna, mrna_probe, mrna, mirdb_score, mirna_idx, mrna_idx))

    if batch_meta:
        mi_idxs = np.array([m[4] for m in batch_meta], dtype=int)
        mr_idxs = np.array([m[5] for m in batch_meta], dtype=int)
        mirna_batch = mirna_arr[:, mi_idxs]   # (n_samples, n_pairs)
        mrna_batch = mrna_arr[:, mr_idxs]     # (n_samples, n_pairs)
        r_values = _batched_pearsonr(mirna_batch, mrna_batch)

        print(f"  Computed {len(batch_meta)} pair correlations in one batch")

        for (mirna, mrna_probe, mrna, mirdb_score, _, _), r in zip(batch_meta, r_values):
            if not np.isfinite(r):
                continue

            mirna_comps = mirna_comparisons.get(mirna, set())
            for comp in mirna_comps:
                log2fc = mirna_fc.get((mirna, comp), np.nan)
                all_candidates.append({
                    'miRNA': mirna,
                    'mRNA_probe': mrna_probe,
                    'mRNA_gene_symbol': mrna,
                    'pearson_r': r,
                    'miRDB_score': mirdb_score,
                    'comparison': comp,
                    'log2FC_miRNA': log2fc,
                    'correlation_direction': 'positive' if r > 0 else 'negative'
                })

            if abs(r) <= 0.1:
                continue

            correlation_direction = 'positive' if r > 0 else 'negative'

            for comp in mirna_comps:
                log2fc = mirna_fc.get((mirna, comp), np.nan)
                pairs_data.append({
                    'miRNA': mirna,
                    'mRNA_probe': mrna_probe,
                    'mRNA_gene_symbol': mrna,
                    'pearson_r': r,
                    'miRDB_score': mirdb_score,
                    'comparison': comp,
                    'log2FC_miRNA': log2fc,
                    'correlation_direction': correlation_direction
                })

            pair_count += 1
    
    pairs_df = pd.DataFrame(pairs_data)
    all_candidates_df = pd.DataFrame(all_candidates)
    
    if len(pairs_df) == 0:
        pairs_df = pd.DataFrame(columns=PAIR_COLUMNS)
    else:
        pairs_df = pairs_df.drop_duplicates(subset=['miRNA', 'mRNA_gene_symbol', 'comparison', 'mRNA_probe'])
        pairs_df = pairs_df.sort_values(['comparison', 'pearson_r'])

    if len(pairs_df) > 0:
        pairs_df = pairs_df[PAIR_COLUMNS]
    if len(all_candidates_df) > 0:
        all_candidates_df = all_candidates_df[PAIR_COLUMNS]
    else:
        all_candidates_df = pd.DataFrame(columns=PAIR_COLUMNS)

    print(f"Candidate overlaps before correlation filtering: {len(all_candidates_df)}")
    print(f"Final pairs after |r| > 0.1 filter: {len(pairs_df)}")
    old_filter_count = len(all_candidates_df[all_candidates_df['pearson_r'] < -0.1]) if len(all_candidates_df) > 0 else 0
    print(f"Pairs that would pass old r < -0.1 filter: {old_filter_count}")

    return pairs_df, all_candidates_df

def print_summary(pairs_df, candidate_pairs=None):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("miRNA-mRNA PAIRS SUMMARY")
    print("="*60)
    
    total_pairs = len(pairs_df)
    print(f"\nTotal pairs found: {total_pairs}")
    
    if total_pairs > 0:
        print("\nBreakdown by comparison:")
        for comp in ['Normal_vs_Primary', 'Primary_vs_Metastatic', 'Normal_vs_Metastatic']:
            comp_pairs = len(pairs_df[pairs_df['comparison'] == comp])
            print(f"  {comp}: {comp_pairs} pairs")
        
        print(f"\nPearson r range: [{pairs_df['pearson_r'].min():.3f}, {pairs_df['pearson_r'].max():.3f}]")
        print(f"miRDB score range: [{pairs_df['miRDB_score'].min():.1f}, {pairs_df['miRDB_score'].max():.1f}]")
        
        print("\nTop 10 most negative Pearson correlations found:")
        top10 = pairs_df.nsmallest(10, 'pearson_r')[
            ['miRNA', 'mRNA_gene_symbol', 'pearson_r', 'miRDB_score', 'comparison', 'mRNA_probe', 'log2FC_miRNA']
        ]
        for _, row in top10.iterrows():
            print(
                f"  {row['miRNA']} -> {row['mRNA_gene_symbol']} "
                f"[probe {row['mRNA_probe']}]: "
                f"r={row['pearson_r']:.3f}, miRDB={row['miRDB_score']:.1f}, "
                f"log2FC_miRNA={row['log2FC_miRNA']:.3f} ({row['comparison']})"
            )
    else:
        if candidate_pairs is not None and len(candidate_pairs) > 0:
            print("\nNote: 0 pairs passed r < -0.1 filter.")
            print("Weakest anti-correlations found:")
            candidate_pairs_sorted = candidate_pairs.sort_values('pearson_r')
            for _, row in candidate_pairs_sorted.head(5).iterrows():
                print(f"  {row['miRNA']} <-> {row['mRNA_gene_symbol']}: r={row['pearson_r']:.3f}, miRDB={row['miRDB_score']:.1f}")
            print("\nConsider checking the correlation distribution or relaxing the correlation threshold for this dataset.")

def main():
    de_mirnas, de_mrnas, mirdb, train_mirna, train_mrna = load_data()
    
    pairs_df, all_candidates_df = build_pairs(de_mirnas, de_mrnas, mirdb, train_mirna, train_mrna)
    
    pairs_df.to_csv(f'{PROCESSED_DIR}mirna_mrna_pairs.csv', index=False)
    print(f"\nSaved {len(pairs_df)} pairs to {PROCESSED_DIR}mirna_mrna_pairs.csv")
    
    print_summary(pairs_df, all_candidates_df)
    
    print("\n" + "="*60)
    print("Pair construction complete!")
    print("="*60)

if __name__ == '__main__':
    main()
