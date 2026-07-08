#!/usr/bin/env python3
"""
GSE45498 Preprocessing and Differential Expression Analysis
NanoString nCounter data: 664 miRNAs, 230 mRNAs, 278 samples

DE method: pydeseq2 (preferred; this is what the paper used) or limma-voom via rpy2 (fallback).
Both methods work on raw count data and produce log2FC with adjusted p-values.
"""

import gzip
import json
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

RAW_DATA_DIR = 'data/raw/'
PROCESSED_DIR = 'data/processed/'
FIGURES_DIR = 'results/figures/'


def _try_import_rpy2():
    try:
        import rpy2
        from rpy2 import robjects
        from rpy2.robjects import pandas2ri
        return True
    except ImportError:
        return False


def _try_import_pydeseq2():
    try:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats
        return True
    except ImportError:
        return False


DE_METHOD = None
# The paper used pydeseq2. limma-voom needs a full R setup that is often missing
# (e.g. Colab ships rpy2 but NOT the R 'limma' package, which used to crash DE and
# silently fall back to stale results). Prefer pydeseq2; use limma only if pydeseq2
# is unavailable. Override with env var DE_METHOD_FORCE=limma if you really want R.
import os as _os
if _os.environ.get('DE_METHOD_FORCE') == 'limma' and _try_import_rpy2():
    DE_METHOD = 'limma-voom (rpy2)'
elif _try_import_pydeseq2():
    DE_METHOD = 'pydeseq2'
elif _try_import_rpy2():
    DE_METHOD = 'limma-voom (rpy2)'
else:
    raise ImportError(
        "Neither rpy2 nor pydeseq2 is available. "
        "Install one with: pip install rpy2 or pip install pydeseq2"
    )

print(f"DE method selected: {DE_METHOD}")


def parse_series_matrix(filepath):
    f = gzip.open(filepath, 'rt')
    lines = f.readlines()
    f.close()

    sample_info = {}

    for line in lines:
        if line.startswith('!Sample_title'):
            parts = line.strip().split('\t')
            for i, p in enumerate(parts[1:], 1):
                if i not in sample_info:
                    sample_info[i] = {}
                sample_info[i]['title'] = p.strip().strip('"')
        elif line.startswith('!Sample_geo_accession'):
            parts = line.strip().split('\t')
            for i, p in enumerate(parts[1:], 1):
                if i not in sample_info:
                    sample_info[i] = {}
                sample_info[i]['geo'] = p.strip().strip('"')
        elif line.startswith('!Sample_characteristics_ch1'):
            parts = line.strip().split('\t')
            for i, p in enumerate(parts[1:], 1):
                if i not in sample_info:
                    sample_info[i] = {}
                if 'tissue:' in p:
                    sample_info[i]['tissue'] = p.strip().strip('"')
                if 'status:' in p:
                    sample_info[i]['status'] = p.strip().strip('"')

    return sample_info


def build_sample_mapping():
    print("Building sample mapping...")

    miRNA_meta = parse_series_matrix(f'{RAW_DATA_DIR}GSE45498-GPL16231_series_matrix.txt.gz')
    mRNA_meta = parse_series_matrix(f'{RAW_DATA_DIR}GSE45498-GPL16299_series_matrix.txt.gz')

    mirna_df = pd.read_csv(f'{RAW_DATA_DIR}GSE45498_raw_data.txt.gz', sep='\t', compression='gzip')
    mrna_df = pd.read_csv(f'{RAW_DATA_DIR}GSE45498_mRNA_non-normalized_data.txt.gz', sep='\t', compression='gzip')

    mirna_raw_cols = list(mirna_df.columns[3:])
    mrna_raw_cols = [c.strip() for c in mrna_df.columns[3:]]

    mirna_to_mrna = {}

    for i in range(1, min(len(mirna_raw_cols), len(mrna_raw_cols)) + 1):
        mirna_tissue = miRNA_meta.get(i, {}).get('tissue', '')
        mirna_status = miRNA_meta.get(i, {}).get('status', '')
        mrna_tissue = mRNA_meta.get(i, {}).get('tissue', '')
        mrna_status = mRNA_meta.get(i, {}).get('status', '')

        if mirna_tissue == mrna_tissue and mirna_status == mrna_status:
            mirna_sample_name = mirna_raw_cols[i - 1]
            mrna_sample_name = mrna_raw_cols[i - 1]
            mirna_to_mrna[mirna_sample_name] = mrna_sample_name

    print(f"Mapped {len(mirna_to_mrna)} miRNA samples to mRNA samples")

    return mirna_to_mrna


def load_expression_data():
    print("Loading expression data...")

    mirna_df = pd.read_csv(f'{RAW_DATA_DIR}GSE45498_raw_data.txt.gz',
                           sep='\t', compression='gzip')

    mrna_df = pd.read_csv(f'{RAW_DATA_DIR}GSE45498_mRNA_non-normalized_data.txt.gz',
                          sep='\t', compression='gzip')

    print(f"Raw miRNA data shape: {mirna_df.shape}")
    print(f"Raw mRNA data shape: {mrna_df.shape}")

    return mirna_df, mrna_df


def extract_sample_metadata(mirna_df, mirna_to_mrna):
    sample_cols = mirna_df.columns[3:]

    labels = []
    valid_samples = []

    for col in sample_cols:
        if col in mirna_to_mrna:
            valid_samples.append(col)
            if col.startswith('Normal'):
                labels.append('normal')
            elif col.startswith('Tumor'):
                labels.append('primary')
            elif col.startswith('Mets'):
                labels.append('metastatic')
            else:
                labels.append('unknown')

    sample_metadata = pd.DataFrame({
        'sample_id': valid_samples,
        'tissue_group': labels
    })

    return sample_metadata


def separate_mirna_mrna(mirna_df):
    endogenous_mask = mirna_df['Code Class'] == 'Endogenous1'
    mirna_data = mirna_df[endogenous_mask].copy()

    print(f"miRNA features (Endogenous1): {mirna_data.shape[0]}")

    return mirna_data


def separate_mrna_controls(mrna_df):
    endogenous_mask = mrna_df['Code Class'] == 'Endogenous'
    mrna_data = mrna_df[endogenous_mask].copy()

    print(f"mRNA features (Endogenous): {mrna_data.shape[0]}")

    return mrna_data


def create_expression_matrices(mirna_df, mrna_df, sample_metadata, mirna_to_mrna):
    mrna_sample_cols = [c.strip() for c in mrna_df.columns[3:]]
    mrna_df_renamed = mrna_df.copy()
    mrna_df_renamed.columns = list(mrna_df.columns[:3]) + mrna_sample_cols

    mirna_sample_ids = sample_metadata['sample_id'].tolist()
    mrna_sample_ids_mapped = [mirna_to_mrna[s] for s in mirna_sample_ids]

    mirna_expr = mirna_df.set_index('Name')[mirna_sample_ids].T
    mirna_expr.index = mirna_sample_ids

    mrna_expr = mrna_df_renamed.set_index('Name')[mrna_sample_ids_mapped].T
    mrna_expr.index = mirna_sample_ids

    print(f"\nmiRNA expression matrix: {mirna_expr.shape}")
    print(f"mRNA expression matrix: {mrna_expr.shape}")
    print(f"Both matrices share same sample IDs (miRNA naming)")

    return mirna_expr, mrna_expr


def apply_log2_transform(df):
    return np.log2(df + 1)


def zscore_standardize(df):
    return (df - df.mean()) / df.std()


def zscore_train_test(train_df, test_df):
    """Fit z-score parameters on train only, then apply them to train and test."""
    mean = train_df.mean()
    std = train_df.std().replace(0, 1.0)
    return (train_df - mean) / std, (test_df - mean) / std


def derive_patient_id(mrna_title):
    """Map an mRNA-platform sample title to its true patient id.

    GSE45498 encodes the patient identity in the mRNA platform (GPL16299) sample
    titles, NOT the miRNA platform. Titles look like:
      '<num>N'        -> normal breast from patient <num>      -> 'P<num>'
      '<num>T'        -> primary tumor from patient <num>      -> 'P<num>'
      'TAS10-15-<x>'  -> lymph-node metastasis (separate id)   -> 'MET<x>'
    A patient who contributed both a normal and a tumor sample shares the same
    'P<num>' key (e.g. 108N and 108T -> P108), and tumor replicates (91_A/91B)
    also collapse to one key. Metastatic samples use an independent numbering
    that does not cross-reference the P<num> patients, so they are treated as
    distinct patients (the conservative choice in the absence of a link table).
    """
    t = str(mrna_title).replace('_mRNA', '').strip()
    m = re.match(r'^(\d+)[NT]', t)
    if m:
        return 'P' + m.group(1)
    m = re.match(r'^TAS10-15-(\d+)', t)
    if m:
        return 'MET' + m.group(1)
    return 'P_' + t  # fallback; should not occur for GSE45498


def build_patient_id_map():
    """Return {miRNA_sample_name: true_patient_id} using the mRNA-title ids.

    Aligns the two platforms positionally within matching tissue+status (the
    same correspondence used by build_sample_mapping), then reads the patient id
    off the mRNA-platform title via derive_patient_id. This recovers the real
    within-patient structure (matched normal/tumor pairs, tumor replicates) that
    the miRNA-only naming (extract_patient_ids) cannot.
    """
    miRNA_meta = parse_series_matrix(f'{RAW_DATA_DIR}GSE45498-GPL16231_series_matrix.txt.gz')
    mRNA_meta = parse_series_matrix(f'{RAW_DATA_DIR}GSE45498-GPL16299_series_matrix.txt.gz')
    mirna_df = pd.read_csv(f'{RAW_DATA_DIR}GSE45498_raw_data.txt.gz',
                           sep='\t', compression='gzip', nrows=1)
    mirna_raw_cols = list(mirna_df.columns[3:])

    patient_map = {}
    for i in range(1, len(mirna_raw_cols) + 1):
        mi_t = miRNA_meta.get(i, {}).get('tissue', '')
        mi_s = miRNA_meta.get(i, {}).get('status', '')
        mr_t = mRNA_meta.get(i, {}).get('tissue', '')
        mr_s = mRNA_meta.get(i, {}).get('status', '')
        if mi_t == mr_t and mi_s == mr_s:
            mirna_name = mirna_raw_cols[i - 1]
            patient_map[mirna_name] = derive_patient_id(mRNA_meta.get(i, {}).get('title', ''))
    return patient_map


def extract_patient_ids(sample_ids):
    """[DEPRECATED — DO NOT USE FOR GROUPING] miRNA-name-based pseudo ids.

    This derives 'P_<k>' from the miRNA platform sample names
    (Normal_k/Tumor_k/Mets_k). Those numbers are sequential PER TISSUE TYPE and
    do NOT identify a patient: Normal_5 and Tumor_5 are different people. Using
    this for the train/test split LEAKED 14 real patients across train and test
    (a patient's matched normal and tumor landed on opposite sides), inflating
    held-out accuracy. Use build_patient_id_map() / derive_patient_id() instead,
    which read the true patient id from the mRNA-platform titles. Retained only
    for backward compatibility / comparison.
    """
    patient_map = {}
    for sid in sample_ids:
        if sid.startswith('Normal'):
            patient_map[sid] = 'P_' + sid.replace('Normal_', '').split('_')[0]
        elif sid.startswith('Tumor'):
            patient_map[sid] = 'P_' + sid.replace('Tumor_', '').split('_')[0]
        elif sid.startswith('Mets'):
            patient_map[sid] = 'P_' + sid.replace('Mets_', '').split('_')[0]
        else:
            patient_map[sid] = 'P_' + sid
    return patient_map


def patient_level_train_test_split(X_mirna, X_mrna, y, patient_map, test_size=0.2, random_state=42):
    sample_ids = y.index.tolist()
    groups = [patient_map[sid] for sid in sample_ids]

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    splits = list(sgkf.split(X_mirna, y, groups))
    train_idx, test_idx = splits[0]

    train_patient_ids = set(groups[i] for i in train_idx)
    test_patient_ids = set(groups[i] for i in test_idx)
    assert train_patient_ids.isdisjoint(test_patient_ids), "PATIENT LEAKAGE DETECTED"

    train_mirna = X_mirna.iloc[train_idx]
    test_mirna = X_mirna.iloc[test_idx]
    train_mrna = X_mrna.iloc[train_idx]
    test_mrna = X_mrna.iloc[test_idx]
    train_labels = y.iloc[train_idx]
    test_labels = y.iloc[test_idx]

    print(f"Train patients: {len(train_patient_ids)}, Test patients: {len(test_patient_ids)}")
    print(f"Train samples: {len(train_idx)}, Test samples: {len(test_idx)}")
    print(f"Train tissue distribution:\n{train_labels.value_counts()}")
    print(f"Test tissue distribution:\n{test_labels.value_counts()}")

    return train_mirna, test_mirna, train_mrna, test_mrna, train_labels, test_labels


def save_splits(train_mirna, test_mirna, train_mrna, test_mrna, train_labels, test_labels):
    train_mirna.to_csv(f'{PROCESSED_DIR}train_mirna.csv')
    test_mirna.to_csv(f'{PROCESSED_DIR}test_mirna.csv')
    train_mrna.to_csv(f'{PROCESSED_DIR}train_mrna.csv')
    test_mrna.to_csv(f'{PROCESSED_DIR}test_mrna.csv')
    train_labels.to_csv(f'{PROCESSED_DIR}train_labels.csv', header=['tissue_group'], index_label='sample_id')
    test_labels.to_csv(f'{PROCESSED_DIR}test_labels.csv', header=['tissue_group'], index_label='sample_id')

    print("Saved train/test splits to /data/processed/")


def save_patient_and_split_artifacts(patient_map, train_labels, test_labels):
    """Persist sample→patient mapping and train/test sample_id lists for downstream CV."""
    all_sample_ids = train_labels.index.union(test_labels.index)
    patient_rows = [
        {'sample_id': sid, 'patient_id': patient_map[sid]}
        for sid in all_sample_ids
    ]
    pd.DataFrame(patient_rows).to_csv(
        f'{PROCESSED_DIR}patient_ids.csv', index=False
    )

    split_payload = {
        'train_sample_ids': train_labels.index.tolist(),
        'test_sample_ids': test_labels.index.tolist(),
    }
    with open(f'{PROCESSED_DIR}split_indices.json', 'w', encoding='utf-8') as f:
        json.dump(split_payload, f, indent=2)

    print(f"Saved patient_ids.csv ({len(patient_rows)} samples)")
    print(
        f"Saved split_indices.json "
        f"({len(split_payload['train_sample_ids'])} train, "
        f"{len(split_payload['test_sample_ids'])} test)"
    )


def run_de_pydeseq2(counts_df, group1_idx, group2_idx, group1_name, group2_name, comparison_name):
    """Run DE analysis using pydeseq2 on raw count data for a pairwise comparison."""
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    subset_idx = list(group1_idx) + list(group2_idx)
    counts_subset = counts_df.loc[subset_idx].copy()

    counts_subset = counts_subset.fillna(0)
    counts_subset = counts_subset.loc[:, counts_subset.sum(axis=0) > 0]
    counts_int = counts_subset.round().astype(int)

    group_labels = []
    for idx in counts_subset.index:
        if idx in group1_idx:
            group_labels.append(group1_name)
        else:
            group_labels.append(group2_name)

    metadata = pd.DataFrame({
        'condition': group_labels
    }, index=counts_subset.index)

    dds = DeseqDataSet(
        counts=counts_int,
        metadata=metadata,
        design_factors='condition',
        ref_level=['condition', group1_name],
        quiet=True,
    )
    dds.deseq2()

    contrast = ['condition', group2_name, group1_name]
    stat = DeseqStats(dds, contrast=contrast, quiet=True)
    stat.summary()

    results = stat.results_df.copy()
    results.index.name = 'molecule_name'
    results = results.reset_index()

    results = results.rename(columns={
        'log2FoldChange': 'log2FC',
        'padj': 'adj_pvalue',
    })

    results['molecule_name'] = results['molecule_name'].astype(str)
    results['comparison'] = comparison_name

    return results


def run_de_limma_voom(counts_df, group1_idx, group2_idx, group1_name, group2_name, comparison_name):
    """Run DE analysis using limma-voom via rpy2 on raw count data."""
    from rpy2 import robjects
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr

    pandas2ri.activate()

    subset_idx = list(group1_idx) + list(group2_idx)
    counts_subset = counts_df.loc[subset_idx].copy()
    counts_subset = counts_subset.loc[:, counts_subset.sum(axis=0) > 0]
    counts_int = counts_subset.apply(pd.to_numeric).astype(int)

    group_labels = []
    for idx in counts_subset.index:
        if idx in group1_idx:
            group_labels.append(group1_name)
        else:
            group_labels.append(group2_name)

    r_counts = pandas2ri.py2rpy(counts_int.T)
    r_group = robjects.FactorVector(group_labels)

    robjects.r.assign('counts', r_counts)
    robjects.r.assign('group', r_group)

    robjects.r('''
    library(limma)
    library(edgeR)

    dge <- DGEList(counts=counts, group=group)
    dge <- calcNormFactors(dge)
    design <- model.matrix(~group)
    v <- voom(dge, design, plot=FALSE)
    fit <- lmFit(v, design)
    fit <- eBayes(fit)
    results <- topTable(fit, coef=ncol(design), number=Inf, adjust.method="BH")
    ''')

    r_results = robjects.r('results')
    results = pandas2ri.rpy2py(r_results)
    results.index.name = 'molecule_name'
    results = results.reset_index()

    results = results.rename(columns={
        'logFC': 'log2FC',
        'adj.P.Val': 'adj_pvalue',
    })

    results['molecule_name'] = results['molecule_name'].astype(str)
    results['comparison'] = comparison_name

    return results


def run_de_analysis(mirna_expr_raw, mrna_expr_raw, train_labels):
    """Run DE analysis on TRAINING data using limma-voom (preferred) or pydeseq2 (fallback).

    DE is performed on raw counts; log2-transformed data is used only for
    Z-score standardization downstream.
    """
    print(f"\nRunning DE analysis on training set using {DE_METHOD}...")

    train_indices = train_labels.index

    mirna_raw_train = mirna_expr_raw.loc[train_indices]
    mrna_raw_train = mrna_expr_raw.loc[train_indices]

    normal_idx = train_labels[train_labels == 'normal'].index
    primary_idx = train_labels[train_labels == 'primary'].index
    metastatic_idx = train_labels[train_labels == 'metastatic'].index

    comparisons = [
        (normal_idx, primary_idx, 'normal', 'primary', 'Normal_vs_Primary'),
        (primary_idx, metastatic_idx, 'primary', 'metastatic', 'Primary_vs_Metastatic'),
        (normal_idx, metastatic_idx, 'normal', 'metastatic', 'Normal_vs_Metastatic'),
    ]

    mirna_de_results = []
    mrna_de_results = []

    de_func = run_de_limma_voom if DE_METHOD == 'limma-voom (rpy2)' else run_de_pydeseq2

    mirna_fc_threshold = 1.0
    mrna_fc_threshold = 0.5

    for g1_idx, g2_idx, g1_name, g2_name, comp_name in comparisons:
        print(f"\n{comp_name}:")

        mirna_de = de_func(mirna_raw_train, g1_idx, g2_idx, g1_name, g2_name, comp_name)

        mrna_de = de_func(mrna_raw_train, g1_idx, g2_idx, g1_name, g2_name, comp_name)

        mirna_de['significant'] = (mirna_de['adj_pvalue'] < 0.05) & (mirna_de['log2FC'].abs() > mirna_fc_threshold)
        mrna_de['significant'] = (mrna_de['adj_pvalue'] < 0.05) & (mrna_de['log2FC'].abs() > mrna_fc_threshold)

        mirna_de_results.append(mirna_de)
        mrna_de_results.append(mrna_de)

        mirna_sig = mirna_de['significant'].sum()
        mrna_sig = mrna_de['significant'].sum()
        print(f"  miRNAs (adj_p<0.05, |log2FC|>{mirna_fc_threshold}): {mirna_sig} significant")
        print(f"  mRNAs (adj_p<0.05, |log2FC|>{mrna_fc_threshold}): {mrna_sig} significant")

    mirna_all = pd.concat(mirna_de_results, ignore_index=True)
    mrna_all = pd.concat(mrna_de_results, ignore_index=True)

    mirna_filtered = mirna_all[mirna_all['significant'] == True].copy()
    mrna_filtered = mrna_all[mrna_all['significant'] == True].copy()

    mirna_filtered = mirna_filtered[['molecule_name', 'log2FC', 'adj_pvalue', 'comparison']]
    mrna_filtered = mrna_filtered[['molecule_name', 'log2FC', 'adj_pvalue', 'comparison']]

    mirna_filtered.to_csv(f'{PROCESSED_DIR}de_mirnas.csv', index=False)
    mrna_filtered.to_csv(f'{PROCESSED_DIR}de_mrnas.csv', index=False)

    print(f"\nSaved de_mirnas.csv ({len(mirna_filtered)} rows) and de_mrnas.csv ({len(mrna_filtered)} rows)")

    return mirna_all, mrna_all, mirna_filtered, mrna_filtered


def create_volcano_plots(mirna_de, mrna_de, comparison_names):
    for comp_name in comparison_names:
        for df, molecule_type, fc_thresh in [(mirna_de, 'miRNA', 1.0), (mrna_de, 'mRNA', 0.5)]:
            comp_data = df[df['comparison'] == comp_name].copy()

            sig_mask = comp_data['significant']

            plt.figure(figsize=(10, 8))

            plt.scatter(comp_data.loc[~sig_mask, 'log2FC'],
                       -np.log10(comp_data.loc[~sig_mask, 'adj_pvalue'].clip(lower=1e-300)),
                       c='lightgray', alpha=0.5, s=20, label='Not significant')

            plt.scatter(comp_data.loc[sig_mask, 'log2FC'],
                       -np.log10(comp_data.loc[sig_mask, 'adj_pvalue'].clip(lower=1e-300)),
                       c='red', alpha=0.7, s=30, label='Significant')

            plt.axhline(-np.log10(0.05), color='blue', linestyle='--', linewidth=1, label='adj_p=0.05')
            plt.axvline(-fc_thresh, color='green', linestyle='--', linewidth=1, label=f'FC=-{fc_thresh}')
            plt.axvline(fc_thresh, color='green', linestyle='--', linewidth=1, label=f'FC=+{fc_thresh}')

            plt.xlabel('log2 Fold Change', fontsize=12)
            plt.ylabel('-log10(adjusted p-value)', fontsize=12)
            plt.title(f'{molecule_type} Volcano Plot ({DE_METHOD}): {comp_name}', fontsize=14)
            plt.legend()
            plt.tight_layout()

            filename = f'{FIGURES_DIR}volcano_{molecule_type.lower()}_{comp_name}.png'
            plt.savefig(filename, dpi=150)
            plt.close()
            print(f"Saved {filename}")


def print_summary(mirna_filtered, mrna_filtered):
    print("\n" + "=" * 60)
    print(f"DIFFERENTIAL EXPRESSION SUMMARY ({DE_METHOD})")
    print("=" * 60)

    for comp_name in ['Normal_vs_Primary', 'Primary_vs_Metastatic', 'Normal_vs_Metastatic']:
        print(f"\n{comp_name}:")
        mirna_count = len(mirna_filtered[mirna_filtered['comparison'] == comp_name])
        mrna_count = len(mrna_filtered[mrna_filtered['comparison'] == comp_name])
        print(f"  DE miRNAs (|log2FC|>1): {mirna_count}")
        print(f"  DE mRNAs (|log2FC|>0.5): {mrna_count}")

    print(f"\nFiltering criteria: adj_pvalue < 0.05 AND |log2FC| > 1 (miRNAs) or > 0.5 (mRNAs)")
    print(f"DE method: {DE_METHOD}")


def main():
    print("=" * 60)
    print("GSE45498 Preprocessing and DE Analysis")
    print("=" * 60)

    mirna_to_mrna = build_sample_mapping()

    mirna_raw, mrna_raw = load_expression_data()

    sample_metadata = extract_sample_metadata(mirna_raw, mirna_to_mrna)
    print(f"\nTotal samples: {len(sample_metadata)}")
    print(f"Tissue distribution:\n{sample_metadata['tissue_group'].value_counts()}")

    mirna_df = separate_mirna_mrna(mirna_raw)
    mrna_df = separate_mrna_controls(mrna_raw)

    mirna_expr, mrna_expr = create_expression_matrices(mirna_df, mrna_df, sample_metadata, mirna_to_mrna)

    mirna_log2 = apply_log2_transform(mirna_expr)
    mrna_log2 = apply_log2_transform(mrna_expr)
    print("\nApplied log2(x+1) transformation")

    labels = sample_metadata.set_index('sample_id')['tissue_group']
    # True patient ids from the mRNA-platform titles (recovers matched
    # normal/tumor pairs and tumor replicates). NOT the miRNA pseudo-ids.
    patient_map = build_patient_id_map()
    n_patients = len(set(patient_map[s] for s in labels.index))
    print(f'Recovered {n_patients} unique patients across {len(labels)} samples '
          f'(true mRNA-title ids)')

    train_mirna_log2, test_mirna_log2, train_mrna_log2, test_mrna_log2, train_labels, test_labels = \
        patient_level_train_test_split(mirna_log2, mrna_log2, labels, patient_map)

    save_patient_and_split_artifacts(patient_map, train_labels, test_labels)

    mirna_de, mrna_de, mirna_filtered, mrna_filtered = run_de_analysis(
        mirna_expr, mrna_expr, train_labels
    )

    print("\nApplying Z-score standardization to training data for ML...")
    train_mirna_zscore, test_mirna_zscore = zscore_train_test(train_mirna_log2, test_mirna_log2)
    train_mrna_zscore, test_mrna_zscore = zscore_train_test(train_mrna_log2, test_mrna_log2)

    de_mirna_names = set(mirna_filtered['molecule_name'].unique())
    de_mrna_names = set(mrna_filtered['molecule_name'].unique())

    train_mirna_zscore = train_mirna_zscore[[c for c in train_mirna_zscore.columns if c in de_mirna_names]]
    test_mirna_zscore = test_mirna_zscore[[c for c in test_mirna_zscore.columns if c in de_mirna_names]]
    train_mrna_zscore = train_mrna_zscore[[c for c in train_mrna_zscore.columns if c in de_mrna_names]]
    test_mrna_zscore = test_mrna_zscore[[c for c in test_mrna_zscore.columns if c in de_mrna_names]]

    n_mirna = train_mirna_zscore.shape[1]
    n_mrna = train_mrna_zscore.shape[1]
    print(f"\nFiltered to {n_mirna} DE miRNAs and {n_mrna} DE mRNAs (total features: {n_mirna + n_mrna})")

    # Save the train/test splits BEFORE plotting, so a plotting failure can never
    # prevent the processed data from being written (previously a missing
    # results/figures dir crashed the script here and left stale split CSVs).
    save_splits(train_mirna_zscore, test_mirna_zscore, train_mrna_zscore, test_mrna_zscore, train_labels, test_labels)

    import os
    os.makedirs(FIGURES_DIR, exist_ok=True)
    try:
        create_volcano_plots(mirna_de, mrna_de,
                            ['Normal_vs_Primary', 'Primary_vs_Metastatic', 'Normal_vs_Metastatic'])
    except Exception as e:
        print(f"WARNING: volcano plots skipped ({e})")

    print_summary(mirna_filtered, mrna_filtered)

    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
