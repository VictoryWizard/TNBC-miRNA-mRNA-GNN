#!/usr/bin/env python3
"""TCGA-BRCA validation with ComBat batch correction (basal and clinical TNBC).

Batch-correction protocol (leakage-free):
  Batch correction uses *reference-batch* ComBat with the GSE45498 (training)
  cohort fixed as the reference batch (``REFERENCE_BATCH = 0``). The reference
  batch's location/scale are held fixed, so the TCGA (test) cohort is projected
  onto the GSE distribution rather than the two cohorts being centered jointly.
  This keeps the training distribution from being shifted by test samples.
  Gene-overlap / finiteness selection is derived from the GSE training matrix
  only (see ``select_overlap_genes``), then TCGA is subset to those genes, so no
  feature selection is performed on the test cohort.
"""

import gzip
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from combat.pycombat import pycombat
from scipy import stats

# GSE45498 (training) cohort is batch 0 and is used as the ComBat reference batch.
REFERENCE_BATCH = 0

PROCESSED_DIR = Path('data/processed')
DATA_DIR = Path('data')
RAW_DATA_DIR = DATA_DIR / 'raw'
RESULTS_DIR = Path('results')
NOTEBOOKS_DIR = Path(__file__).resolve().parent

TCGA_BASEL_PATH = DATA_DIR / 'tcga_brca_basal_mrna_expression.csv'
TCGA_CLINICAL_TNBC_PATH = DATA_DIR / 'tcga_brca_clinical_tnbc_mrna_expression.csv'
PANEL_PATH = DATA_DIR / 'tcga_gene_panel.txt'
MRNA_RAW_PATH = RAW_DATA_DIR / 'GSE45498_mRNA_non-normalized_data.txt.gz'
MIRNA_RAW_PATH = RAW_DATA_DIR / 'GSE45498_raw_data.txt.gz'
MIRNA_SERIES_MATRIX = RAW_DATA_DIR / 'GSE45498-GPL16231_series_matrix.txt.gz'
MRNA_SERIES_MATRIX = RAW_DATA_DIR / 'GSE45498-GPL16299_series_matrix.txt.gz'


@dataclass
class CohortMetrics:
    label: str
    n_tcga: int
    n_overlap: int
    n_de_overlap: int
    rho_pre: float
    p_pre: float
    rho_post_de: float
    p_post_de: float
    per_sample_median: float
    per_sample_q25: float
    per_sample_q75: float


def parse_series_matrix(filepath: Path) -> dict:
    """Parse GEO series matrix sample metadata (same logic as 01_preprocessing_DE)."""
    with gzip.open(filepath, 'rt') as f:
        lines = f.readlines()

    sample_info = {}
    for line in lines:
        if line.startswith('!Sample_title'):
            parts = line.strip().split('\t')
            for i, p in enumerate(parts[1:], 1):
                sample_info.setdefault(i, {})['title'] = p.strip().strip('"')
        elif line.startswith('!Sample_geo_accession'):
            parts = line.strip().split('\t')
            for i, p in enumerate(parts[1:], 1):
                sample_info.setdefault(i, {})['geo'] = p.strip().strip('"')
        elif line.startswith('!Sample_characteristics_ch1'):
            parts = line.strip().split('\t')
            for i, p in enumerate(parts[1:], 1):
                sample_info.setdefault(i, {})
                if 'tissue:' in p:
                    sample_info[i]['tissue'] = p.strip().strip('"')
                if 'status:' in p:
                    sample_info[i]['status'] = p.strip().strip('"')
    return sample_info


def build_mirna_to_mrna_mapping() -> dict:
    """Map miRNA sample column names to mRNA column names for paired GSE45498 samples."""
    mirna_meta = parse_series_matrix(MIRNA_SERIES_MATRIX)
    mrna_meta = parse_series_matrix(MRNA_SERIES_MATRIX)

    mirna_df = pd.read_csv(MIRNA_RAW_PATH, sep='\t', compression='gzip', nrows=0)
    mrna_df = pd.read_csv(MRNA_RAW_PATH, sep='\t', compression='gzip', nrows=0)

    mirna_raw_cols = list(mirna_df.columns[3:])
    mrna_raw_cols = [c.strip() for c in mrna_df.columns[3:]]

    mirna_to_mrna = {}
    for i in range(1, min(len(mirna_raw_cols), len(mrna_raw_cols)) + 1):
        if (
            mirna_meta.get(i, {}).get('tissue') == mrna_meta.get(i, {}).get('tissue')
            and mirna_meta.get(i, {}).get('status') == mrna_meta.get(i, {}).get('status')
        ):
            mirna_to_mrna[mirna_raw_cols[i - 1]] = mrna_raw_cols[i - 1]

    return mirna_to_mrna


def load_gse_mrna_log2() -> pd.DataFrame:
    """Load GSE mRNA from raw counts and apply log2(x+1) to match TCGA preprocessing."""
    if not MRNA_RAW_PATH.exists():
        raise FileNotFoundError(f'Missing raw mRNA file: {MRNA_RAW_PATH}')

    mirna_to_mrna = build_mirna_to_mrna_mapping()
    mirna_header = pd.read_csv(MIRNA_RAW_PATH, sep='\t', compression='gzip', nrows=0)
    mirna_sample_cols = list(mirna_header.columns[3:])
    sample_ids = [s for s in mirna_sample_cols if s in mirna_to_mrna]
    mrna_sample_ids_mapped = [mirna_to_mrna[s] for s in sample_ids]

    mrna_df = pd.read_csv(MRNA_RAW_PATH, sep='\t', compression='gzip')
    mrna_sample_cols = [c.strip() for c in mrna_df.columns[3:]]
    mrna_df.columns = list(mrna_df.columns[:3]) + mrna_sample_cols

    endogenous = mrna_df['Code Class'] == 'Endogenous'
    mrna_endogenous = mrna_df.loc[endogenous].copy()
    mrna_expr = mrna_endogenous.set_index('Name')[mrna_sample_ids_mapped].T
    mrna_expr.index = sample_ids

    numeric = mrna_expr.apply(pd.to_numeric, errors='coerce')
    mrna_log2 = np.log2(numeric + 1)
    print(
        f'Loaded GSE mRNA from raw: {mrna_log2.shape[0]} samples x '
        f'{mrna_log2.shape[1]} endogenous genes (log2(x+1), not z-scored)'
    )
    return mrna_log2


def load_de_mrna_genes() -> set:
    """DE mRNA symbols used by the classifier pipeline (from processed splits)."""
    train = pd.read_csv(PROCESSED_DIR / 'train_mrna.csv', index_col=0, nrows=0)
    test = pd.read_csv(PROCESSED_DIR / 'test_mrna.csv', index_col=0, nrows=0)
    return set(train.columns) | set(test.columns)


def load_tcga(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index.name = 'sample_id'
    df = df.loc[:, ~df.columns.duplicated()]
    return np.log2(df + 1)


def load_gene_panel() -> set:
    genes = PANEL_PATH.read_text().strip().split('\n')
    return set(g.strip() for g in genes if g.strip())


def ensure_clinical_tnbc_expression() -> None:
    if TCGA_CLINICAL_TNBC_PATH.exists():
        return
    script = NOTEBOOKS_DIR / 'download_tcga_clinical_tnbc.py'
    print(f'Clinical TNBC matrix missing; running {script.name}...')
    subprocess.run([sys.executable, str(script)], check=True)


def select_overlap_genes(gse_df: pd.DataFrame, genes: list) -> list:
    """Finiteness filtering derived from the GSE *training* matrix only.

    Bug fix (feature-selection-on-test): finiteness is checked on the GSE training
    cohort only; TCGA is later subset to these genes by the caller, so the test
    cohort never influences which features are kept.
    """
    genes = [g for g in genes if g in gse_df.columns]
    if not genes:
        return []
    gse_sub = gse_df[genes]
    ok = gse_sub.columns[gse_sub.notna().all() & np.isfinite(gse_sub).all()]
    return sorted(ok)


def drop_genes_with_missing(gse_df: pd.DataFrame, tcga_df: pd.DataFrame, genes: list) -> list:
    """Train-only finiteness selection, intersected with genes present in TCGA.

    Finiteness is decided on GSE (train) alone; TCGA only contributes column
    membership so the matrices can be aligned (no test-driven feature selection).
    """
    train_ok = select_overlap_genes(gse_df, genes)
    return [g for g in train_ok if g in tcga_df.columns]


def run_combat(gse_df: pd.DataFrame, tcga_df: pd.DataFrame, overlap_genes: list) -> pd.DataFrame:
    """Reference-batch ComBat: GSE (train) is the fixed reference; TCGA is projected onto it.

    Using ``ref_batch=REFERENCE_BATCH`` freezes the reference (training) batch's
    location/scale so corrected TCGA values are projected onto the GSE distribution
    instead of both cohorts being centered jointly (which would leak the test cohort
    into the correction). If the installed pyComBat lacks ``ref_batch`` support we
    fall back to standard ComBat and emit a clear leakage warning.
    """
    gse_overlap = gse_df[overlap_genes]
    tcga_overlap = tcga_df[overlap_genes]

    combined = pd.concat([gse_overlap, tcga_overlap], axis=0)
    batch = np.array([REFERENCE_BATCH] * len(gse_overlap) + [1] * len(tcga_overlap))

    data_t = combined.T
    data_t.columns = combined.index
    try:
        corrected_t = pycombat(data=data_t, batch=batch, ref_batch=REFERENCE_BATCH)
    except TypeError:
        print(
            'WARNING: installed pyComBat has no ref_batch support; falling back to '
            'joint ComBat. This reintroduces transductive leakage (corrected TCGA '
            'values depend on the test cohort). Upgrade to inmoose.pycombat for a '
            'frozen reference-batch fit.'
        )
        corrected_t = pycombat(data=data_t, batch=batch)
    corrected = corrected_t.T
    corrected.columns = overlap_genes
    return corrected


def per_gene_mean_spearman(
    gse_expr: pd.DataFrame, tcga_expr: pd.DataFrame, genes: list
) -> tuple[float, float]:
    """Spearman correlation of per-gene mean expression (GSE vs TCGA)."""
    gse_mean = gse_expr[genes].mean(axis=0)
    tcga_mean = tcga_expr[genes].mean(axis=0)
    rho, pval = stats.spearmanr(gse_mean.values, tcga_mean.values)
    return float(rho), float(pval)


def per_sample_profile_correlations(
    corrected: pd.DataFrame, n_gse: int, genes: list
) -> tuple[float, float, float]:
    """Per TCGA sample: Spearman(profile, mean GSE profile) after ComBat."""
    gse_corrected = corrected.iloc[:n_gse][genes]
    tcga_corrected = corrected.iloc[n_gse:][genes]
    gse_mean_profile = gse_corrected.mean(axis=0)

    corrs = np.array(
        [
            stats.spearmanr(gse_mean_profile.values, tcga_corrected.loc[idx].values)[0]
            for idx in tcga_corrected.index
        ]
    )
    q25, median, q75 = np.percentile(corrs, [25, 50, 75])
    return float(median), float(q25), float(q75)


def validate_cohort(
    gse_df: pd.DataFrame,
    tcga_df: pd.DataFrame,
    panel_genes: set,
    de_mrna_genes: set,
    label: str,
) -> CohortMetrics:
    gse_genes = set(gse_df.columns)
    tcga_genes = set(tcga_df.columns)
    overlap = sorted(gse_genes & tcga_genes & panel_genes)
    overlap = drop_genes_with_missing(gse_df, tcga_df, overlap)
    de_in_overlap = sorted(de_mrna_genes & set(overlap))

    if len(overlap) < 2:
        raise ValueError(f'{label}: fewer than 2 overlapping genes.')
    if len(de_in_overlap) < 2:
        raise ValueError(f'{label}: fewer than 2 DE genes in overlap.')

    rho_pre, p_pre = per_gene_mean_spearman(gse_df, tcga_df, overlap)
    corrected = run_combat(gse_df, tcga_df, overlap)
    n_gse = len(gse_df)
    gse_corrected = corrected.iloc[:n_gse]
    tcga_corrected = corrected.iloc[n_gse:]
    rho_post_de, p_post_de = per_gene_mean_spearman(gse_corrected, tcga_corrected, de_in_overlap)
    med, q25, q75 = per_sample_profile_correlations(corrected, n_gse, overlap)

    return CohortMetrics(
        label=label,
        n_tcga=len(tcga_df),
        n_overlap=len(overlap),
        n_de_overlap=len(de_in_overlap),
        rho_pre=rho_pre,
        p_pre=p_pre,
        rho_post_de=rho_post_de,
        p_post_de=p_post_de,
        per_sample_median=med,
        per_sample_q25=q25,
        per_sample_q75=q75,
    )


def print_metrics_table(metrics: list[CohortMetrics], de_total: int) -> None:
    col_w = 28
    headers = ['Metric'] + [m.label for m in metrics]
    print('\n' + ''.join(h.ljust(col_w) for h in headers))
    print('-' * (col_w * len(headers)))

    def row(name: str, values: list[str]) -> None:
        print(name.ljust(col_w) + ''.join(v.ljust(col_w) for v in values))

    row('TCGA samples', [str(m.n_tcga) for m in metrics])
    row('Genes for ComBat', [str(m.n_overlap) for m in metrics])
    row('DE genes in overlap', [f'{m.n_de_overlap}/{de_total}' for m in metrics])
    row(
        'Pre-ComBat rho (all overlap)',
        [f'{m.rho_pre:.4f} (p={m.p_pre:.2e})' for m in metrics],
    )
    row(
        'Post-ComBat rho (DE only)',
        [f'{m.rho_post_de:.4f} (p={m.p_post_de:.2e})' for m in metrics],
    )
    row(
        'Per-sample rho median [IQR]',
        [
            f'{m.per_sample_median:.4f} [{m.per_sample_q25:.4f}, {m.per_sample_q75:.4f}]'
            for m in metrics
        ],
    )


def write_summary(
    metrics: list[CohortMetrics],
    n_gse: int,
    n_gse_genes: int,
    de_total: int,
    panel_size: int,
) -> None:
    lines = [
        'TCGA-BRCA Validation Summary (GSE45498 vs TCGA)',
        '=' * 72,
        f'GSE45498 samples:                    {n_gse}',
        f'GSE endogenous mRNA genes (log2):    {n_gse_genes}',
        f'DE mRNAs in classifier pipeline:     {de_total}',
        f'NanoString reference panel:          {panel_size}',
        f'GSE expression scale:                log2(x+1) from raw',
        '',
    ]
    for metric in metrics:
        lines.extend(
            [
                metric.label,
                '-' * 72,
                f'  TCGA samples:                         {metric.n_tcga}',
                f'  Overlapping genes (ComBat):           {metric.n_overlap}',
                f'  DE mRNAs in overlap:                  {metric.n_de_overlap}/{de_total}',
                f'  Pre-ComBat Spearman rho (all overlap): {metric.rho_pre:.4f} (p={metric.p_pre:.2e})',
                f'  Post-ComBat Spearman rho (DE only):    {metric.rho_post_de:.4f} '
                f'(p={metric.p_post_de:.2e})',
                f'  Per-sample profile rho median [IQR]:   {metric.per_sample_median:.4f} '
                f'[{metric.per_sample_q25:.4f}, {metric.per_sample_q75:.4f}]',
                '',
            ]
        )
    if len(metrics) == 2:
        lines.append('Cohort note:')
        lines.append(
            '  PAM50 basal = PanCan Subtype BRCA_Basal (molecular proxy, not pure IHC TNBC).'
        )
        lines.append(
            '  Clinical TNBC = ER-/PR-/HER2- by IHC/FISH (brca_tcga clinical, PanCan expression).'
        )
        lines.append('')

    out = RESULTS_DIR / 'tcga_validation_summary.txt'
    out.write_text('\n'.join(lines))


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not TCGA_BASEL_PATH.exists():
        raise FileNotFoundError(f'Missing basal TCGA matrix: {TCGA_BASEL_PATH}')

    ensure_clinical_tnbc_expression()

    gse_df = load_gse_mrna_log2()
    panel_genes = load_gene_panel()
    de_mrna_genes = load_de_mrna_genes()

    cohorts = [
        ('PAM50 basal (171)', TCGA_BASEL_PATH),
        ('Clinical TNBC (IHC)', TCGA_CLINICAL_TNBC_PATH),
    ]

    all_metrics: list[CohortMetrics] = []
    for label, path in cohorts:
        print(f'\n=== {label} ===')
        tcga_df = load_tcga(path)
        print(f'TCGA samples: {len(tcga_df)}, genes: {tcga_df.shape[1]}')
        print('Running ComBat and metrics...')
        metrics = validate_cohort(gse_df, tcga_df, panel_genes, de_mrna_genes, label)
        all_metrics.append(metrics)
        print(
            f'  Pre-ComBat rho ({metrics.n_overlap} genes): {metrics.rho_pre:.4f} '
            f'(p={metrics.p_pre:.2e})'
        )
        print(
            f'  Post-ComBat DE rho ({metrics.n_de_overlap} genes): {metrics.rho_post_de:.4f} '
            f'(p={metrics.p_post_de:.2e})'
        )
        print(
            f'  Per-sample median rho: {metrics.per_sample_median:.4f} '
            f'IQR=[{metrics.per_sample_q25:.4f}, {metrics.per_sample_q75:.4f}]'
        )

    print_metrics_table(all_metrics, len(de_mrna_genes))
    write_summary(all_metrics, len(gse_df), gse_df.shape[1], len(de_mrna_genes), len(panel_genes))
    print(f'\nSaved to {RESULTS_DIR / "tcga_validation_summary.txt"}')


if __name__ == '__main__':
    main()
