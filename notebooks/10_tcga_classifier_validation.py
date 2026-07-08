#!/usr/bin/env python3
"""TCGA-BRCA classifier validation: predict on external TCGA samples (not correlation).

Compares four approaches:
  A) Pretrained XGBoost (full GSE45498 feature set; missing TCGA miRNA imputed at 0)
  B) XGBoost retrained on GSE45498 using only mRNA genes overlapping with TCGA panel
  C) XGBoost retrained on top-10 mRNA genes by feature_importances_ (TCGA-overlapping)
  D) Simple DE gene-signature score (top up/down genes, threshold at 0)

TCGA cohort: PAM50 basal-like primary tumors vs solid tissue normal (sample type 11).
This is a 2-class external validation (normal vs primary); metastatic class is absent
in TCGA-BRCA and predictions of metastatic are collapsed to primary for reporting.

Leakage / confound controls:
  * Batch correction uses *reference-batch* ComBat with GSE45498 (training) fixed
    as the reference batch (``REFERENCE_BATCH = 0``); TCGA is projected onto the
    GSE distribution rather than the two cohorts being centered jointly. See
    ``run_combat``.
  * Gene-overlap / finiteness selection is derived from the GSE training matrix
    only (``select_overlap_genes``); TCGA is then subset to those genes so no
    feature selection touches the test cohort.
  * Class-vs-source confound: the TCGA normal and tumor matrices must be put on a
    comparable scale BEFORE harmonization, otherwise the classifier can separate
    classes on a normalization artifact rather than biology. We apply a uniform
    within-cohort CPM (library-size) normalization to both raw count matrices via
    ``cpm_log2`` so normals and tumors are comparable. RESIDUAL CONFOUND: ideally
    both classes should come from a single uniform expression source (same
    pipeline/quantifier); CPM only equalizes library size, not quantifier or
    pipeline differences. Prefer one uniform source when available.
"""

from __future__ import annotations

import gzip
import io
import json
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

_NOTEBOOK_DIR = Path(__file__).resolve().parent
if str(_NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOK_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from combat.pycombat import pycombat
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from xgboost import XGBClassifier

import xgboost as xgb_lib
from cv_utils import load_patient_groups

PROCESSED_DIR = Path('data/processed')
DATA_DIR = Path('data')
RAW_DATA_DIR = DATA_DIR / 'raw'
RESULTS_DIR = Path('results')
TABLES_DIR = RESULTS_DIR / 'tables'
FIGURES_DIR = RESULTS_DIR / 'figures'
MODELS_DIR = Path('models')

TCGA_BASEL_PATH = DATA_DIR / 'tcga_brca_basal_mrna_expression.csv'
TCGA_NORMAL_PATH = DATA_DIR / 'tcga_brca_normal_mrna_expression.csv'
TCGA_BASAL_GDC_PATH = DATA_DIR / 'tcga_brca_basal_gdc_star_expression.csv'
PANEL_PATH = DATA_DIR / 'tcga_gene_panel.txt'
ENTREZ_CACHE_PATH = DATA_DIR / 'tcga_gene_entrez_map.json'
MRNA_RAW_PATH = RAW_DATA_DIR / 'GSE45498_mRNA_non-normalized_data.txt.gz'
MIRNA_RAW_PATH = RAW_DATA_DIR / 'GSE45498_raw_data.txt.gz'
MIRNA_SERIES_MATRIX = RAW_DATA_DIR / 'GSE45498-GPL16231_series_matrix.txt.gz'
MRNA_SERIES_MATRIX = RAW_DATA_DIR / 'GSE45498-GPL16299_series_matrix.txt.gz'
MODEL_PATH = MODELS_DIR / 'xgboost_model.pkl'
DE_MRNA_PATH = PROCESSED_DIR / 'de_mrnas.csv'
TOP_MRNA_N = 10
DE_COMPARISON = 'Normal_vs_Primary'

# GSE45498 (training) cohort is batch 0 and is used as the ComBat reference batch.
REFERENCE_BATCH = 0

CBIO_BASE = 'https://www.cbioportal.org/api'
EXPRESSION_STUDY = 'brca_tcga_pan_can_atlas_2018'
MOLECULAR_PROFILE = f'{EXPRESSION_STUDY}_rna_seq_v2_mrna'

LABEL_MAP = {'normal': 0, 'primary': 1, 'metastatic': 2}
TCGA_BINARY_NAMES = ('normal', 'primary')
TCGA_BINARY_IDS = (0, 1)

XGB_PARAM_GRID = {
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
}


@dataclass
class ApproachResult:
    approach: str
    metrics: Dict
    n_tcga_normal: int
    n_tcga_tumor: int
    n_overlap_genes: int
    notes: str


def _xgb_gpu_kwargs() -> dict:
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        import shutil
        cuda_available = shutil.which('nvidia-smi') is not None

    if not cuda_available:
        return {}

    major = int(xgb_lib.__version__.split('.')[0])
    if major >= 2:
        return {'device': 'cuda'}
    return {'tree_method': 'gpu_hist', 'predictor': 'gpu_predictor'}


def parse_series_matrix(filepath: Path) -> dict:
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
    """Load all GSE45498 mRNA samples as log2(x+1) (same scale as TCGA in notebook 07)."""
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
        f'GSE45498 mRNA: {mrna_log2.shape[0]} samples x '
        f'{mrna_log2.shape[1]} endogenous genes (log2(x+1))'
    )
    return mrna_log2


def load_gse45498_training_data() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load processed miRNA+mRNA training matrix and labels (same as notebook 04)."""
    x_train = pd.read_csv(PROCESSED_DIR / 'train_mirna.csv', index_col=0)
    x_train_mrna = pd.read_csv(PROCESSED_DIR / 'train_mrna.csv', index_col=0)
    train_labels = pd.read_csv(PROCESSED_DIR / 'train_labels.csv', index_col='sample_id')[
        'tissue_group'
    ]

    train_labels = train_labels.loc[x_train.index]
    x_train = x_train.loc[train_labels.index]
    x_train_mrna = x_train_mrna.loc[train_labels.index]

    X_train = pd.concat([x_train, x_train_mrna], axis=1)
    y_train = train_labels.map(LABEL_MAP).astype(int)
    return X_train, y_train, train_labels.index.tolist()


def load_overlap_mrna_training_data() -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """GSE45498 training matrix restricted to mRNA genes overlapping TCGA panel."""
    _, _, train_ids = load_gse45498_training_data()
    x_train_mrna = pd.read_csv(PROCESSED_DIR / 'train_mrna.csv', index_col=0)
    train_labels = pd.read_csv(PROCESSED_DIR / 'train_labels.csv', index_col='sample_id')[
        'tissue_group'
    ]
    train_labels = train_labels.loc[x_train_mrna.index]

    panel_genes = load_gene_panel()
    overlap_genes = sorted(set(x_train_mrna.columns) & panel_genes)
    X_train = x_train_mrna.loc[train_ids, overlap_genes]
    y_train = train_labels.loc[train_ids].map(LABEL_MAP).astype(int)
    return X_train, y_train, train_ids, overlap_genes


def _api_get(url: str, timeout: int = 60) -> object:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def _api_post(url: str, body: object, timeout: int = 180) -> object:
    payload = json.dumps(body).encode()
    request = urllib.request.Request(
        url,
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def load_gene_panel() -> set[str]:
    genes = PANEL_PATH.read_text().strip().split('\n')
    return {g.strip() for g in genes if g.strip()}


def load_entrez_map(symbols: list[str]) -> dict[str, int]:
    if ENTREZ_CACHE_PATH.exists():
        cached = json.loads(ENTREZ_CACHE_PATH.read_text())
        missing = [s for s in symbols if s not in cached]
    else:
        cached = {}
        missing = list(symbols)

    for symbol in missing:
        for attempt in range(3):
            try:
                gene = _api_get(f'{CBIO_BASE}/genes/{symbol}', timeout=30)
                cached[symbol] = int(gene['entrezGeneId'])
                break
            except urllib.error.HTTPError:
                break
            except urllib.error.URLError:
                time.sleep(2 * (attempt + 1))
        time.sleep(0.05)

    ENTREZ_CACHE_PATH.write_text(json.dumps(cached, indent=2, sort_keys=True))
    return {symbol: cached[symbol] for symbol in symbols if symbol in cached}


def fetch_expression_matrix(
    sample_ids: list[str], symbols: list[str], sample_chunk: int = 25, gene_chunk: int = 80
) -> pd.DataFrame:
    entrez_map = load_entrez_map(symbols)
    entrez_ids = [entrez_map[s] for s in symbols if s in entrez_map]
    id_to_symbol = {entrez_map[s]: s for s in symbols if s in entrez_map}

    records: list[dict] = []
    url = f'{CBIO_BASE}/molecular-profiles/{MOLECULAR_PROFILE}/molecular-data/fetch'
    for sample_start in range(0, len(sample_ids), sample_chunk):
        sample_batch = sample_ids[sample_start : sample_start + sample_chunk]
        for gene_start in range(0, len(entrez_ids), gene_chunk):
            gene_batch = entrez_ids[gene_start : gene_start + gene_chunk]
            body = {'sampleIds': sample_batch, 'entrezGeneIds': gene_batch}
            batch = _api_post(url, body, timeout=300)
            records.extend(batch)

    if not records:
        raise RuntimeError('No expression values returned from cBioPortal.')

    long = pd.DataFrame(records)
    matrix = long.pivot(index='sampleId', columns='entrezGeneId', values='value')
    matrix.columns = [id_to_symbol[int(col)] for col in matrix.columns]
    matrix = matrix.reindex(columns=[s for s in symbols if s in matrix.columns])
    matrix.index.name = 'sample_id'
    return matrix


def fetch_gdc_normal_file_index() -> pd.DataFrame:
    """List GDC STAR count files for TCGA-BRCA solid tissue normal samples."""
    filters = {
        'op': 'and',
        'content': [
            {
                'op': 'in',
                'content': {'field': 'cases.project.project_id', 'value': ['TCGA-BRCA']},
            },
            {
                'op': 'in',
                'content': {
                    'field': 'cases.samples.sample_type',
                    'value': ['Solid Tissue Normal'],
                },
            },
            {
                'op': 'in',
                'content': {'field': 'files.data_type', 'value': ['Gene Expression Quantification']},
            },
        ],
    }
    params = urllib.parse.urlencode(
        {
            'filters': json.dumps(filters),
            'fields': 'file_id,cases.samples.submitter_id',
            'size': '500',
            'format': 'json',
        }
    )
    with urllib.request.urlopen(f'https://api.gdc.cancer.gov/files?{params}', timeout=120) as response:
        payload = json.loads(response.read())

    rows = []
    for hit in payload['data']['hits']:
        submitter = hit['cases'][0]['samples'][0]['submitter_id']
        # TCGA-XX-XXXX-11A -> TCGA-XX-XXXX-11 (match basal matrix sample_id style)
        sample_id = submitter[:15] if submitter.startswith('TCGA-') else submitter
        rows.append({'file_id': hit['file_id'], 'sample_id': sample_id})

    frame = pd.DataFrame(rows).drop_duplicates('sample_id', keep='first')
    if frame.empty:
        raise RuntimeError('No TCGA solid tissue normal expression files found in GDC.')
    return frame


def _parse_gdc_counts(text: str, symbols: set[str]) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith('#') or line.startswith('N_') or line.startswith('gene_id'):
            continue
        parts = line.split('\t')
        if len(parts) < 4:
            continue
        gene_name = parts[1]
        if gene_name in symbols:
            values[gene_name] = float(parts[3])  # unstranded counts (RSEM-comparable scale)
    return values


def download_gdc_normal_expression(symbols: list[str], batch_size: int = 20) -> pd.DataFrame:
    """Download TCGA-BRCA normal unstranded counts from GDC and cache locally."""
    index = fetch_gdc_normal_file_index()
    symbol_set = set(symbols)
    records: dict[str, dict[str, float]] = {}

    file_ids = index['file_id'].tolist()
    sample_ids = index['sample_id'].tolist()
    print(f'Downloading {len(file_ids)} TCGA normal samples from GDC...')

    for start in range(0, len(file_ids), batch_size):
        batch_ids = file_ids[start : start + batch_size]
        batch_samples = sample_ids[start : start + batch_size]
        request = urllib.request.Request(
            'https://api.gdc.cancer.gov/data',
            data=json.dumps({'ids': batch_ids}).encode(),
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            tar = tarfile.open(fileobj=io.BytesIO(response.read()))
            members_by_id = {
                m.name.split('/')[0]: m
                for m in tar.getmembers()
                if m.isfile() and '/' in m.name
            }
            for file_id, sample_id in zip(batch_ids, batch_samples):
                member = members_by_id.get(file_id)
                if member is None:
                    continue
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                text = extracted.read().decode('utf-8')
                records[sample_id] = _parse_gdc_counts(text, symbol_set)

    if not records:
        raise RuntimeError('Failed to download TCGA normal expression from GDC.')

    matrix = pd.DataFrame.from_dict(records, orient='index')
    matrix = matrix.reindex(columns=[s for s in symbols if s in matrix.columns])
    matrix.index.name = 'sample_id'
    return matrix.sort_index()


def fetch_gdc_tumor_file_index(basal_ids: list[str]) -> pd.DataFrame:
    """GDC STAR-count files for TCGA-BRCA primary tumors, restricted to PAM50-basal barcodes."""
    filters = {
        'op': 'and',
        'content': [
            {'op': 'in', 'content': {'field': 'cases.project.project_id', 'value': ['TCGA-BRCA']}},
            {'op': 'in', 'content': {'field': 'cases.samples.sample_type', 'value': ['Primary Tumor']}},
            {'op': 'in', 'content': {'field': 'files.data_type', 'value': ['Gene Expression Quantification']}},
        ],
    }
    params = urllib.parse.urlencode({
        'filters': json.dumps(filters),
        'fields': 'file_id,cases.samples.submitter_id',
        'size': '2000', 'format': 'json',
    })
    with urllib.request.urlopen(f'https://api.gdc.cancer.gov/files?{params}', timeout=180) as response:
        payload = json.loads(response.read())
    basal_set = {s[:15] if s.startswith('TCGA-') else s for s in basal_ids}
    rows = []
    for hit in payload['data']['hits']:
        submitter = hit['cases'][0]['samples'][0]['submitter_id']
        sample_id = submitter[:15] if submitter.startswith('TCGA-') else submitter
        if sample_id in basal_set:
            rows.append({'file_id': hit['file_id'], 'sample_id': sample_id})
    frame = pd.DataFrame(rows).drop_duplicates('sample_id', keep='first')
    if frame.empty:
        raise RuntimeError('No basal primary-tumor STAR files matched in GDC — check basal IDs are TCGA-XX-XXXX-01 style.')
    return frame


def download_gdc_tumor_expression(symbols: list[str], basal_ids: list[str], batch_size: int = 20) -> pd.DataFrame:
    """Basal primary-tumor unstranded STAR counts from GDC (mirrors download_gdc_normal_expression)."""
    index = fetch_gdc_tumor_file_index(basal_ids)
    symbol_set = set(symbols)
    records: dict[str, dict[str, float]] = {}
    file_ids, sample_ids = index['file_id'].tolist(), index['sample_id'].tolist()
    print(f'Downloading {len(file_ids)} basal TCGA tumor samples from GDC...')
    for start in range(0, len(file_ids), batch_size):
        batch_ids = file_ids[start:start + batch_size]
        batch_samples = sample_ids[start:start + batch_size]
        request = urllib.request.Request(
            'https://api.gdc.cancer.gov/data',
            data=json.dumps({'ids': batch_ids}).encode(),
            headers={'Content-Type': 'application/json'}, method='POST',
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            tar = tarfile.open(fileobj=io.BytesIO(response.read()))
            members_by_id = {m.name.split('/')[0]: m for m in tar.getmembers() if m.isfile() and '/' in m.name}
            for file_id, sample_id in zip(batch_ids, batch_samples):
                member = members_by_id.get(file_id)
                if member is None:
                    continue
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                records[sample_id] = _parse_gdc_counts(extracted.read().decode('utf-8'), symbol_set)
    if not records:
        raise RuntimeError('Failed to download basal tumor expression from GDC.')
    matrix = pd.DataFrame.from_dict(records, orient='index')
    matrix = matrix.reindex(columns=[s for s in symbols if s in matrix.columns])
    matrix.index.name = 'sample_id'
    return matrix.sort_index()


def ensure_tcga_basal_gdc_expression() -> None:
    if TCGA_BASAL_GDC_PATH.exists():
        return
    print('Basal GDC tumor matrix missing; downloading primary-tumor STAR counts from GDC...')
    basal_ids = load_tcga_mrna(TCGA_BASEL_PATH).index.astype(str).tolist()  # PAM50-basal labels from cBioPortal
    expression = download_gdc_tumor_expression(sorted(load_gene_panel()), basal_ids)
    expression.to_csv(TCGA_BASAL_GDC_PATH)


def fetch_normal_sample_ids() -> list[str]:
    """Return TCGA solid tissue normal sample IDs (sample type code 11)."""
    return fetch_gdc_normal_file_index()['sample_id'].tolist()


def ensure_tcga_normal_expression() -> None:
    if TCGA_NORMAL_PATH.exists():
        return

    print('TCGA normal matrix missing; downloading solid tissue normal samples from GDC...')
    symbols = sorted(load_gene_panel())
    expression = download_gdc_normal_expression(symbols)
    expression.to_csv(TCGA_NORMAL_PATH)
    print(
        f'Saved {expression.shape[0]} normal samples x {expression.shape[1]} genes '
        f'to {TCGA_NORMAL_PATH}'
    )


def load_tcga_mrna(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index.name = 'sample_id'
    df = df.loc[:, ~df.columns.duplicated()]
    return np.log2(df + 1)


def cpm_log2(path: Path) -> pd.DataFrame:
    """Load a raw STAR-count matrix and return log2(CPM + 1).

    Confound fix (class confounded with data source/normalization): the TCGA
    normal and tumor matrices are raw STAR counts that can differ in library
    size, so a plain log2(count+1) leaves them on incomparable scales and lets a
    classifier separate normal vs tumor on a technical artifact. We library-size
    normalize each sample to counts-per-million before log2 so normals and tumors
    are on a comparable within-cohort scale prior to harmonization. (Residual
    confound: CPM does not remove quantifier/pipeline differences; a single
    uniform expression source for both classes is preferable.)
    """
    df = pd.read_csv(path, index_col=0)
    df.index.name = 'sample_id'
    df = df.loc[:, ~df.columns.duplicated()]
    counts = df.apply(pd.to_numeric, errors='coerce')
    lib_size = counts.sum(axis=1).replace(0, np.nan)
    cpm = counts.div(lib_size, axis=0) * 1e6
    return np.log2(cpm + 1)


def load_tcga_validation_cohort() -> tuple[pd.DataFrame, pd.Series]:
    """PAM50 basal primary tumors + solid tissue normals (2-class labels).

    Both matrices are raw STAR counts; CPM-normalize each uniformly (cpm_log2) so
    class is not confounded with library size / normalization scale.
    """
    if not TCGA_BASEL_PATH.exists():
        raise FileNotFoundError(f'Missing basal TCGA matrix: {TCGA_BASEL_PATH}')
    ensure_tcga_normal_expression()
    ensure_tcga_basal_gdc_expression()

    # Consistent within-cohort CPM normalization for BOTH classes before harmonization.
    tumor = cpm_log2(TCGA_BASAL_GDC_PATH)
    normal = cpm_log2(TCGA_NORMAL_PATH)

    tumor.index = tumor.index.astype(str)
    normal.index = normal.index.astype(str)

    combined = pd.concat([normal, tumor], axis=0)
    labels = pd.Series(
        [0] * len(normal) + [1] * len(tumor),
        index=combined.index,
        name='tcga_label',
    )
    print(
        f'TCGA validation cohort: {len(normal)} normal, {len(tumor)} basal primary '
        f'({len(combined)} total)'
    )
    return combined, labels


def drop_genes_with_missing(gse_df: pd.DataFrame, tcga_df: pd.DataFrame, genes: list) -> list:
    """Train-only finiteness selection, intersected with genes present in TCGA.

    Bug fix (feature-selection-on-test): finiteness is decided on the GSE training
    cohort only; TCGA only contributes column membership for alignment, so the test
    cohort never drives which features are kept.
    """
    genes = [g for g in genes if g in gse_df.columns]
    if not genes:
        return []
    gse_sub = gse_df[genes]
    ok = gse_sub.columns[gse_sub.notna().all() & np.isfinite(gse_sub).all()]
    return [g for g in sorted(ok) if g in tcga_df.columns]


def run_combat(gse_df: pd.DataFrame, tcga_df: pd.DataFrame, overlap_genes: list) -> pd.DataFrame:
    """Reference-batch ComBat: GSE (train) is the fixed reference; TCGA is projected onto it.

    Using ``ref_batch=REFERENCE_BATCH`` freezes the reference (training) batch's
    location/scale so corrected TCGA values are projected onto the GSE distribution
    rather than both cohorts being centered jointly (which leaks the test cohort into
    the correction). Falls back to standard ComBat with a clear warning if the
    installed pyComBat lacks ``ref_batch`` support.
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


def zscore_from_gse_train(
    corrected: pd.DataFrame,
    train_ids: Sequence[str],
    genes: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit z-score on GSE45498 training samples; return (gse_train_z, params)."""
    train_expr = corrected.loc[list(train_ids), genes]
    mean = train_expr.mean()
    std = train_expr.std().replace(0, 1.0)
    gse_train_z = (train_expr - mean) / std
    return gse_train_z, pd.DataFrame({'mean': mean, 'std': std})


def apply_zscore(expr: pd.DataFrame, genes: Sequence[str], params: pd.DataFrame) -> pd.DataFrame:
    mean = params.loc[list(genes), 'mean']
    std = params.loc[list(genes), 'std']
    return (expr[genes] - mean) / std


def prepare_tcga_mrna_features(
    gse_log2: pd.DataFrame,
    tcga_log2: pd.DataFrame,
    train_ids: Sequence[str],
    genes: Sequence[str],
) -> tuple[pd.DataFrame, list[str]]:
    """ComBat + train-fitted z-score for TCGA mRNA features."""
    overlap = drop_genes_with_missing(gse_log2, tcga_log2, list(genes))
    if len(overlap) < 2:
        raise ValueError('Fewer than 2 overlapping genes with finite values in both cohorts.')

    corrected = run_combat(gse_log2, tcga_log2, overlap)
    n_gse = len(gse_log2)
    tcga_corrected = corrected.iloc[n_gse:]
    _, z_params = zscore_from_gse_train(corrected.iloc[:n_gse], train_ids, overlap)
    tcga_z = apply_zscore(tcga_corrected, overlap, z_params)
    return tcga_z, overlap


def build_full_model_tcga_matrix(
    model: XGBClassifier,
    tcga_mrna_z: pd.DataFrame,
    overlap_genes: Sequence[str],
) -> pd.DataFrame:
    """Align TCGA samples to full model feature space (miRNA imputed at 0)."""
    feature_names = list(model.feature_names_in_)
    X = pd.DataFrame(0.0, index=tcga_mrna_z.index, columns=feature_names)

    for gene in overlap_genes:
        if gene in X.columns:
            X[gene] = tcga_mrna_z[gene].values
    return X


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    run_grid_search: bool = False,
    best_params: dict | None = None,
) -> tuple[XGBClassifier, dict]:
    """Train 3-class XGBoost with optional grid search (same grid as notebook 04)."""
    gpu_kwargs = _xgb_gpu_kwargs()
    base_model = XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        **gpu_kwargs,
    )

    groups = load_patient_groups(X_train.index).values

    if run_grid_search or best_params is None:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(
            base_model,
            param_grid=XGB_PARAM_GRID,
            scoring='f1_macro',
            cv=cv,
            n_jobs=-1,
            verbose=1,
            return_train_score=True,
        )
        grid.fit(X_train, y_train, groups=groups)
        best_params = dict(grid.best_params_)
        print(f'Best parameters: {best_params}')
        print(f'Best CV macro F1: {grid.best_score_:.4f}')
        model = grid.best_estimator_
    else:
        model = base_model
        model.set_params(**best_params)

    model.fit(X_train, y_train)
    return model, best_params


def load_or_train_full_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[XGBClassifier, dict]:
    if MODEL_PATH.exists():
        print(f'Loading pretrained model from {MODEL_PATH}')
        model = pd.read_pickle(MODEL_PATH)
        params = {
            k: getattr(model, k)
            for k in ('max_depth', 'n_estimators', 'learning_rate')
            if hasattr(model, k)
        }
        return model, params

    print('No saved model found; training XGBoost on GSE45498 training set...')
    model, best_params = train_xgboost_model(X_train, y_train, run_grid_search=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(model, MODEL_PATH)
    print(f'Saved model to {MODEL_PATH}')
    return model, best_params


def collapse_to_binary_predictions(y_pred: np.ndarray) -> np.ndarray:
    """Map 3-class predictions to 2-class TCGA labels (metastatic -> primary)."""
    return np.where(y_pred == 0, 0, 1)


def binary_probabilities(y_prob_3class: np.ndarray) -> np.ndarray:
    """Combine primary + metastatic probabilities for binary tumor class."""
    return np.column_stack([y_prob_3class[:, 0], y_prob_3class[:, 1] + y_prob_3class[:, 2]])


def _binary_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score_primary: np.ndarray,
    *,
    n_metastatic_predictions: int = 0,
) -> Dict:
    """Shared 2-class metrics given binary preds and a continuous tumor score."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score_primary = np.asarray(y_score_primary)
    classes = np.array(TCGA_BINARY_IDS)
    class_names = list(TCGA_BINARY_NAMES)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    per_class_f1_vals = f1_score(
        y_true, y_pred, average=None, labels=classes, zero_division=0
    )
    per_class_f1 = {
        name: float(score) for name, score in zip(class_names, per_class_f1_vals)
    }

    y_true_bin = np.column_stack([(y_true == c).astype(int) for c in classes])
    per_class_auc = {}
    score_normal = -y_score_primary
    for idx, name in enumerate(class_names):
        score = y_score_primary if name == 'primary' else score_normal
        if y_true_bin[:, idx].sum() in (0, len(y_true_bin)):
            per_class_auc[name] = float('nan')
        else:
            per_class_auc[name] = float(roc_auc_score(y_true_bin[:, idx], score))

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    return {
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'per_class_f1': per_class_f1,
        'per_class_auc': per_class_auc,
        'confusion_matrix': cm,
        'class_names': class_names,
        'n_metastatic_predictions': n_metastatic_predictions,
    }


def compute_tcga_binary_metrics(
    y_true: np.ndarray,
    y_pred_3class: np.ndarray,
    y_prob_3class: np.ndarray,
) -> Dict:
    """Accuracy, macro F1, per-class F1/AUROC for 2-class TCGA validation."""
    y_pred = collapse_to_binary_predictions(np.asarray(y_pred_3class))
    y_prob = binary_probabilities(np.asarray(y_prob_3class))
    n_metastatic_preds = int((np.asarray(y_pred_3class) == 2).sum())
    return _binary_classification_metrics(
        y_true,
        y_pred,
        y_prob[:, 1],
        n_metastatic_predictions=n_metastatic_preds,
    )


def compute_tcga_direct_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> Dict:
    """Binary metrics for direct 2-class predictions (e.g. signature threshold)."""
    return _binary_classification_metrics(
        y_true,
        np.asarray(y_pred),
        np.asarray(y_score),
        n_metastatic_predictions=0,
    )


def print_tcga_metrics(metrics: Dict, title: str) -> None:
    print(f'\n{title}')
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1:  {metrics['macro_f1']:.4f}")
    print('\nPer-class F1:')
    for name in metrics['class_names']:
        print(f"  {name}: {metrics['per_class_f1'][name]:.4f}")
    print('\nPer-class AUROC (one-vs-rest, 2-class):')
    for name in metrics['class_names']:
        print(f"  {name}: {metrics['per_class_auc'][name]:.4f}")
    print(
        f"\nMetastatic-class predictions (collapsed to primary): "
        f"{metrics['n_metastatic_predictions']}"
    )
    cm = metrics['confusion_matrix']
    header = '          ' + '  '.join(f'{n:>10}' for n in metrics['class_names'])
    print('\nConfusion matrix (rows=true, cols=pred):')
    print(header)
    for i, name in enumerate(metrics['class_names']):
        row = '  '.join(f'{v:10d}' for v in cm[i])
        print(f'{name:>10}  {row}')


def metrics_to_row(result: ApproachResult) -> dict:
    m = result.metrics
    cm = m['confusion_matrix']
    return {
        'approach': result.approach,
        'accuracy': m['accuracy'],
        'macro_f1': m['macro_f1'],
        'f1_normal': m['per_class_f1']['normal'],
        'f1_primary': m['per_class_f1']['primary'],
        'auc_normal': m['per_class_auc']['normal'],
        'auc_primary': m['per_class_auc']['primary'],
        'n_tcga_normal': result.n_tcga_normal,
        'n_tcga_tumor': result.n_tcga_tumor,
        'n_tcga_total': result.n_tcga_normal + result.n_tcga_tumor,
        'n_overlap_genes': result.n_overlap_genes,
        'n_metastatic_predictions': m['n_metastatic_predictions'],
        'cm_normal_normal': int(cm[0, 0]),
        'cm_normal_primary': int(cm[0, 1]),
        'cm_primary_normal': int(cm[1, 0]),
        'cm_primary_primary': int(cm[1, 1]),
        'notes': result.notes,
    }


def save_confusion_figure(results: list[ApproachResult], output_path: Path) -> None:
    n = len(results)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for ax, result in zip(axes_flat, results):
        cm = result.metrics['confusion_matrix']
        class_names = result.metrics['class_names']
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set(
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            ylabel='True label',
            xlabel='Predicted label',
            title=f"{result.approach}\n(acc={result.metrics['accuracy']:.3f}, "
            f"macro F1={result.metrics['macro_f1']:.3f})",
        )
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], 'd'),
                    ha='center',
                    va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                )

    for ax in axes_flat[len(results):]:
        ax.axis('off')

    fig.suptitle(
        'TCGA-BRCA classifier validation (normal vs basal primary; 2-class)',
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def print_comparison_summary(results: list[ApproachResult]) -> None:
    print('\n' + '=' * 72)
    print('TCGA CLASSIFIER VALIDATION SUMMARY')
    print('=' * 72)
    print(
        'External cohort: TCGA-BRCA solid tissue normal (sample type 11) vs '
        'PAM50 basal-like primary tumors.'
    )
    print(
        'Evaluation is 2-class (normal vs primary). GSE45498 models are 3-class; '
        'metastatic predictions are collapsed to primary for metrics.'
    )
    print(
        'Batch correction: reference-batch ComBat (GSE45498 = reference) on CPM-'
        'normalized log2 mRNA, then z-score using GSE45498 train stats.'
    )
    print('-' * 72)

    for result in results:
        m = result.metrics
        print(f"\n{result.approach}")
        print(f"  TCGA samples:     {result.n_tcga_normal} normal + {result.n_tcga_tumor} tumor")
        print(f"  Overlap genes:    {result.n_overlap_genes}")
        print(f"  Accuracy:         {m['accuracy']:.4f}")
        print(f"  Macro F1:         {m['macro_f1']:.4f}")
        print(f"  F1 normal:        {m['per_class_f1']['normal']:.4f}")
        print(f"  F1 primary:       {m['per_class_f1']['primary']:.4f}")
        print(f"  AUROC normal:     {m['per_class_auc']['normal']:.4f}")
        print(f"  AUROC primary:    {m['per_class_auc']['primary']:.4f}")
        print(f"  Notes:            {result.notes}")

    best = max(results, key=lambda r: r.metrics['macro_f1'])
    print(f"\nBest macro F1 on TCGA: {best.approach} ({best.metrics['macro_f1']:.4f})")
    print('=' * 72)


def select_top_mrna_importance_genes(
    model: XGBClassifier,
    tcga_available_genes: Sequence[str],
    top_n: int = TOP_MRNA_N,
) -> list[str]:
    """Top mRNA features by XGBoost feature_importances_, restricted to TCGA overlap."""
    panel = load_gene_panel()
    importances = pd.Series(model.feature_importances_, index=model.feature_names_in_)
    mrna_imp = importances[importances.index.isin(panel)]
    mrna_imp = mrna_imp[mrna_imp.index.isin(tcga_available_genes)]
    return mrna_imp.sort_values(ascending=False).head(top_n).index.tolist()


def load_top_mrna_training_data(genes: Sequence[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Processed GSE45498 train matrix for a selected mRNA gene subset."""
    x_train_mrna = pd.read_csv(PROCESSED_DIR / 'train_mrna.csv', index_col=0)
    train_labels = pd.read_csv(PROCESSED_DIR / 'train_labels.csv', index_col='sample_id')[
        'tissue_group'
    ]
    train_labels = train_labels.loc[x_train_mrna.index]
    genes = [g for g in genes if g in x_train_mrna.columns]
    X_train = x_train_mrna.loc[train_labels.index, genes]
    y_train = train_labels.map(LABEL_MAP).astype(int)
    return X_train, y_train


def load_de_signature_genes(
    tcga_available_genes: Sequence[str],
    top_n: int = TOP_MRNA_N,
) -> tuple[list[str], list[str]]:
    """Top up/down mRNA genes from Normal_vs_Primary DE, restricted to TCGA overlap."""
    if not DE_MRNA_PATH.exists():
        raise FileNotFoundError(f'Missing DE results: {DE_MRNA_PATH}')

    de = pd.read_csv(DE_MRNA_PATH)
    de_np = de[de['comparison'] == DE_COMPARISON].drop_duplicates('molecule_name')
    available = set(tcga_available_genes)

    up = (
        de_np[de_np['log2FC'] > 0]
        .sort_values('log2FC', ascending=False)['molecule_name']
        .tolist()
    )
    down = (
        de_np[de_np['log2FC'] < 0]
        .sort_values('log2FC')['molecule_name']
        .tolist()
    )
    up_genes = [g for g in up if g in available][:top_n]
    down_genes = [g for g in down if g in available][:top_n]
    return up_genes, down_genes


def compute_signature_scores(
    tcga_z: pd.DataFrame,
    up_genes: Sequence[str],
    down_genes: Sequence[str],
) -> pd.Series:
    """Risk score = mean(z upregulated) - mean(z downregulated); higher => tumor-like."""
    up_cols = [g for g in up_genes if g in tcga_z.columns]
    down_cols = [g for g in down_genes if g in tcga_z.columns]
    if not up_cols and not down_cols:
        raise ValueError('No signature genes available in TCGA feature matrix.')

    up_mean = tcga_z[up_cols].mean(axis=1) if up_cols else 0.0
    down_mean = tcga_z[down_cols].mean(axis=1) if down_cols else 0.0
    return up_mean - down_mean


def run_approach_a(
    model: XGBClassifier,
    gse_log2: pd.DataFrame,
    tcga_log2: pd.DataFrame,
    tcga_labels: pd.Series,
    X_train_full: pd.DataFrame,
    train_ids: list[str],
) -> ApproachResult:
    mrna_features = [c for c in X_train_full.columns if c in load_gene_panel()]
    tcga_mrna_z, overlap = prepare_tcga_mrna_features(
        gse_log2, tcga_log2, train_ids, mrna_features
    )
    X_tcga = build_full_model_tcga_matrix(model, tcga_mrna_z, overlap)

    y_pred = model.predict(X_tcga)
    y_prob = model.predict_proba(X_tcga)
    metrics = compute_tcga_binary_metrics(tcga_labels.values, y_pred, y_prob)
    print_tcga_metrics(
        metrics,
        'Approach A: pretrained XGBoost (full features; TCGA miRNA imputed at 0)',
    )

    n_normal = int((tcga_labels == 0).sum())
    n_tumor = int((tcga_labels == 1).sum())
    return ApproachResult(
        approach='pretrained_full_model',
        metrics=metrics,
        n_tcga_normal=n_normal,
        n_tcga_tumor=n_tumor,
        n_overlap_genes=len(overlap),
        notes=(
            'Full 146-feature model; TCGA mRNA via reference-batch ComBat+z-score; '
            'missing miRNA and mRNA imputed at train-mean (z=0). '
            'TCGA normals and basal tumors both from GDC STAR counts, uniformly '
            'CPM-normalized (cpm_log2) before harmonization to avoid a source/'
            'normalization confound.'
        ),
    )


def run_approach_b(
    gse_log2: pd.DataFrame,
    tcga_log2: pd.DataFrame,
    tcga_labels: pd.Series,
    train_ids: list[str],
    overlap_genes: list[str],
    best_params: dict,
) -> ApproachResult:
    X_train_overlap, y_train, _, _ = load_overlap_mrna_training_data()
    model, _ = train_xgboost_model(
        X_train_overlap, y_train, run_grid_search=False, best_params=best_params
    )

    tcga_mrna_z, overlap = prepare_tcga_mrna_features(
        gse_log2, tcga_log2, train_ids, overlap_genes
    )
    X_tcga = tcga_mrna_z.reindex(columns=model.feature_names_in_, fill_value=0.0)

    y_pred = model.predict(X_tcga)
    y_prob = model.predict_proba(X_tcga)
    metrics = compute_tcga_binary_metrics(tcga_labels.values, y_pred, y_prob)
    print_tcga_metrics(
        metrics,
        'Approach B: XGBoost retrained on overlapping mRNA genes only',
    )

    n_normal = int((tcga_labels == 0).sum())
    n_tumor = int((tcga_labels == 1).sum())
    return ApproachResult(
        approach='overlap_gene_retrain',
        metrics=metrics,
        n_tcga_normal=n_normal,
        n_tcga_tumor=n_tumor,
        n_overlap_genes=len(overlap),
        notes=(
            f'Retrained on {len(overlap_genes)} GSE45498 mRNA genes overlapping TCGA panel; '
            'same hyperparameters as Approach A; no miRNA features'
        ),
    )


def run_approach_c(
    model: XGBClassifier,
    gse_log2: pd.DataFrame,
    tcga_log2: pd.DataFrame,
    tcga_labels: pd.Series,
    train_ids: list[str],
    overlap_genes: list[str],
    best_params: dict,
) -> ApproachResult:
    tcga_mrna_z, overlap = prepare_tcga_mrna_features(
        gse_log2, tcga_log2, train_ids, overlap_genes
    )
    top_genes = select_top_mrna_importance_genes(model, overlap, top_n=TOP_MRNA_N)
    if not top_genes:
        raise ValueError('No top-importance mRNA genes overlap with TCGA after ComBat.')

    print(f'Approach C top genes ({len(top_genes)}): {", ".join(top_genes)}')

    X_train_top, y_train = load_top_mrna_training_data(top_genes)
    top_model, _ = train_xgboost_model(
        X_train_top, y_train, run_grid_search=False, best_params=best_params
    )

    X_tcga = tcga_mrna_z.reindex(columns=top_model.feature_names_in_, fill_value=0.0)
    y_pred = top_model.predict(X_tcga)
    y_prob = top_model.predict_proba(X_tcga)
    metrics = compute_tcga_binary_metrics(tcga_labels.values, y_pred, y_prob)
    print_tcga_metrics(
        metrics,
        'Approach C: XGBoost retrained on top mRNA genes by feature importance',
    )

    n_normal = int((tcga_labels == 0).sum())
    n_tumor = int((tcga_labels == 1).sum())
    return ApproachResult(
        approach='top_feature_retrain',
        metrics=metrics,
        n_tcga_normal=n_normal,
        n_tcga_tumor=n_tumor,
        n_overlap_genes=len(top_genes),
        notes=(
            f'Top {len(top_genes)} mRNA genes by feature_importances_ overlapping TCGA: '
            f'{", ".join(top_genes)}; same hyperparameters as Approach A'
        ),
    )


def run_approach_d(
    gse_log2: pd.DataFrame,
    tcga_log2: pd.DataFrame,
    tcga_labels: pd.Series,
    train_ids: list[str],
    overlap_genes: list[str],
) -> ApproachResult:
    tcga_mrna_z, overlap = prepare_tcga_mrna_features(
        gse_log2, tcga_log2, train_ids, overlap_genes
    )
    up_genes, down_genes = load_de_signature_genes(overlap, top_n=TOP_MRNA_N)
    if not up_genes and not down_genes:
        raise ValueError('No DE signature genes overlap with TCGA after ComBat.')

    print(f'Approach D up genes ({len(up_genes)}): {", ".join(up_genes) or "(none)"}')
    print(f'Approach D down genes ({len(down_genes)}): {", ".join(down_genes) or "(none)"}')

    scores = compute_signature_scores(tcga_mrna_z, up_genes, down_genes)
    y_pred = (scores > 0).astype(int).values
    metrics = compute_tcga_direct_binary_metrics(tcga_labels.values, y_pred, scores.values)
    print_tcga_metrics(
        metrics,
        'Approach D: DE gene signature score (threshold score > 0 => tumor)',
    )

    n_normal = int((tcga_labels == 0).sum())
    n_tumor = int((tcga_labels == 1).sum())
    return ApproachResult(
        approach='gene_signature_score',
        metrics=metrics,
        n_tcga_normal=n_normal,
        n_tcga_tumor=n_tumor,
        n_overlap_genes=len(set(up_genes) | set(down_genes)),
        notes=(
            f'DE {DE_COMPARISON}: mean z-score of {len(up_genes)} up genes minus '
            f'{len(down_genes)} down genes; classify tumor if score > 0'
        ),
    )


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    gse_log2 = load_gse_mrna_log2()
    tcga_log2, tcga_labels = load_tcga_validation_cohort()

    X_train_full, y_train, train_ids = load_gse45498_training_data()
    print(f'GSE45498 training matrix: {X_train_full.shape[0]} samples x {X_train_full.shape[1]} features')

    _, _, _, overlap_genes = load_overlap_mrna_training_data()
    print(f'mRNA genes overlapping TCGA panel: {len(overlap_genes)}')

    full_model, best_params = load_or_train_full_model(X_train_full, y_train)

    result_a = run_approach_a(
        full_model, gse_log2, tcga_log2, tcga_labels, X_train_full, train_ids
    )
    result_b = run_approach_b(
        gse_log2, tcga_log2, tcga_labels, train_ids, overlap_genes, best_params
    )
    result_c = run_approach_c(
        full_model, gse_log2, tcga_log2, tcga_labels, train_ids, overlap_genes, best_params
    )
    result_d = run_approach_d(
        gse_log2, tcga_log2, tcga_labels, train_ids, overlap_genes
    )

    results = [result_a, result_b, result_c, result_d]
    print_comparison_summary(results)

    summary_df = pd.DataFrame([metrics_to_row(r) for r in results])
    csv_path = TABLES_DIR / 'tcga_classifier_validation.csv'
    summary_df.to_csv(csv_path, index=False)

    fig_path = FIGURES_DIR / 'tcga_classifier_confusion.png'
    save_confusion_figure(results, fig_path)

    print('\nSaved outputs:')
    print(f'  {csv_path}')
    print(f'  {fig_path}')


if __name__ == '__main__':
    main()
