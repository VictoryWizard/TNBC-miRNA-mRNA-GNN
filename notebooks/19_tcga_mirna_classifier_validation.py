#!/usr/bin/env python3
"""TCGA-BRCA miRNA external classifier validation (mirrors notebook 10 for mRNA).

IMPORTANT — separate GDC download required before first run:
  TCGA miRNA-seq is NOT bundled with local mRNA/cBioPortal files. This script
  queries GDC for TCGA-BRCA ``miRNA Expression Quantification`` files, downloads
  read counts, caches them under ``data/``, then aligns to GSE45498 DE miRNA
  features for the same 2-class external check (solid tissue normal vs basal primary).

Cohort: PAM50 basal-like primary tumors (sample IDs from existing basal mRNA matrix
when available) + GDC solid tissue normal (sample type 11). Metastatic predictions
from the 3-class GSE45498 model are collapsed to primary for reporting.

Leakage controls:
  * Batch correction uses *reference-batch* ComBat with GSE45498 (training) fixed
    as the reference batch (``REFERENCE_BATCH = 0``); TCGA is projected onto the
    GSE distribution rather than the two cohorts being centered jointly. See
    ``run_combat``.
  * miRNA-overlap / finiteness selection is derived from the GSE training matrix
    only (``drop_features_with_missing``); TCGA is then subset to those features so
    no feature selection touches the test cohort.
  * Both TCGA normal and tumor miRNA matrices come from the same GDC miRNA-Seq
    source and identical log2 processing, so they are already on a comparable scale.
"""

from __future__ import annotations

import gzip
import io
import json
import re
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set

_NOTEBOOK_DIR = Path(__file__).resolve().parent
if str(_NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOK_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from combat.pycombat import pycombat
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from xgboost import XGBClassifier

from cv_utils import load_patient_groups

PROCESSED_DIR = Path('data/processed')
DATA_DIR = Path('data')
RAW_DATA_DIR = DATA_DIR / 'raw'
RESULTS_DIR = Path('results')
TABLES_DIR = RESULTS_DIR / 'tables'
FIGURES_DIR = RESULTS_DIR / 'figures'
MODELS_DIR = Path('models')

TCGA_BASEL_MRNA_PATH = DATA_DIR / 'tcga_brca_basal_mrna_expression.csv'
TCGA_NORMAL_MIRNA_PATH = DATA_DIR / 'tcga_brca_normal_mirna_expression.csv'
TCGA_TUMOR_MIRNA_PATH = DATA_DIR / 'tcga_brca_basal_mirna_expression.csv'
MIRNA_PANEL_PATH = DATA_DIR / 'tcga_mirna_panel.txt'
MODEL_PATH = MODELS_DIR / 'xgboost_model.pkl'
MIRNA_RAW_PATH = RAW_DATA_DIR / 'GSE45498_raw_data.txt.gz'

LABEL_MAP = {'normal': 0, 'primary': 1, 'metastatic': 2}
TCGA_BINARY_NAMES = ('normal', 'primary')
TOP_MIRNA_N = 10

# GSE45498 (training) cohort is batch 0 and is used as the ComBat reference batch.
REFERENCE_BATCH = 0

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
    n_overlap_mirnas: int
    notes: str


def normalize_mirna_name(name: str) -> str:
    """Map GDC-style ``hsa-mir-21`` to NanoString ``hsa-miR-21``."""
    s = str(name).strip()
    m = re.match(r'^(hsa-)(mir|miR|let)-(.+)$', s, flags=re.IGNORECASE)
    if not m:
        return s
    prefix, family, rest = m.group(1), m.group(2).lower(), m.group(3)
    if family == 'let':
        return f'{prefix}let-{rest}'
    return f'{prefix}miR-{rest}'


def mirna_base_key(name: str) -> str:
    """Canonical precursor-level key for matching NanoString panel names to GDC ids.

    Arms (-3p/-5p) are folded into the precursor, so hsa-miR-490-3p, hsa-mir-490,
    and the two genomic copies hsa-let-7f-1/-2 all collapse to one base key. This
    intentionally ignores 3p/5p arm identity (see Methods/limitations).
    """
    s = str(name).strip()
    s = re.sub(r'\s*\(.*$', '', s)                  # drop "(+++ See note...)" annotations
    s = s.strip().lower().replace('hsa-', '')
    s = re.sub(r'-(3p|5p)(\.\d+)?$', '', s)         # fold arm into precursor
    s = re.sub(r'(?<=\d)([a-z]*)-\d+$', r'\1', s)   # drop precursor copy suffix (-1/-2)
    return s


def build_panel_basekeys(panel: Sequence[str]) -> Dict[str, List[str]]:
    """base_key -> panel columns containing it; compound probes (A+B) split on +/ /."""
    mapping: Dict[str, List[str]] = {}
    for col in panel:
        clean = re.sub(r'\s*\(.*$', '', col)        # drop annotation before splitting
        for part in re.split(r'[+/]', clean):
            key = mirna_base_key(part)
            if not key:
                continue
            mapping.setdefault(key, [])
            if col not in mapping[key]:
                mapping[key].append(col)
    return mapping


def load_gse_mirna_panel() -> List[str]:
    train_mirna = pd.read_csv(PROCESSED_DIR / 'train_mirna.csv', index_col=0, nrows=0)
    return sorted(train_mirna.columns.tolist())


def load_gse_training_mirna_matrix() -> tuple[pd.DataFrame, pd.Series, List[str]]:
    x_train_mirna = pd.read_csv(PROCESSED_DIR / 'train_mirna.csv', index_col=0)
    x_train_mrna = pd.read_csv(PROCESSED_DIR / 'train_mrna.csv', index_col=0)
    labels = pd.read_csv(PROCESSED_DIR / 'train_labels.csv', index_col='sample_id')['tissue_group']
    labels = labels.loc[x_train_mirna.index]
    x_train = pd.concat([x_train_mirna, x_train_mrna], axis=1)
    y_train = labels.map(LABEL_MAP).astype(int)
    return x_train, y_train, x_train_mirna.index.tolist()


def load_gse_log2_mirna(train_ids: Sequence[str], mirnas: Sequence[str]) -> pd.DataFrame:
    """Log2 GSE NanoString miRNA counts (training samples x panel miRNAs) for ComBat ref.

    The raw GSE file stores miRNAs as ROWS (the 'Name' column) and samples as
    COLUMNS, so panel-miRNA rows and training-sample columns are selected, then
    transposed to a samples x miRNAs matrix.
    """
    if not MIRNA_RAW_PATH.exists():
        raise FileNotFoundError(
            f'Missing {MIRNA_RAW_PATH} for GSE miRNA batch correction reference.'
        )
    raw = pd.read_csv(MIRNA_RAW_PATH, sep='\t', compression='gzip')
    raw.columns = [c.strip() for c in raw.columns]
    name_col = raw.columns[1]                       # 'Name' column holds miRNA ids
    raw[name_col] = raw[name_col].astype(str).str.strip()
    train_set = set(train_ids)
    panel_set = set(mirnas)
    sample_cols = [c for c in raw.columns[3:] if c in train_set]
    keep_rows = raw[raw[name_col].isin(panel_set)]
    if keep_rows.empty or not sample_cols:
        raise ValueError(
            'No overlapping miRNA rows or training samples between raw GSE file and train panel.'
        )
    mat = keep_rows.set_index(name_col)[sample_cols].T
    mat.index.name = 'sample_id'
    return np.log2(mat.astype(float) + 1)


def fetch_gdc_mirna_file_index(sample_types: Sequence[str]) -> pd.DataFrame:
    """List GDC miRNA quantification files for TCGA-BRCA sample types."""
    filters = {
        'op': 'and',
        'content': [
            {
                'op': 'in',
                'content': {'field': 'cases.project.project_id', 'value': ['TCGA-BRCA']},
            },
            {
                'op': 'in',
                'content': {'field': 'cases.samples.sample_type', 'value': list(sample_types)},
            },
            {
                'op': 'in',
                'content': {
                    'field': 'files.data_type',
                    'value': ['miRNA Expression Quantification'],
                },
            },
            {
                'op': 'in',
                'content': {'field': 'files.experimental_strategy', 'value': ['miRNA-Seq']},
            },
        ],
    }
    params = urllib.parse.urlencode(
        {
            'filters': json.dumps(filters),
            'fields': 'file_id,cases.samples.submitter_id',
            'size': '2000',
            'format': 'json',
        }
    )
    with urllib.request.urlopen(f'https://api.gdc.cancer.gov/files?{params}', timeout=120) as response:
        payload = json.loads(response.read())

    rows = []
    for hit in payload['data']['hits']:
        submitter = hit['cases'][0]['samples'][0]['submitter_id']
        sample_id = submitter[:15] if submitter.startswith('TCGA-') else submitter
        rows.append({'file_id': hit['file_id'], 'sample_id': sample_id})

    frame = pd.DataFrame(rows).drop_duplicates('sample_id', keep='first')
    if frame.empty:
        raise RuntimeError(f'No TCGA-BRCA miRNA files for sample types {sample_types}.')
    return frame


def _parse_gdc_mirna_quant(text: str, wanted_keys: Set[str]) -> Dict[str, float]:
    """Parse a GDC miRNA quantification TSV into {base_key: read_count}.

    Values are summed across precursors that collapse to the same base key
    (e.g. hsa-mir-16-1 + hsa-mir-16-2 -> mir-16).
    """
    values: Dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith('#'):
            continue
        parts = line.split('\t')
        if len(parts) < 2:
            continue
        raw_id = parts[0]
        if raw_id.lower() in ('mirna_id', 'id', 'composite_element_ref'):
            continue
        try:
            val = float(parts[1])
        except ValueError:
            if len(parts) >= 3:
                try:
                    val = float(parts[2])
                except ValueError:
                    continue
            else:
                continue
        key = mirna_base_key(raw_id)
        if key in wanted_keys:
            values[key] = values.get(key, 0.0) + val
    return values


def download_gdc_mirna_expression(
    sample_types: Sequence[str],
    mirna_panel: Sequence[str],
    batch_size: int = 20,
) -> pd.DataFrame:
    """Download TCGA-BRCA miRNA expression from GDC (requires network)."""
    index = fetch_gdc_mirna_file_index(sample_types)
    basekey_map = build_panel_basekeys(mirna_panel)
    mirna_keys = set(basekey_map)
    records: Dict[str, Dict[str, float]] = {}

    file_ids = index['file_id'].tolist()
    sample_ids = index['sample_id'].tolist()
    print(
        f'Downloading {len(file_ids)} TCGA miRNA files from GDC '
        f'(sample types: {", ".join(sample_types)})...'
    )

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
                raw = extracted.read()
                try:
                    text = raw.decode('utf-8')
                except UnicodeDecodeError:
                    text = gzip.decompress(raw).decode('utf-8')
                records[sample_id] = _parse_gdc_mirna_quant(text, mirna_keys)

        print(f'  batch {start // batch_size + 1}: {min(start + batch_size, len(file_ids))}'
              f'/{len(file_ids)} files', flush=True)

    if not records:
        raise RuntimeError('GDC miRNA download returned no usable expression values.')

    # Reassemble panel columns from base-key reads; compound probes (A+B) sum
    # their components, plain/arm names take their single base key.
    panel_records: Dict[str, Dict[str, float]] = {}
    for sample_id, kv in records.items():
        row: Dict[str, float] = {}
        for col in mirna_panel:
            clean = re.sub(r'\s*\(.*$', '', col)
            comp_vals = [kv[mirna_base_key(p)] for p in re.split(r'[+/]', clean)
                         if mirna_base_key(p) in kv]
            if comp_vals:
                row[col] = float(sum(comp_vals))
        panel_records[sample_id] = row

    matrix = pd.DataFrame.from_dict(panel_records, orient='index')
    matrix = matrix.reindex(columns=[m for m in mirna_panel if m in matrix.columns])
    matrix.index.name = 'sample_id'
    recovered = matrix.shape[1]
    print(f'  Recovered {recovered}/{len(mirna_panel)} panel miRNAs from GDC.')
    return matrix.sort_index()


def ensure_tcga_mirna_matrices(mirna_panel: Sequence[str]) -> None:
    """Cache normal and basal-primary miRNA matrices (GDC download on first run)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not TCGA_NORMAL_MIRNA_PATH.exists():
        print('\n*** GDC DOWNLOAD: TCGA-BRCA solid tissue normal miRNA-seq ***')
        normal = download_gdc_mirna_expression(['Solid Tissue Normal'], mirna_panel)
        normal.to_csv(TCGA_NORMAL_MIRNA_PATH)
        print(f'Saved {TCGA_NORMAL_MIRNA_PATH} ({normal.shape})')

    if not TCGA_TUMOR_MIRNA_PATH.exists():
        print('\n*** GDC DOWNLOAD: TCGA-BRCA primary tumor miRNA-seq ***')
        tumor_all = download_gdc_mirna_expression(['Primary Tumor'], mirna_panel)
        if TCGA_BASEL_MRNA_PATH.exists():
            basal_ids = pd.read_csv(TCGA_BASEL_MRNA_PATH, index_col=0).index.astype(str)
            overlap_ids = tumor_all.index.intersection(basal_ids)
            tumor = tumor_all.loc[overlap_ids]
            print(
                f'  Restricted to {len(tumor)} basal-primary samples overlapping '
                f'mRNA basal matrix ({len(basal_ids)} IDs).'
            )
        else:
            tumor = tumor_all
            print(
                '  WARNING: basal mRNA matrix missing; using all TCGA-BRCA primary tumors. '
                f'({tumor.shape[0]} samples)'
            )
        tumor.to_csv(TCGA_TUMOR_MIRNA_PATH)
        print(f'Saved {TCGA_TUMOR_MIRNA_PATH} ({tumor.shape})')


def load_tcga_mirna_cohort() -> tuple[pd.DataFrame, pd.Series]:
    ensure_tcga_mirna_matrices(load_gse_mirna_panel())
    normal = np.log2(pd.read_csv(TCGA_NORMAL_MIRNA_PATH, index_col=0) + 1)
    tumor = np.log2(pd.read_csv(TCGA_TUMOR_MIRNA_PATH, index_col=0) + 1)
    normal.index = normal.index.astype(str)
    tumor.index = tumor.index.astype(str)
    combined = pd.concat([normal, tumor], axis=0)
    labels = pd.Series(
        [0] * len(normal) + [1] * len(tumor),
        index=combined.index,
        name='tcga_label',
    )
    print(
        f'TCGA miRNA cohort: {len(normal)} normal + {len(tumor)} basal primary '
        f'({len(combined)} total)'
    )
    return combined, labels


def drop_features_with_missing(
    gse_df: pd.DataFrame,
    tcga_df: pd.DataFrame,
    features: List[str],
) -> List[str]:
    # Bug fix (feature-selection-on-test): finiteness decided on GSE (train) only;
    # TCGA contributes column membership for alignment, not the selection.
    features = [f for f in features if f in gse_df.columns]
    if not features:
        return []
    gse_sub = gse_df[features]
    ok = gse_sub.columns[gse_sub.notna().all() & np.isfinite(gse_sub).all()]
    return [f for f in sorted(ok) if f in tcga_df.columns]


def run_combat(gse_df: pd.DataFrame, tcga_df: pd.DataFrame, overlap: List[str]) -> pd.DataFrame:
    """Reference-batch ComBat: GSE (train) is the fixed reference; TCGA is projected onto it.

    ``ref_batch=REFERENCE_BATCH`` freezes the reference (training) batch so corrected
    TCGA values are projected onto the GSE distribution instead of both cohorts being
    centered jointly (which leaks the test cohort). Falls back to standard ComBat with
    a clear warning if the installed pyComBat lacks ``ref_batch`` support.
    """
    combined = pd.concat([gse_df[overlap], tcga_df[overlap]], axis=0)
    batch = np.array([REFERENCE_BATCH] * len(gse_df) + [1] * len(tcga_df))
    data_t = combined.T
    data_t.columns = combined.index

    # Preferred: inmoose.pycombat_norm has a correct reference-batch implementation.
    # combat.pycombat's ref_batch path is buggy (IndexError on the reference batch).
    try:
        from inmoose.pycombat import pycombat_norm
        out = pycombat_norm(data_t, batch, ref_batch=REFERENCE_BATCH)
        if not isinstance(out, pd.DataFrame):
            out = pd.DataFrame(out, index=data_t.index, columns=data_t.columns)
        return out.T
    except Exception as exc_inmoose:
        print(f'inmoose reference-batch ComBat unavailable ({exc_inmoose}); trying combat.pycombat...')

    try:
        corrected_t = pycombat(data=data_t, batch=batch, ref_batch=REFERENCE_BATCH)
    except (TypeError, IndexError):
        print(
            'WARNING: installed pyComBat lacks a working ref_batch; falling back to '
            'joint ComBat. This reintroduces transductive leakage (corrected TCGA '
            'values depend on the test cohort). Install inmoose for a frozen '
            'reference-batch fit.'
        )
        corrected_t = pycombat(data=data_t, batch=batch)
    return corrected_t.T


def zscore_from_gse_train(
    corrected: pd.DataFrame,
    train_ids: Sequence[str],
    features: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_expr = corrected.loc[list(train_ids), features]
    mean = train_expr.mean()
    std = train_expr.std().replace(0, 1.0)
    return (train_expr - mean) / std, pd.DataFrame({'mean': mean, 'std': std})


def apply_zscore(expr: pd.DataFrame, features: Sequence[str], params: pd.DataFrame) -> pd.DataFrame:
    mean = params.loc[list(features), 'mean']
    std = params.loc[list(features), 'std']
    return (expr[features] - mean) / std


def prepare_tcga_mirna_features(
    gse_log2: pd.DataFrame,
    tcga_log2: pd.DataFrame,
    train_ids: Sequence[str],
    mirnas: Sequence[str],
) -> tuple[pd.DataFrame, List[str]]:
    overlap = drop_features_with_missing(gse_log2, tcga_log2, list(mirnas))
    if len(overlap) < 2:
        raise ValueError('Fewer than 2 overlapping miRNAs with finite values in both cohorts.')
    corrected = run_combat(gse_log2, tcga_log2, overlap)
    n_gse = len(gse_log2)
    tcga_corrected = corrected.iloc[n_gse:]
    _, z_params = zscore_from_gse_train(corrected.iloc[:n_gse], train_ids, overlap)
    tcga_z = apply_zscore(tcga_corrected, overlap, z_params)
    return tcga_z, overlap


def collapse_to_binary_predictions(y_pred: np.ndarray) -> np.ndarray:
    return np.where(y_pred == 0, 0, 1)


def binary_probabilities(y_prob_3class: np.ndarray) -> np.ndarray:
    return y_prob_3class[:, 1] + y_prob_3class[:, 2]


def compute_tcga_binary_metrics(
    y_true: np.ndarray,
    y_pred_3class: np.ndarray,
    y_prob_3class: np.ndarray,
) -> Dict:
    y_pred = collapse_to_binary_predictions(y_pred_3class)
    y_prob = binary_probabilities(y_prob_3class)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'per_class_f1': {
            'normal': float(f1_score(y_true, y_pred, labels=[0], average='macro', zero_division=0)),
            'primary': float(f1_score(y_true, y_pred, labels=[1], average='macro', zero_division=0)),
        },
        'per_class_auc': {
            'normal': float(roc_auc_score((y_true == 0).astype(int), 1.0 - y_prob)),
            'primary': float(roc_auc_score((y_true == 1).astype(int), y_prob)),
        },
        'confusion_matrix': cm,
        'class_names': list(TCGA_BINARY_NAMES),
        'n_metastatic_predictions': int((y_pred_3class == 2).sum()),
    }


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    run_grid_search: bool = False,
    best_params: Dict | None = None,
) -> tuple[XGBClassifier, Dict]:
    base = XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    groups = load_patient_groups(X_train.index).values
    if run_grid_search or best_params is None:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(
            base,
            param_grid=XGB_PARAM_GRID,
            scoring='f1_macro',
            cv=cv,
            n_jobs=-1,
            verbose=1,
        )
        grid.fit(X_train, y_train, groups=groups)
        best_params = dict(grid.best_params_)
        model = grid.best_estimator_
    else:
        model = base.set_params(**best_params)
        model.fit(X_train, y_train)
    return model, best_params


def build_full_model_tcga_matrix(
    model: XGBClassifier,
    tcga_mirna_z: pd.DataFrame,
    overlap_mirnas: Sequence[str],
) -> pd.DataFrame:
    X = pd.DataFrame(0.0, index=tcga_mirna_z.index, columns=model.feature_names_in_)
    for mirna in overlap_mirnas:
        if mirna in X.columns:
            X[mirna] = tcga_mirna_z[mirna].values
    return X


def select_top_mirna_importance(
    model: XGBClassifier,
    available: Sequence[str],
    top_n: int = TOP_MIRNA_N,
) -> List[str]:
    imp = pd.Series(model.feature_importances_, index=model.feature_names_in_)
    mirna_imp = imp[imp.index.map(lambda x: str(x).lower().startswith('hsa-'))]
    mirna_imp = mirna_imp[mirna_imp.index.isin(available)]
    return mirna_imp.sort_values(ascending=False).head(top_n).index.tolist()


def save_results(results: List[ApproachResult]) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in results:
        m = r.metrics
        cm = m['confusion_matrix']
        rows.append({
            'approach': r.approach,
            'accuracy': m['accuracy'],
            'macro_f1': m['macro_f1'],
            'f1_normal': m['per_class_f1']['normal'],
            'f1_primary': m['per_class_f1']['primary'],
            'auc_normal': m['per_class_auc']['normal'],
            'auc_primary': m['per_class_auc']['primary'],
            'n_tcga_normal': r.n_tcga_normal,
            'n_tcga_tumor': r.n_tcga_tumor,
            'n_overlap_mirnas': r.n_overlap_mirnas,
            'n_metastatic_predictions': m['n_metastatic_predictions'],
            'cm_normal_normal': int(cm[0, 0]),
            'cm_normal_primary': int(cm[0, 1]),
            'cm_primary_normal': int(cm[1, 0]),
            'cm_primary_primary': int(cm[1, 1]),
            'notes': r.notes,
        })
    out = TABLES_DIR / 'tcga_mirna_classifier_validation.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'\nSaved {out}')


def main() -> None:
    print('=' * 72)
    print('TCGA-BRCA miRNA CLASSIFIER VALIDATION')
    print('=' * 72)
    print(
        'Requires a separate GDC download for miRNA-seq quantification files.\n'
        'Cached outputs: data/tcga_brca_normal_mirna_expression.csv, '
        'data/tcga_brca_basal_mirna_expression.csv'
    )

    X_train_full, y_train, train_ids = load_gse_training_mirna_matrix()
    mirna_panel = load_gse_mirna_panel()
    gse_mirna_log2 = load_gse_log2_mirna(train_ids, mirna_panel)
    tcga_log2, tcga_labels = load_tcga_mirna_cohort()

    if MODEL_PATH.exists():
        model_full = pd.read_pickle(MODEL_PATH)
        full_params = {
            k: getattr(model_full, k)
            for k in ('max_depth', 'n_estimators', 'learning_rate')
            if hasattr(model_full, k)
        }
    else:
        print(f'Training full XGBoost model (no {MODEL_PATH})...')
        model_full, full_params = train_xgboost(X_train_full, y_train, run_grid_search=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(model_full, MODEL_PATH)

    results: List[ApproachResult] = []

    # A) Pretrained full model; TCGA mRNA features imputed at 0
    tcga_mirna_z, overlap_a = prepare_tcga_mirna_features(
        gse_mirna_log2, tcga_log2, train_ids, mirna_panel,
    )
    X_tcga_a = build_full_model_tcga_matrix(model_full, tcga_mirna_z, overlap_a)
    y_pred_a = model_full.predict(X_tcga_a)
    y_prob_a = model_full.predict_proba(X_tcga_a)
    metrics_a = compute_tcga_binary_metrics(tcga_labels.values, y_pred_a, y_prob_a)
    results.append(ApproachResult(
        approach='pretrained_full_model',
        metrics=metrics_a,
        n_tcga_normal=int((tcga_labels == 0).sum()),
        n_tcga_tumor=int((tcga_labels == 1).sum()),
        n_overlap_mirnas=len(overlap_a),
        notes=(
            'Full multi-modal model; TCGA miRNA via ComBat+log2+z-score; '
            'missing mRNA imputed at 0. GDC miRNA-seq download required.'
        ),
    ))

    # B) Retrain on overlapping miRNAs only
    overlap_b = [m for m in mirna_panel if m in overlap_a]
    X_tr_b = X_train_full[overlap_b]
    model_b, _ = train_xgboost(X_tr_b, y_train, best_params=full_params)
    X_tcga_b = tcga_mirna_z[overlap_b]
    y_pred_b = model_b.predict(X_tcga_b)
    y_prob_b = model_b.predict_proba(X_tcga_b)
    metrics_b = compute_tcga_binary_metrics(tcga_labels.values, y_pred_b, y_prob_b)
    results.append(ApproachResult(
        approach='overlap_mirna_retrain',
        metrics=metrics_b,
        n_tcga_normal=int((tcga_labels == 0).sum()),
        n_tcga_tumor=int((tcga_labels == 1).sum()),
        n_overlap_mirnas=len(overlap_b),
        notes='Retrained on TCGA-overlapping miRNAs only; same HPs as full model.',
    ))

    # C) Top-N miRNA by importance (TCGA-overlapping)
    top_mirnas = select_top_mirna_importance(model_full, overlap_b, TOP_MIRNA_N)
    if len(top_mirnas) >= 2:
        X_tr_c = X_train_full[top_mirnas]
        model_c, _ = train_xgboost(X_tr_c, y_train, best_params=full_params)
        X_tcga_c = tcga_mirna_z[top_mirnas]
        y_pred_c = model_c.predict(X_tcga_c)
        y_prob_c = model_c.predict_proba(X_tcga_c)
        metrics_c = compute_tcga_binary_metrics(tcga_labels.values, y_pred_c, y_prob_c)
        results.append(ApproachResult(
            approach=f'top{TOP_MIRNA_N}_mirna_retrain',
            metrics=metrics_c,
            n_tcga_normal=int((tcga_labels == 0).sum()),
            n_tcga_tumor=int((tcga_labels == 1).sum()),
            n_overlap_mirnas=len(top_mirnas),
            notes=f'Top {TOP_MIRNA_N} miRNAs by feature_importances_ (TCGA overlap).',
        ))

    for r in results:
        m = r.metrics
        print(f"\n{r.approach}: acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f} "
              f"(overlap miRNAs={r.n_overlap_mirnas})")
        print(f"  {r.notes}")

    save_results(results)
    MIRNA_PANEL_PATH.write_text('\n'.join(mirna_panel) + '\n', encoding='utf-8')
    print(f'Wrote miRNA panel list -> {MIRNA_PANEL_PATH}')


if __name__ == '__main__':
    main()
