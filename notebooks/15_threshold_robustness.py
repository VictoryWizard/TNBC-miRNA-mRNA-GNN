#!/usr/bin/env python3
"""Threshold robustness analysis for miRDB and correlation cutoffs (notebook 15).

Reconstructs miRNA-mRNA regulatory pairs across a threshold grid using training
data only and reports how sensitive pair counts are to threshold choice.

XGBoost classification performance is evaluated once on the fixed 146 DE miRNA/mRNA
expression features from the main pipeline. Regulatory-pair thresholds affect
network/GAT construction, not XGBoost inputs.
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

_NOTEBOOK_DIR = Path(__file__).resolve().parent
if str(_NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOK_DIR))

warnings.filterwarnings('ignore')

PROCESSED_DIR = Path('data/processed')
RAW_DATA_DIR = Path('data/raw')
TABLES_DIR = Path('results/tables')
FIGURES_DIR = Path('results/figures')
MODELS_DIR = Path('models')

MIRDB_THRESHOLDS = [50, 60, 70, 80, 90]
CORRELATION_CUTOFFS = [0.05, 0.1, 0.15, 0.2, 0.3]
DEFAULT_MIRDB = 70
DEFAULT_CORR = 0.1
MIN_MIRDB_FOR_PRECOMPUTE = min(MIRDB_THRESHOLDS)

CLASS_NAMES = ['normal', 'primary', 'metastatic']
CLASS_IDS = [0, 1, 2]
LABEL_MAP = {'normal': 0, 'primary': 1, 'metastatic': 2}

XGBOOST_NOTE = (
    'XGBoost uses fixed DE expression features; threshold sweep affects regulatory '
    'pair construction, not XGBoost input features.'
)
PERFORMANCE_NOTE = (
    'Identical across all thresholds: XGBoost is trained on fixed 146-feature '
    'expression matrix; miRDB/correlation cutoffs do not change XGBoost inputs.'
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pair_mod = _load_module('pair_construction', _NOTEBOOK_DIR / '02_pair_construction.py')
xgb_mod = _load_module('xgboost_baseline', _NOTEBOOK_DIR / '04_xgboost_baseline.py')
from evaluation_metrics import compute_classification_metrics  # noqa: E402


def load_pair_inputs():
    """Load DE tables, miRDB, and train expression (train only for correlations)."""
    de_mirnas = pd.read_csv(PROCESSED_DIR / 'de_mirnas.csv')
    de_mirnas['molecule_name'] = de_mirnas['molecule_name'].map(pair_mod.normalize_mirna_name)
    de_mrnas = pd.read_csv(PROCESSED_DIR / 'de_mrnas.csv')

    mrna_raw = pd.read_csv(
        RAW_DATA_DIR / 'GSE45498_mRNA_non-normalized_data.txt.gz',
        sep='\t',
        compression='gzip',
    )
    refseq_to_symbol = dict(zip(mrna_raw['Accession'].str.split('.').str[0], mrna_raw['Name']))

    mirdb = pd.read_csv(
        RAW_DATA_DIR / 'miRDB_v6.0_prediction_result.txt.gz',
        sep='\t',
        header=None,
        compression='gzip',
        names=['miRNA', 'target_refseq', 'score'],
    )
    mirdb = mirdb[mirdb['miRNA'].str.startswith('hsa-')].copy()
    mirdb['target_refseq'] = mirdb['target_refseq'].astype(str)
    mirdb['target_probe'] = mirdb['target_refseq'].str.split('.').str[0]
    mirdb['target_gene'] = mirdb['target_probe'].map(refseq_to_symbol)
    mirdb = mirdb.dropna(subset=['target_gene']).copy()

    train_mirna = pd.read_csv(PROCESSED_DIR / 'train_mirna.csv', index_col=0)
    train_mrna = pd.read_csv(PROCESSED_DIR / 'train_mrna.csv', index_col=0)
    return de_mirnas, de_mrnas, mirdb, train_mirna, train_mrna


def precompute_candidate_correlations(
    de_mirnas: pd.DataFrame,
    mirdb: pd.DataFrame,
    train_mirna: pd.DataFrame,
    train_mrna: pd.DataFrame,
) -> pd.DataFrame:
    """Compute Pearson r for all miRDB candidates above the minimum score threshold."""
    de_mirna_names = set(de_mirnas['molecule_name'].unique())
    measurable_mrna = set(train_mrna.columns)

    mirdb_candidates = mirdb[
        (mirdb['score'] > MIN_MIRDB_FOR_PRECOMPUTE)
        & (mirdb['target_gene'].isin(measurable_mrna))
        & (mirdb['miRNA'].isin(de_mirna_names))
    ].copy()

    mirna_cols = train_mirna.columns.tolist()
    mrna_cols = train_mrna.columns.tolist()
    mirna_indices = {c: i for i, c in enumerate(mirna_cols)}
    mrna_indices = {c: i for i, c in enumerate(mrna_cols)}

    mirna_arr = train_mirna.values.astype(float)
    mrna_arr = train_mrna.values.astype(float)
    mirna_std = np.std(mirna_arr, axis=0)
    mrna_std = np.std(mrna_arr, axis=0)

    mirna_comparisons = de_mirnas.groupby('molecule_name')['comparison'].apply(set).to_dict()
    mirna_fc = de_mirnas.set_index(['molecule_name', 'comparison'])['log2FC'].to_dict()

    batch_meta: List[Tuple] = []
    for _, row in mirdb_candidates.iterrows():
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

    if not batch_meta:
        return pd.DataFrame(columns=pair_mod.PAIR_COLUMNS)

    mi_idxs = np.array([m[4] for m in batch_meta], dtype=int)
    mr_idxs = np.array([m[5] for m in batch_meta], dtype=int)
    mirna_batch = mirna_arr[:, mi_idxs]
    mrna_batch = mrna_arr[:, mr_idxs]
    r_values = pair_mod._batched_pearsonr(mirna_batch, mrna_batch)

    rows = []
    for (mirna, mrna_probe, mrna, mirdb_score, _, _), r in zip(batch_meta, r_values):
        if not np.isfinite(r):
            continue
        for comp in mirna_comparisons.get(mirna, set()):
            rows.append({
                'miRNA': mirna,
                'mRNA_probe': mrna_probe,
                'mRNA_gene_symbol': mrna,
                'pearson_r': float(r),
                'miRDB_score': float(mirdb_score),
                'comparison': comp,
                'log2FC_miRNA': mirna_fc.get((mirna, comp), np.nan),
                'correlation_direction': 'positive' if r > 0 else 'negative',
            })

    candidates = pd.DataFrame(rows)
    if candidates.empty:
        return candidates
    return candidates.drop_duplicates(
        subset=['miRNA', 'mRNA_gene_symbol', 'comparison', 'mRNA_probe']
    )


def filter_pairs(
    candidates: pd.DataFrame,
    mirdb_threshold: float,
    correlation_cutoff: float,
) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=pair_mod.PAIR_COLUMNS)
    filtered = candidates[
        (candidates['miRDB_score'] > mirdb_threshold)
        & (candidates['pearson_r'].abs() > correlation_cutoff)
    ].copy()
    if filtered.empty:
        return pd.DataFrame(columns=pair_mod.PAIR_COLUMNS)
    return filtered[pair_mod.PAIR_COLUMNS].sort_values(['comparison', 'pearson_r'])


def count_pair_stats(pairs_df: pd.DataFrame) -> Tuple[int, int, int]:
    if pairs_df.empty:
        return 0, 0, 0
    return (
        len(pairs_df),
        pairs_df['miRNA'].nunique(),
        pairs_df['mRNA_gene_symbol'].nunique(),
    )


def macro_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true_bin = label_binarize(y_true, classes=CLASS_IDS)
    aucs = []
    for idx in range(len(CLASS_NAMES)):
        if y_true_bin[:, idx].sum() in (0, len(y_true_bin)):
            continue
        aucs.append(roc_auc_score(y_true_bin[:, idx], y_prob[:, idx]))
    return float(np.mean(aucs)) if aucs else float('nan')


def load_fixed_xgboost_metrics() -> Dict:
    """Load held-out XGBoost metrics once from saved predictions or the saved model."""
    pred_path = TABLES_DIR / 'xgboost_test_predictions.csv'
    if pred_path.exists():
        print(f'Loading fixed XGBoost test predictions from {pred_path}', flush=True)
        preds = pd.read_csv(pred_path)
        y_true = preds['true_label'].map(LABEL_MAP).astype(int).values
        y_pred = preds['pred_label'].map(LABEL_MAP).astype(int).values
        y_prob = preds[['prob_normal', 'prob_primary', 'prob_metastatic']].values
        source = str(pred_path)
    else:
        model_path = MODELS_DIR / 'xgboost_model.pkl'
        if not model_path.exists():
            raise FileNotFoundError(
                'Neither results/tables/xgboost_test_predictions.csv nor '
                'models/xgboost_model.pkl was found.'
            )
        print(f'Generating XGBoost test predictions from {model_path}', flush=True)
        model = joblib.load(model_path)
        _, X_test, _, y_test, _ = xgb_mod.load_data()
        y_true = np.asarray(y_test)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        source = str(model_path)

    metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        class_names=CLASS_NAMES,
        class_ids=CLASS_IDS,
    )
    metrics['macro_auroc'] = macro_auroc(y_true, y_prob)
    metrics['source'] = source
    return metrics


def plot_heatmap(
    pivot: pd.DataFrame,
    title: str,
    path: Path,
    fmt: str = '.3f',
    subtitle: str | None = None,
    cmap: str = 'viridis',
) -> None:
    data = pivot.values.astype(float)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data, aspect='auto', cmap=cmap)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(i) for i in pivot.index])
    ax.set_xlabel('Correlation cutoff (|r| > cutoff)')
    ax.set_ylabel('miRDB score threshold (score > threshold)')
    ax.set_title(title)
    if subtitle:
        ax.text(
            0.5, -0.14, subtitle,
            transform=ax.transAxes,
            ha='center',
            va='top',
            fontsize=9,
            wrap=True,
        )
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            text = '—' if np.isnan(val) else format(val, fmt)
            ax.text(j, i, text, ha='center', va='center', color='white', fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def assess_default_defensibility(
    default_pairs: int,
    pair_min: int,
    pair_max: int,
    pair_counts_df: pd.DataFrame,
) -> Tuple[bool, str]:
    """Assess whether default thresholds are a reasonable middle ground for pair counts."""
    if pair_max == pair_min:
        return True, 'All thresholds yield the same pair count.'

    percentile = (default_pairs - pair_min) / (pair_max - pair_min)
    mid_range = 0.25 <= percentile <= 0.75

    permissive = pair_counts_df.loc[pair_counts_df['n_pairs'].idxmax()]
    restrictive = pair_counts_df.loc[pair_counts_df['n_pairs'].idxmin()]

    notes = (
        f'Default yields {default_pairs} pairs ({percentile:.0%} of permissive-restrictive range). '
        f'Most permissive: miRDB>{int(permissive["mirdb_threshold"])}, '
        f'|r|>{permissive["correlation_cutoff"]} ({int(permissive["n_pairs"])} pairs). '
        f'Most restrictive with pairs: miRDB>{int(restrictive["mirdb_threshold"])}, '
        f'|r|>{restrictive["correlation_cutoff"]} ({int(restrictive["n_pairs"])} pairs).'
    )
    defensible = mid_range and default_pairs > 0
    return defensible, notes


def build_summary(
    results_df: pd.DataFrame,
    pair_counts_df: pd.DataFrame,
    xgb_metrics: Dict,
    default_defensible: bool,
    default_notes: str,
) -> pd.DataFrame:
    default_pairs = int(
        pair_counts_df[
            (pair_counts_df['mirdb_threshold'] == DEFAULT_MIRDB)
            & (pair_counts_df['correlation_cutoff'] == DEFAULT_CORR)
        ]['n_pairs'].iloc[0]
    )
    pair_min = int(pair_counts_df['n_pairs'].min())
    pair_max = int(pair_counts_df['n_pairs'].max())
    pair_sensitive = pair_max >= 2 * max(pair_min, 1)

    rows = [
        {
            'metric': 'analysis_scope',
            'value': 'regulatory_pair_construction',
            'notes': (
                'Thresholds affect the number of miRNA-mRNA regulatory pairs used for '
                'network/GAT construction.'
            ),
        },
        {
            'metric': 'xgboost_independence',
            'value': True,
            'notes': (
                'XGBoost classification performance is independent of these thresholds because '
                'XGBoost is trained on fixed DE miRNA/mRNA expression features (146 features), '
                'not on pair-derived features.'
            ),
        },
        {
            'metric': 'evaluation_focus',
            'value': 'pair_threshold_robustness',
            'notes': (
                'This notebook evaluates threshold robustness for regulatory-pair/network '
                'construction, not threshold-dependent XGBoost feature selection.'
            ),
        },
        {
            'metric': 'default_mirdb_threshold',
            'value': DEFAULT_MIRDB,
            'notes': 'Pipeline default in notebook 02 (score > 70)',
        },
        {
            'metric': 'default_correlation_cutoff',
            'value': DEFAULT_CORR,
            'notes': 'Pipeline default in notebook 02 (|r| > 0.1)',
        },
        {
            'metric': 'default_pair_count',
            'value': default_pairs,
            'notes': f'Pairs at miRDB>{DEFAULT_MIRDB}, |r|>{DEFAULT_CORR}',
        },
        {
            'metric': 'fixed_xgboost_accuracy',
            'value': xgb_metrics['accuracy'],
            'notes': f"From {xgb_metrics['source']}",
        },
        {
            'metric': 'fixed_xgboost_macro_f1',
            'value': xgb_metrics['macro_f1'],
            'notes': 'Same value for every threshold row',
        },
        {
            'metric': 'fixed_xgboost_macro_auroc',
            'value': xgb_metrics['macro_auroc'],
            'notes': 'Same value for every threshold row',
        },
        {
            'metric': 'pair_count_min',
            'value': pair_min,
            'notes': 'Most restrictive threshold combination with fewest pairs',
        },
        {
            'metric': 'pair_count_max',
            'value': pair_max,
            'notes': 'Most permissive threshold combination with most pairs',
        },
        {
            'metric': 'pair_counts_sensitive',
            'value': pair_sensitive,
            'notes': 'True if max pair count is at least 2x the minimum',
        },
        {
            'metric': 'xgboost_performance_stable',
            'value': True,
            'notes': (
                'By design: thresholds do not change XGBoost inputs; accuracy and macro F1 '
                'are identical across the grid.'
            ),
        },
        {
            'metric': 'default_thresholds_defensible',
            'value': default_defensible,
            'notes': default_notes,
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 72)
    print('THRESHOLD ROBUSTNESS ANALYSIS')
    print('=' * 72)
    print(XGBOOST_NOTE)
    print(
        'Evaluating pair-count sensitivity across thresholds; XGBoost metrics are computed '
        'once on fixed expression features.',
        flush=True,
    )

    xgb_metrics = load_fixed_xgboost_metrics()
    print(
        f"Fixed XGBoost test performance: accuracy={xgb_metrics['accuracy']:.4f}, "
        f"macro_f1={xgb_metrics['macro_f1']:.4f}, "
        f"macro_auroc={xgb_metrics['macro_auroc']:.4f}",
        flush=True,
    )

    de_mirnas, _, mirdb, train_mirna, train_mrna = load_pair_inputs()
    print('Precomputing train-set Pearson correlations for miRDB score > 50...', flush=True)
    candidates = precompute_candidate_correlations(de_mirnas, mirdb, train_mirna, train_mrna)
    print(f'Candidate rows before threshold filtering: {len(candidates)}', flush=True)

    results_rows = []
    pair_count_rows = []
    total = len(MIRDB_THRESHOLDS) * len(CORRELATION_CUTOFFS)
    combo_i = 0

    default_n_pairs = None

    for mirdb_threshold in MIRDB_THRESHOLDS:
        for corr_cutoff in CORRELATION_CUTOFFS:
            combo_i += 1
            pairs_df = filter_pairs(candidates, mirdb_threshold, corr_cutoff)
            n_pairs, n_mirnas, n_mrnas = count_pair_stats(pairs_df)

            if mirdb_threshold == DEFAULT_MIRDB and corr_cutoff == DEFAULT_CORR:
                default_n_pairs = n_pairs

            print(
                f'[{combo_i}/{total}] miRDB > {mirdb_threshold}, |r| > {corr_cutoff}: '
                f'pairs={n_pairs}, miRNAs={n_mirnas}, mRNAs={n_mrnas}',
                flush=True,
            )

            pair_count_rows.append({
                'mirdb_threshold': mirdb_threshold,
                'correlation_cutoff': corr_cutoff,
                'n_pairs': n_pairs,
                'n_unique_mirnas': n_mirnas,
                'n_unique_mrnas': n_mrnas,
            })

            results_rows.append({
                'mirdb_threshold': mirdb_threshold,
                'correlation_cutoff': corr_cutoff,
                'n_pairs': n_pairs,
                'n_unique_mirnas': n_mirnas,
                'n_unique_mrnas': n_mrnas,
                'accuracy': xgb_metrics['accuracy'],
                'macro_f1': xgb_metrics['macro_f1'],
                'f1_normal': xgb_metrics['per_class_f1']['normal'],
                'f1_primary': xgb_metrics['per_class_f1']['primary'],
                'f1_metastatic': xgb_metrics['per_class_f1']['metastatic'],
                'macro_auroc': xgb_metrics['macro_auroc'],
                'best_params': 'n/a (fixed saved model; no retraining in threshold sweep)',
                'notes': XGBOOST_NOTE,
            })

    pair_counts_df = pd.DataFrame(pair_count_rows)
    if default_n_pairs is None:
        default_n_pairs = 0
    if default_n_pairs > 0:
        pair_counts_df['percent_of_default_pairs'] = (
            100.0 * pair_counts_df['n_pairs'] / default_n_pairs
        ).round(1)
    else:
        pair_counts_df['percent_of_default_pairs'] = np.where(
            pair_counts_df['n_pairs'] == 0, 0.0, np.nan
        )

    results_df = pd.DataFrame(results_rows)
    if default_n_pairs > 0:
        results_df['percent_of_default_pairs'] = (
            100.0 * results_df['n_pairs'] / default_n_pairs
        ).round(1)

    pair_min = int(pair_counts_df['n_pairs'].min())
    pair_max = int(pair_counts_df['n_pairs'].max())
    default_defensible, default_notes = assess_default_defensibility(
        default_n_pairs, pair_min, pair_max, pair_counts_df
    )
    summary_df = build_summary(
        results_df, pair_counts_df, xgb_metrics, default_defensible, default_notes
    )

    results_path = TABLES_DIR / 'threshold_robustness_results.csv'
    pair_counts_path = TABLES_DIR / 'threshold_robustness_pair_counts.csv'
    summary_path = TABLES_DIR / 'threshold_robustness_summary.csv'
    fig_acc = FIGURES_DIR / 'threshold_robustness_heatmap_accuracy.png'
    fig_f1 = FIGURES_DIR / 'threshold_robustness_heatmap_macro_f1.png'
    fig_pairs = FIGURES_DIR / 'threshold_robustness_pair_counts.png'
    fig_pairs_heatmap = FIGURES_DIR / 'threshold_robustness_heatmap_pair_counts.png'

    results_df.to_csv(results_path, index=False)
    pair_counts_df.to_csv(pair_counts_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    acc_pivot = results_df.pivot(
        index='mirdb_threshold', columns='correlation_cutoff', values='accuracy'
    ).sort_index(ascending=False)
    f1_pivot = results_df.pivot(
        index='mirdb_threshold', columns='correlation_cutoff', values='macro_f1'
    ).sort_index(ascending=False)
    pairs_pivot = pair_counts_df.pivot(
        index='mirdb_threshold', columns='correlation_cutoff', values='n_pairs'
    ).sort_index(ascending=False)

    plot_heatmap(
        acc_pivot,
        'Held-out XGBoost accuracy (fixed across thresholds)',
        fig_acc,
        subtitle=PERFORMANCE_NOTE,
    )
    plot_heatmap(
        f1_pivot,
        'Held-out XGBoost macro F1 (fixed across thresholds)',
        fig_f1,
        subtitle=PERFORMANCE_NOTE,
    )
    plot_heatmap(
        pairs_pivot,
        'Regulatory pair count by threshold',
        fig_pairs_heatmap,
        fmt='.0f',
        cmap='YlOrRd',
    )

    # Line-style summary: pair counts vs correlation cutoff, one line per miRDB threshold.
    fig, ax = plt.subplots(figsize=(8, 6))
    for mirdb_threshold in sorted(MIRDB_THRESHOLDS):
        subset = pair_counts_df[pair_counts_df['mirdb_threshold'] == mirdb_threshold]
        ax.plot(
            subset['correlation_cutoff'],
            subset['n_pairs'],
            marker='o',
            label=f'miRDB > {mirdb_threshold}',
        )
    ax.set_xlabel('Correlation cutoff (|r| > cutoff)')
    ax.set_ylabel('Number of regulatory pairs (train data only)')
    ax.set_title('Regulatory pair counts across threshold combinations')
    ax.legend(title='miRDB threshold', fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_pairs, dpi=300)
    plt.close(fig)

    permissive = pair_counts_df.loc[pair_counts_df['n_pairs'].idxmax()]
    restrictive = pair_counts_df.loc[pair_counts_df['n_pairs'].idxmin()]

    print('\n' + '=' * 72)
    print('THRESHOLD ROBUSTNESS SUMMARY')
    print('=' * 72)
    print(f'Threshold combinations tested: {total}')
    print(
        f'Default threshold pair count (miRDB>{DEFAULT_MIRDB}, |r|>{DEFAULT_CORR}): '
        f'{default_n_pairs}'
    )
    print(f'Pair-count range across thresholds: {pair_min} to {pair_max}')
    print(
        f'Most permissive: miRDB>{int(permissive["mirdb_threshold"])}, '
        f'|r|>{permissive["correlation_cutoff"]} -> {int(permissive["n_pairs"])} pairs'
    )
    print(
        f'Most restrictive: miRDB>{int(restrictive["mirdb_threshold"])}, '
        f'|r|>{restrictive["correlation_cutoff"]} -> {int(restrictive["n_pairs"])} pairs'
    )
    print(
        f'Fixed XGBoost performance: accuracy={xgb_metrics["accuracy"]:.4f}, '
        f'macro_f1={xgb_metrics["macro_f1"]:.4f}'
    )
    print(
        f'Default threshold defensible as middle-ground pair selection: '
        f'{"yes" if default_defensible else "no"}'
    )
    print(f'  {default_notes}')
    print('\nSaved files:')
    for path in (
        results_path,
        pair_counts_path,
        summary_path,
        fig_acc,
        fig_f1,
        fig_pairs,
        fig_pairs_heatmap,
    ):
        print(f'  {path}')


if __name__ == '__main__':
    main()
