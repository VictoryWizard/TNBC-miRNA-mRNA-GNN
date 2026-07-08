#!/usr/bin/env python3
"""Binary XGBoost: primary tumor vs metastatic tissue (normal samples excluded).

Uses NESTED patient-aware cross-validation: an outer 5-fold StratifiedGroupKFold
produces genuinely out-of-sample pooled out-of-fold predictions, while an inner
StratifiedGroupKFold GridSearchCV re-selects hyperparameters within each outer
training fold (so model selection never sees the outer test rows). 95% CIs use a
patient-cluster bootstrap (resampling patients, not samples). Hyperparameter grid
matches notebook 04.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple

_NOTEBOOK_DIR = Path(__file__).resolve().parent
if str(_NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOK_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

import xgboost as xgb_lib
from cv_utils import load_patient_groups
from evaluation_metrics import print_test_metrics
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

PROCESSED_DIR = Path('data/processed')
RESULTS_DIR = Path('results')
TABLES_DIR = RESULTS_DIR / 'tables'
FIGURES_DIR = RESULTS_DIR / 'figures'

CLASS_NAMES = ('primary', 'metastatic')
CLASS_IDS = (0, 1)
LABEL_MAP = {'primary': 0, 'metastatic': 1}
N_SPLITS = 5
N_BOOTSTRAP = 1000
RANDOM_STATE = 42

XGB_PARAM_GRID = {
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
}


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
        print('XGBoost: using device=cuda')
        return {'device': 'cuda'}
    print('XGBoost: using tree_method=gpu_hist')
    return {'tree_method': 'gpu_hist', 'predictor': 'gpu_predictor'}


def load_primary_metastatic_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load processed miRNA+mRNA features for primary and metastatic samples only."""
    x_train = pd.read_csv(PROCESSED_DIR / 'train_mirna.csv', index_col=0)
    x_train_mrna = pd.read_csv(PROCESSED_DIR / 'train_mrna.csv', index_col=0)
    x_test = pd.read_csv(PROCESSED_DIR / 'test_mirna.csv', index_col=0)
    x_test_mrna = pd.read_csv(PROCESSED_DIR / 'test_mrna.csv', index_col=0)

    train_labels = pd.read_csv(PROCESSED_DIR / 'train_labels.csv', index_col='sample_id')[
        'tissue_group'
    ]
    test_labels = pd.read_csv(PROCESSED_DIR / 'test_labels.csv', index_col='sample_id')[
        'tissue_group'
    ]

    X_train = pd.concat([x_train, x_train_mrna], axis=1)
    X_test = pd.concat([x_test, x_test_mrna], axis=1)
    common_cols = X_train.columns.intersection(X_test.columns)
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    X = pd.concat([X_train, X_test], axis=0)
    labels = pd.concat([train_labels, test_labels], axis=0)
    labels = labels.loc[X.index]

    mask = labels.isin(['primary', 'metastatic'])
    X = X.loc[mask]
    labels = labels.loc[mask]

    y = labels.map(LABEL_MAP).astype(int)
    groups = load_patient_groups(X.index)
    return X, y, groups


def make_xgb_classifier(**extra_params) -> XGBClassifier:
    gpu_kwargs = _xgb_gpu_kwargs()
    params = dict(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
        **gpu_kwargs,
        **extra_params,
    )
    return XGBClassifier(**params)


def select_hyperparameters_inner(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    n_inner_splits: int,
) -> dict:
    """Inner-loop hyperparameter selection (nested CV).

    Runs GridSearchCV with patient-grouped StratifiedGroupKFold using ONLY the
    rows of the current outer training fold. The outer test fold is never seen
    here, so the resulting pooled out-of-fold predictions are genuinely
    out-of-sample (no model-selection leakage).
    """
    cv = StratifiedGroupKFold(
        n_splits=n_inner_splits, shuffle=True, random_state=RANDOM_STATE
    )
    grid = GridSearchCV(
        make_xgb_classifier(),
        param_grid=XGB_PARAM_GRID,
        scoring='f1_macro',
        cv=cv,
        n_jobs=-1,
        verbose=0,
        return_train_score=False,
    )
    grid.fit(X_train, y_train, groups=groups_train.values)
    return dict(grid.best_params_)


def pooled_cv_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Nested patient-aware CV with genuinely out-of-sample pooled predictions.

    Outer loop: StratifiedGroupKFold over patients. For each outer fold the
    hyperparameters are re-selected by an inner StratifiedGroupKFold GridSearchCV
    fit ONLY on the outer training rows (select_hyperparameters_inner), then a
    model with those params is fit on the outer training fold and used to predict
    the held-out outer test fold. Returns out-of-fold predictions, metastatic
    probabilities, and the per-fold selected hyperparameters.
    """
    outer_cv = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    # Inner splits capped so every inner fold can keep >=1 patient per class.
    n_inner_splits = max(2, N_SPLITS - 1)

    y_pred = np.zeros(len(y), dtype=int)
    y_prob_meta = np.zeros(len(y), dtype=float)
    fold_params: list[dict] = []

    X_arr = X.values
    y_arr = y.values
    groups_arr = groups.values

    for fold, (train_idx, test_idx) in enumerate(
        outer_cv.split(X_arr, y_arr, groups_arr), start=1
    ):
        train_patients = set(groups_arr[train_idx])
        test_patients = set(groups_arr[test_idx])
        assert train_patients.isdisjoint(test_patients), 'PATIENT LEAKAGE in CV fold'

        X_tr = X.iloc[train_idx]
        y_tr = y.iloc[train_idx]
        groups_tr = groups.iloc[train_idx]

        best_params = select_hyperparameters_inner(
            X_tr, y_tr, groups_tr, n_inner_splits
        )
        fold_params.append(best_params)

        model = make_xgb_classifier(**best_params)
        model.fit(X_tr, y_tr)
        y_pred[test_idx] = model.predict(X.iloc[test_idx])
        y_prob_meta[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]
        print(
            f'  Fold {fold}: train={len(train_idx)} test={len(test_idx)} '
            f'(patients {len(train_patients)}/{len(test_patients)}) '
            f'inner-selected params={best_params}'
        )

    y_prob = np.column_stack([1.0 - y_prob_meta, y_prob_meta])
    return y_pred, y_prob, fold_params


def _summarize_fold_params(fold_params: list[dict]) -> dict:
    """Modal (most-frequently selected) value per hyperparameter across folds.

    Descriptive only: with nested CV each outer fold has its own selected
    hyperparameters; this collapses them for reporting and does not represent a
    model fit/selected on the entire dataset.
    """
    from collections import Counter

    summary: dict = {}
    keys = sorted({k for params in fold_params for k in params})
    for key in keys:
        values = [params[key] for params in fold_params if key in params]
        if values:
            summary[key] = Counter(values).most_common(1)[0][0]
    return summary


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict:
    """Accuracy, macro F1, per-class F1, metastatic precision/recall, AUROC, confusion matrix."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    classes = np.array(CLASS_IDS)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    per_class_f1_vals = f1_score(
        y_true, y_pred, average=None, labels=classes, zero_division=0
    )
    per_class_f1 = {
        name: float(score) for name, score in zip(CLASS_NAMES, per_class_f1_vals)
    }

    per_class_auc = {}
    for idx, name in enumerate(CLASS_NAMES):
        y_bin = (y_true == classes[idx]).astype(int)
        if y_bin.sum() in (0, len(y_bin)):
            per_class_auc[name] = float('nan')
        else:
            per_class_auc[name] = float(roc_auc_score(y_bin, y_prob[:, idx]))

    cm = confusion_matrix(y_true, y_pred, labels=classes)

    return {
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'per_class_f1': per_class_f1,
        'per_class_auc': per_class_auc,
        'confusion_matrix': cm,
        'class_names': list(CLASS_NAMES),
        'precision_metastatic': float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        'recall_metastatic': float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        'auroc': float(roc_auc_score(y_true, y_prob[:, 1])),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RANDOM_STATE,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """95% percentile PATIENT-CLUSTER bootstrap CIs for accuracy and macro F1.

    Resamples patients (groups) with replacement rather than individual samples,
    so the CI reflects the unit of statistical independence (patients, since one
    patient may contribute several tissue samples). This matches the
    patient-cluster bootstrap in 14_statistical_tests.py.
    """
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    uniques = np.unique(groups)
    # Precompute the sample-row indices belonging to each unique patient.
    idx_by_patient = [np.flatnonzero(groups == g) for g in uniques]
    n_patients = len(uniques)

    acc_samples = np.empty(n_bootstrap)
    f1_samples = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        drawn = rng.integers(0, n_patients, size=n_patients)
        idx = np.concatenate([idx_by_patient[c] for c in drawn])
        yt = y_true[idx]
        yp = y_pred[idx]
        acc_samples[i] = accuracy_score(yt, yp)
        f1_samples[i] = f1_score(yt, yp, average='macro', zero_division=0)

    acc_ci = tuple(np.percentile(acc_samples, [2.5, 97.5]))
    f1_ci = tuple(np.percentile(f1_samples, [2.5, 97.5]))
    return acc_ci, f1_ci


def save_confusion_matrix(metrics: Dict, output_path: Path) -> None:
    class_names = metrics['class_names']
    cm = metrics['confusion_matrix']

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True label',
        xlabel='Predicted label',
        title='Primary vs metastatic — pooled 5-fold CV (OOF)',
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
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def metrics_to_row(
    metrics: Dict,
    acc_ci: Tuple[float, float],
    f1_ci: Tuple[float, float],
    n_primary: int,
    n_metastatic: int,
    best_params: dict,
) -> dict:
    cm = metrics['confusion_matrix']
    n_total = n_primary + n_metastatic
    return {
        'n_total': n_total,
        'n_primary': n_primary,
        'n_metastatic': n_metastatic,
        'pct_primary': n_primary / n_total,
        'pct_metastatic': n_metastatic / n_total,
        'accuracy': metrics['accuracy'],
        'accuracy_ci_low': acc_ci[0],
        'accuracy_ci_high': acc_ci[1],
        'macro_f1': metrics['macro_f1'],
        'macro_f1_ci_low': f1_ci[0],
        'macro_f1_ci_high': f1_ci[1],
        'f1_primary': metrics['per_class_f1']['primary'],
        'f1_metastatic': metrics['per_class_f1']['metastatic'],
        'precision_metastatic': metrics['precision_metastatic'],
        'recall_metastatic': metrics['recall_metastatic'],
        'auroc': metrics['auroc'],
        'auc_primary_ovr': metrics['per_class_auc']['primary'],
        'auc_metastatic_ovr': metrics['per_class_auc']['metastatic'],
        'cm_primary_primary': int(cm[0, 0]),
        'cm_primary_metastatic': int(cm[0, 1]),
        'cm_metastatic_primary': int(cm[1, 0]),
        'cm_metastatic_metastatic': int(cm[1, 1]),
        'max_depth': best_params.get('max_depth'),
        'n_estimators': best_params.get('n_estimators'),
        'learning_rate': best_params.get('learning_rate'),
        'cv_folds': N_SPLITS,
        'bootstrap_iterations': N_BOOTSTRAP,
    }


def print_summary(
    metrics: Dict,
    acc_ci: Tuple[float, float],
    f1_ci: Tuple[float, float],
    n_primary: int,
    n_metastatic: int,
    best_params: dict,
) -> None:
    n_total = n_primary + n_metastatic
    print('\n' + '=' * 72)
    print('PRIMARY vs METASTATIC CLASSIFICATION (normal samples excluded)')
    print('=' * 72)
    print(f'Samples:  {n_primary} primary + {n_metastatic} metastatic = {n_total} total')
    print(
        f'Balance:  primary {100 * n_primary / n_total:.1f}% | '
        f'metastatic {100 * n_metastatic / n_total:.1f}% '
        f'(ratio {n_primary / n_metastatic:.2f}:1)'
    )
    print(f'CV:       nested {N_SPLITS}-fold StratifiedGroupKFold (outer OOF preds, '
          f'inner GridSearchCV per fold; patient-cluster bootstrap CIs)')
    print(f'XGBoost:  {best_params} (modal across outer folds)')
    print('-' * 72)
    print(f"Accuracy:             {metrics['accuracy']:.4f}  "
          f'[95% CI {acc_ci[0]:.4f}, {acc_ci[1]:.4f}]')
    print(f"Macro F1:               {metrics['macro_f1']:.4f}  "
          f'[95% CI {f1_ci[0]:.4f}, {f1_ci[1]:.4f}]')
    print(f"F1 primary:             {metrics['per_class_f1']['primary']:.4f}")
    print(f"F1 metastatic:          {metrics['per_class_f1']['metastatic']:.4f}")
    print(f"Precision (metastatic): {metrics['precision_metastatic']:.4f}")
    print(f"Recall (metastatic):    {metrics['recall_metastatic']:.4f}")
    print(f"AUROC (binary):         {metrics['auroc']:.4f}")
    print('=' * 72)

    print_test_metrics(metrics, title='Pooled out-of-fold predictions (5-fold CV)')


def heldout_test_eval(X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> Tuple[Dict, dict, int, int]:
    """Train on TRAIN primary+met, evaluate on the held-out TEST primary+met.

    Mirrors the 3-class evaluation (grid-search HPs on train, score once on the
    held-out test) so primary-vs-metastatic is reported on the SAME held-out set,
    not a different (CV) scheme. The nested-CV result remains as the robustness
    estimate over the full cohort.
    """
    import json
    split = json.load(open(PROCESSED_DIR / 'split_indices.json'))
    train_ids = [s for s in split['train_sample_ids'] if s in X.index]
    test_ids = [s for s in split['test_sample_ids'] if s in X.index]
    Xtr, ytr, gtr = X.loc[train_ids], y.loc[train_ids], groups.loc[train_ids]
    Xte, yte = X.loc[test_ids], y.loc[test_ids]
    best = select_hyperparameters_inner(Xtr, ytr, gtr, max(2, N_SPLITS - 1))
    model = make_xgb_classifier(**best)
    model.fit(Xtr, ytr)
    y_prob = model.predict_proba(Xte)
    y_pred = model.predict(Xte)
    m = compute_binary_metrics(yte.values, y_pred, y_prob)
    return m, best, len(train_ids), len(test_ids)


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    X, y, groups = load_primary_metastatic_data()
    n_primary = int((y == 0).sum())
    n_metastatic = int((y == 1).sum())

    print(f'Feature matrix: {X.shape[0]} samples x {X.shape[1]} features')
    print(f'Class counts:   primary={n_primary}, metastatic={n_metastatic}')
    print(f'Unique patients: {groups.nunique()}')

    print('\nRunning nested patient-aware CV (inner GridSearchCV per outer fold)'
          ' with genuinely out-of-sample pooled OOF predictions...')
    y_pred, y_prob, fold_params = pooled_cv_predictions(X, y, groups)

    # Report the modal hyperparameters selected across outer folds. With nested
    # CV there is no single global "best_params"; this is a descriptive summary
    # of what the inner loops chose, NOT a model selected on the full dataset.
    best_params = _summarize_fold_params(fold_params)
    print(f'Modal inner-selected hyperparameters across folds: {best_params}')

    y_true = y.values
    metrics = compute_binary_metrics(y_true, y_pred, y_prob)

    print(f'\nPatient-cluster bootstrap ({N_BOOTSTRAP} iterations) for 95% CIs...')
    acc_ci, f1_ci = bootstrap_ci(y_true, y_pred, groups.values)

    print_summary(metrics, acc_ci, f1_ci, n_primary, n_metastatic, best_params)

    csv_path = TABLES_DIR / 'primary_vs_metastatic.csv'
    pd.DataFrame([metrics_to_row(
        metrics, acc_ci, f1_ci, n_primary, n_metastatic, best_params
    )]).to_csv(csv_path, index=False)

    fig_path = FIGURES_DIR / 'primary_vs_metastatic_confusion.png'
    save_confusion_matrix(metrics, fig_path)

    # Held-out test evaluation (same 51-sample split as the 3-class task) for a
    # consistent, apples-to-apples comparison; nested CV above is the robustness estimate.
    print('\n' + '=' * 72)
    print('PRIMARY vs METASTATIC — HELD-OUT TEST (same split as 3-class task)')
    print('=' * 72)
    ho_metrics, ho_params, n_tr, n_te = heldout_test_eval(X, y, groups)
    print(f'Train (primary+met): {n_tr} | Held-out test (primary+met): {n_te}')
    print(f"Accuracy: {ho_metrics['accuracy']:.4f} | Macro F1: {ho_metrics['macro_f1']:.4f} "
          f"| AUROC: {ho_metrics['auroc']:.4f}")
    print(f"F1 primary: {ho_metrics['per_class_f1']['primary']:.4f} | "
          f"F1 metastatic: {ho_metrics['per_class_f1']['metastatic']:.4f}")
    ho_row = metrics_to_row(ho_metrics, (float('nan'), float('nan')),
                            (float('nan'), float('nan')),
                            int((ho_metrics['confusion_matrix'][0].sum())),
                            int((ho_metrics['confusion_matrix'][1].sum())), ho_params)
    ho_row['evaluation'] = 'held_out_test'
    ho_path = TABLES_DIR / 'primary_vs_metastatic_heldout.csv'
    pd.DataFrame([ho_row]).to_csv(ho_path, index=False)

    print('\nSaved outputs:')
    print(f'  {csv_path}  (nested-CV, full cohort — robustness)')
    print(f'  {ho_path}  (held-out test — consistent with 3-class)')
    print(f'  {fig_path}')


if __name__ == '__main__':
    main()
