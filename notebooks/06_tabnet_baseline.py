#!/usr/bin/env python3
"""TabNet baseline for 3-class TNBC tissue classification.

REVIEW — suspected causes of ~72% test accuracy vs ~93% for XGBoost/GAT (do not auto-apply all):

1. Class weights (FIXED below): ``compute_class_weights`` previously normalized weights
   to sum=3 instead of mean=1 (XGBoost/GAT use inverse-freq / mean).
2. Architecture grid may underfit 146 features: n_d/n_a capped at 32; consider
   ``REVIEW_PARAM_GRID`` (wider dims, more steps) after re-running grid search.
3. Early stopping (FIXED below): now uses a custom macro-F1 ``eval_metric`` (class ``MacroF1``)
   so the early-stopping objective matches model selection (val macro F1). Previously it used
   ``balanced_accuracy``, a mismatched objective that could pick a suboptimal epoch.
4. Final refit trains on ~80% of train (fold-0 holdout) like GAT; not a bug but reduces
   data vs using full train for final fit.
5. TabNet sparsemax on z-scored miRNA+mRNA may over-sparsify; lambda_sparse=1e-4 is default.
"""

import sys
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

_NOTEBOOK_DIR = Path(__file__).resolve().parent
if str(_NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOK_DIR))

import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

from cv_utils import load_patient_groups, patient_holdout_val_split
from evaluation_metrics import (
    compute_classification_metrics,
    print_test_metrics,
    save_test_metrics,
)


class MacroF1(Metric):
    """Macro-F1 early-stopping metric for TabNet.

    Aligns the early-stopping objective with model selection: the grid search ranks
    configs by validation macro-F1 (grid_search_cv), so early stopping must monitor the
    same metric. Previously eval_metric=['balanced_accuracy'] picked the best epoch by a
    different objective, an inconsistency that could select a suboptimal epoch.
    """

    def __init__(self):
        self._name = 'macro_f1'
        self._maximize = True

    def __call__(self, y_true, y_score):
        y_pred = np.argmax(y_score, axis=1)
        return float(f1_score(y_true, y_pred, average='macro', zero_division=0))


PROCESSED_DIR = Path('data/processed')
RESULTS_DIR = Path('results')
TABLES_DIR = RESULTS_DIR / 'tables'
FIGURES_DIR = RESULTS_DIR / 'figures'
MODELS_DIR = Path('models')

LABEL_MAP = {'normal': 0, 'primary': 1, 'metastatic': 2}
CLASS_NAMES = [k for k, _ in sorted(LABEL_MAP.items(), key=lambda x: x[1])]
RANDOM_STATE = 42
MAX_EPOCHS = 300
PATIENCE = 30

PARAM_GRID = {
    'n_d': [8, 16, 32],
    'n_a': [8, 16, 32],
    'n_steps': [3, 5],
    'lr': [0.01, 0.001],
}

# REVIEW: wider search if re-tuning after class-weight fix (not used by default).
REVIEW_PARAM_GRID = {
    'n_d': [32, 64],
    'n_a': [32, 64],
    'n_steps': [5, 8],
    'lr': [0.02, 0.005, 0.001],
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed: int = RANDOM_STATE) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data():
    x_train_mirna = pd.read_csv(PROCESSED_DIR / 'train_mirna.csv', index_col=0)
    x_train_mrna = pd.read_csv(PROCESSED_DIR / 'train_mrna.csv', index_col=0)
    x_test_mirna = pd.read_csv(PROCESSED_DIR / 'test_mirna.csv', index_col=0)
    x_test_mrna = pd.read_csv(PROCESSED_DIR / 'test_mrna.csv', index_col=0)

    train_labels = pd.read_csv(PROCESSED_DIR / 'train_labels.csv', index_col='sample_id')[
        'tissue_group'
    ]
    test_labels = pd.read_csv(PROCESSED_DIR / 'test_labels.csv', index_col='sample_id')[
        'tissue_group'
    ]

    train_labels = train_labels.loc[x_train_mirna.index]
    x_train_mirna = x_train_mirna.loc[train_labels.index]
    x_train_mrna = x_train_mrna.loc[train_labels.index]

    test_labels = test_labels.loc[x_test_mirna.index]
    x_test_mirna = x_test_mirna.loc[test_labels.index]
    x_test_mrna = x_test_mrna.loc[test_labels.index]

    X_train = pd.concat([x_train_mirna, x_train_mrna], axis=1)
    X_test = pd.concat([x_test_mirna, x_test_mrna], axis=1)

    common_cols = X_train.columns.intersection(X_test.columns)
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    y_train = train_labels.map(LABEL_MAP)
    y_test = test_labels.map(LABEL_MAP)

    if y_train.isna().any() or y_test.isna().any():
        le = LabelEncoder()
        y_all = pd.concat([train_labels, test_labels])
        encoded = le.fit_transform(y_all.astype(str))
        y_train = pd.Series(encoded[:len(y_train)], index=y_train.index)
        y_test = pd.Series(encoded[len(y_train):], index=y_test.index)

    return X_train, X_test, y_train.astype(int), y_test.astype(int)


def prepare_feature_matrices(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Impute missing values with train-only medians (features are already train-fitted z-scores)."""
    train_median = X_train.median()
    X_train_np = X_train.fillna(train_median).values.astype(np.float32)
    X_test_np = X_test.fillna(train_median).values.astype(np.float32)
    return X_train_np, X_test_np


def compute_class_weights(y_train: np.ndarray) -> np.ndarray:
    """Inverse-frequency weights normalized to mean 1 (matches XGBoost / GAT)."""
    counts = np.bincount(y_train, minlength=len(CLASS_NAMES)).astype(float)
    counts[counts == 0] = 1.0
    weights = len(y_train) / (len(CLASS_NAMES) * counts)
    weights = weights / weights.mean()
    return weights


def sample_weights_for_labels(y_train: np.ndarray, class_weights: np.ndarray) -> np.ndarray:
    """Map per-class weights to per-sample weights for TabNet's fit() API."""
    return class_weights[y_train.astype(int)]


def train_fold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_d: int,
    n_a: int,
    n_steps: int,
    lr: float,
    class_weights: np.ndarray,
) -> Tuple[TabNetClassifier, float]:
    set_seed()
    model = TabNetClassifier(
        n_d=n_d,
        n_a=n_a,
        n_steps=n_steps,
        gamma=1.5,
        lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=lr),
        scheduler_params=dict(step_size=50, gamma=0.9),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='sparsemax',
        seed=RANDOM_STATE,
        verbose=0,
        device_name='cuda' if torch.cuda.is_available() else 'cpu',
    )
    model.fit(
        X_train=X_tr,
        y_train=y_tr,
        eval_set=[(X_val, y_val)],
        eval_name=['val'],
        # Monitor macro-F1 so early stopping matches grid-search selection (val macro F1).
        eval_metric=[MacroF1],
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        batch_size=64,
        virtual_batch_size=32,
        weights=sample_weights_for_labels(y_tr, class_weights),
    )
    y_val_pred = model.predict(X_val)
    val_macro_f1 = float(f1_score(y_val, y_val_pred, average='macro', zero_division=0))
    return model, val_macro_f1


def grid_search_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    n_folds: int = 5,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    keys = list(PARAM_GRID.keys())
    combos = list(product(*[PARAM_GRID[k] for k in keys]))
    print(f'Grid search: {len(combos)} combinations x {n_folds}-fold patient-aware CV')

    sgkf = StratifiedGroupKFold(
        n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE
    )

    results = []
    for ci, vals in enumerate(combos, 1):
        params = dict(zip(keys, vals))
        print(f'\n[{ci}/{len(combos)}] {params}')

        fold_val_macro_f1 = []
        for fold_i, (tr_idx, val_idx) in enumerate(sgkf.split(X_train, y_train, groups)):
            X_tr = X_train[tr_idx]
            y_tr = y_train[tr_idx]
            X_val = X_train[val_idx]
            y_val = y_train[val_idx]

            cw = compute_class_weights(y_tr)
            model, val_macro_f1 = train_fold(
                X_tr=X_tr, y_tr=y_tr,
                X_val=X_val, y_val=y_val,
                n_d=int(params['n_d']),
                n_a=int(params['n_a']),
                n_steps=int(params['n_steps']),
                lr=params['lr'],
                class_weights=cw,
            )
            fold_val_macro_f1.append(val_macro_f1)
            print(f'  fold {fold_i + 1}/{n_folds} val_macro_f1={val_macro_f1:.4f}')

        avg_f1 = float(np.mean(fold_val_macro_f1))
        std_f1 = float(np.std(fold_val_macro_f1))
        row = {**params, 'mean_val_macro_f1': avg_f1, 'std_val_macro_f1': std_f1}
        results.append(row)
        print(f'  => avg_val_macro_f1={avg_f1:.4f} +/- {std_f1:.4f}')

    df = pd.DataFrame(results)
    df = df.sort_values('mean_val_macro_f1', ascending=False).reset_index(drop=True)
    best_row = df.iloc[0]
    best_params = {k: best_row[k] for k in keys}
    print(f'\nBest params: {best_params}')
    print(f'Best mean val macro F1: {best_row["mean_val_macro_f1"]:.4f}')
    return best_params, df


def evaluate(model: TabNetClassifier, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    return compute_classification_metrics(
        y_true=np.asarray(y_test),
        y_pred=np.asarray(y_pred),
        y_prob=np.asarray(y_prob),
        class_names=CLASS_NAMES,
        class_ids=[0, 1, 2],
    )


def extract_importance(model: TabNetClassifier, feature_names: List[str]) -> pd.DataFrame:
    importance = model.feature_importances_
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance,
    })
    df = df.sort_values('importance', ascending=False).reset_index(drop=True)
    return df


def update_comparison_csv(
    path: Path,
    accuracy: float,
    macro_f1: float,
    per_class_auc: Dict[str, float],
) -> None:
    row = {
        'model': 'TabNet Baseline',
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'auc_roc_normal': per_class_auc.get('normal', float('nan')),
        'auc_roc_primary': per_class_auc.get('primary', float('nan')),
        'auc_roc_metastatic': per_class_auc.get('metastatic', float('nan')),
    }

    row_df = pd.DataFrame([row])

    if path.exists():
        existing = pd.read_csv(path)
        existing = existing[existing['model'] != 'TabNet Baseline']
        updated = pd.concat([existing, row_df], ignore_index=True)
    else:
        updated = row_df

    updated = updated[[
        'model',
        'accuracy',
        'macro_f1',
        'auc_roc_normal',
        'auc_roc_primary',
        'auc_roc_metastatic',
    ]]
    updated.to_csv(path, index=False)


def main() -> None:
    set_seed()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_data()
    print(f'Train matrix shape: {X_train.shape}')
    print(f'Test matrix shape:  {X_test.shape}')
    print(f'Class distribution train: {y_train.value_counts().sort_index().to_dict()}')
    _n_mi = sum(c.lower().startswith(('hsa-', 'mir', 'let-')) for c in X_train.columns)
    print(f'Features: {X_train.shape[1]} DE columns '
          f'({_n_mi} miRNA + {X_train.shape[1] - _n_mi} mRNA), train-only z-score from preprocessing.')

    X_train_np, X_test_np = prepare_feature_matrices(X_train, X_test)
    y_train_np = y_train.values
    y_test_np = y_test.values

    train_groups = load_patient_groups(X_train.index).values
    best_params, grid_df = grid_search_cv(X_train_np, y_train_np, train_groups)

    grid_path = TABLES_DIR / 'tabnet_grid_search.csv'
    grid_df.to_csv(grid_path, index=False)
    print(f'Saved grid search results to {grid_path}')

    train_sample_ids = X_train.index.tolist()
    tr_ids, val_ids = patient_holdout_val_split(
        train_sample_ids,
        y=y_train,
        groups=load_patient_groups(X_train.index),
        n_splits=5,
        val_fold=0,
        random_state=RANDOM_STATE,
    )
    tr_pos = [train_sample_ids.index(sid) for sid in tr_ids]
    val_pos = [train_sample_ids.index(sid) for sid in val_ids]
    X_tr, X_val = X_train_np[tr_pos], X_train_np[val_pos]
    y_tr, y_val = y_train_np[tr_pos], y_train_np[val_pos]
    print(f'Final train/val split: {len(tr_ids)} train samples, {len(val_ids)} val samples')

    cw = compute_class_weights(y_tr)
    model, _ = train_fold(
        X_tr=X_tr, y_tr=y_tr,
        X_val=X_val, y_val=y_val,
        n_d=int(best_params['n_d']),
        n_a=int(best_params['n_a']),
        n_steps=int(best_params['n_steps']),
        lr=best_params['lr'],
        class_weights=cw,
    )

    test_metrics = evaluate(
        model=model, X_test=X_test_np, y_test=y_test_np,
    )
    print_test_metrics(test_metrics)
    cm_fig, per_class_path = save_test_metrics(
        test_metrics, 'tabnet', FIGURES_DIR, TABLES_DIR
    )

    importance_df = extract_importance(model, list(X_train.columns))
    importance_path = TABLES_DIR / 'tabnet_importance.csv'
    importance_df.to_csv(importance_path, index=False)

    model_path = MODELS_DIR / 'tabnet_model.pt'
    model.save_model(str(model_path).replace('.pt', ''))

    comparison_path = TABLES_DIR / 'model_comparison.csv'
    update_comparison_csv(
        comparison_path,
        test_metrics['accuracy'],
        test_metrics['macro_f1'],
        test_metrics['per_class_auc'],
    )

    print('\nSaved artifacts:')
    print(f'- {model_path}')
    print(f'- {importance_path}')
    print(f'- {cm_fig}')
    print(f'- {per_class_path}')
    print(f'- {comparison_path}')
    print(f'- {grid_path}')


if __name__ == '__main__':
    main()
