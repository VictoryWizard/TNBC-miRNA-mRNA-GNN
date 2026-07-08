#!/usr/bin/env python3
"""XGBoost baseline for 3-class TNBC tissue classification."""

import sys
from pathlib import Path
import warnings

_NOTEBOOK_DIR = Path(__file__).resolve().parent
if str(_NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOK_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder

import shap
from cv_utils import load_patient_groups
from evaluation_metrics import (
    compute_classification_metrics,
    print_test_metrics,
    save_test_metrics,
)
import xgboost as xgb_lib
from xgboost import XGBClassifier


warnings.filterwarnings('ignore')


def _xgb_gpu_kwargs() -> dict:
    """Return XGBoost device kwargs when a CUDA GPU is available, else empty dict.

    Handles both XGBoost >= 2.0 (device='cuda') and < 2.0 (tree_method='gpu_hist').
    """
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

PROCESSED_DIR = Path('data/processed')
RESULTS_DIR = Path('results')
TABLES_DIR = RESULTS_DIR / 'tables'
FIGURES_DIR = RESULTS_DIR / 'figures'
MODELS_DIR = Path('models')


def load_data():
    """Load and align miRNA/mRNA train and test sets with labels."""
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

    # Ensure sample alignment between feature tables and labels.
    train_labels = train_labels.loc[x_train.index]
    x_train = x_train.loc[train_labels.index]
    x_train_mrna = x_train_mrna.loc[train_labels.index]

    test_labels = test_labels.loc[x_test.index]
    x_test = x_test.loc[test_labels.index]
    x_test_mrna = x_test_mrna.loc[test_labels.index]

    # Concatenate miRNA and mRNA features horizontally.
    X_train = pd.concat([x_train, x_train_mrna], axis=1)
    X_test = pd.concat([x_test, x_test_mrna], axis=1)

    # Keep column compatibility across train/test.
    common_cols = X_train.columns.intersection(X_test.columns)
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    # Map labels to required numeric encoding: normal=0, primary=1, metastatic=2.
    label_map = {'normal': 0, 'primary': 1, 'metastatic': 2}
    y_train = train_labels.map(label_map)
    y_test = test_labels.map(label_map)

    # Safety fallback if labels are already numeric in this environment.
    if y_train.isna().any() or y_test.isna().any():
        le = LabelEncoder()
        y_all = pd.concat([train_labels, test_labels])
        encoded = le.fit_transform(y_all.astype(str))
        y_train = pd.Series(encoded[: len(y_train)], index=y_train.index)
        y_test = pd.Series(encoded[len(y_train):], index=y_test.index)
        # update mapping used in outputs.
        class_to_label = {idx: name for idx, name in enumerate(le.classes_)}
    else:
        class_to_label = {0: 'normal', 1: 'primary', 2: 'metastatic'}

    return X_train, X_test, y_train.astype(int), y_test.astype(int), class_to_label


def train_model(X_train, y_train, groups):
    """Run 5-fold patient-aware stratified group CV and fit best XGBoost model."""
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

    param_grid = {
        'max_depth': [3, 5, 7],
        'n_estimators': [100, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1],
    }

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        base_model,
        param_grid=param_grid,
        scoring='f1_macro',
        cv=cv,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )

    grid.fit(X_train, y_train, groups=groups)
    print(f"Best parameters: {grid.best_params_}")
    print(f"Best CV macro F1: {grid.best_score_:.4f}")

    best_model = grid.best_estimator_
    return best_model, grid.best_params_


def compute_metrics(model, X_test, y_test, class_to_label):
    """Evaluate held-out test set: accuracy, macro F1, per-class F1/AUROC, confusion matrix."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    class_ids = sorted(class_to_label.keys())
    class_names = [class_to_label[c] for c in class_ids]

    return compute_classification_metrics(
        y_true=np.asarray(y_test),
        y_pred=np.asarray(y_pred),
        y_prob=np.asarray(y_prob),
        class_names=class_names,
        class_ids=class_ids,
    )


def _shap_importance_to_series(model, X_train):
    """Compute average absolute SHAP importance and return feature importance series."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    if isinstance(shap_values, list):
        # Multi-class variant: one array per class, each [n_samples, n_features]
        means = [np.abs(np.asarray(sv)).mean(axis=0) for sv in shap_values]
        mean_abs = np.mean(np.vstack(means), axis=0)
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            # Typical shape for newer SHAP versions: (n_samples, n_features, n_classes)
            if arr.shape[0] == X_train.shape[0] and arr.shape[2] == model.n_classes_:
                mean_abs = np.abs(arr).mean(axis=(0, 2))
            # Alternative transpose: (n_classes, n_samples, n_features)
            elif arr.shape[1] == X_train.shape[0] and arr.shape[0] == model.n_classes_:
                mean_abs = np.abs(arr).mean(axis=(0, 2))
            else:
                raise ValueError('Unexpected SHAP array shape for 3D output.')
        elif arr.ndim == 2:
            # Binary/single-output fallback
            mean_abs = np.abs(arr).mean(axis=0)
        else:
            raise ValueError('Unexpected SHAP output shape.')

    return pd.Series(mean_abs, index=X_train.columns, name='mean_abs_shap')


def save_shap_plot(shap_series, output_path):
    """Save top 30 features as a horizontal SHAP importance bar chart."""
    top_features = shap_series.sort_values(ascending=False).head(30).iloc[::-1]

    plt.figure(figsize=(10, 8))
    top_features.plot(kind='barh')
    plt.title('Top 30 SHAP Feature Importance')
    plt.xlabel('mean |SHAP value|')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def update_model_comparison(path, metrics, per_class_auc):
    """Write or refresh model_comparison.csv with XGBoost as the first row."""
    row = {
        'model': 'XGBoost Baseline',
        'accuracy': metrics['accuracy'],
        'macro_f1': metrics['macro_f1'],
        'auc_roc_normal': per_class_auc.get('normal', float('nan')),
        'auc_roc_primary': per_class_auc.get('primary', float('nan')),
        'auc_roc_metastatic': per_class_auc.get('metastatic', float('nan')),
    }

    new_row = pd.DataFrame([row])
    if path.exists():
        existing = pd.read_csv(path)
        existing = existing[existing['model'] != 'XGBoost Baseline']
        updated = pd.concat([new_row, existing], ignore_index=True)
    else:
        updated = new_row

    # Ensure stable output order for downstream scripts.
    columns = [
        'model',
        'accuracy',
        'macro_f1',
        'auc_roc_normal',
        'auc_roc_primary',
        'auc_roc_metastatic',
    ]
    updated = updated.reindex(columns=columns)
    updated.to_csv(path, index=False)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test, class_to_label = load_data()
    print(f"Train matrix shape: {X_train.shape}")
    print(f"Test matrix shape:  {X_test.shape}")
    print(f"Class distribution train: {y_train.value_counts().sort_index().to_dict()}")

    train_groups = load_patient_groups(X_train.index).values
    model, best_params = train_model(X_train, y_train, train_groups)

    # Train final model on full training set using the best hyperparameters.
    model.set_params(**best_params)
    model.fit(X_train, y_train)

    test_metrics = compute_metrics(model, X_test, y_test, class_to_label)
    print_test_metrics(test_metrics)
    cm_fig, per_class_path = save_test_metrics(
        test_metrics, 'xgboost', FIGURES_DIR, TABLES_DIR
    )

    # Compute and save SHAP importance.
    shap_series = _shap_importance_to_series(model, X_train)

    importance_path = TABLES_DIR / 'xgboost_importance.csv'
    shap_series.to_csv(importance_path, header=['mean_abs_shap'], index_label='feature')

    plot_path = FIGURES_DIR / 'xgboost_shap.png'
    save_shap_plot(shap_series, plot_path)

    # Save trained model.
    model_path = MODELS_DIR / 'xgboost_model.pkl'
    pd.to_pickle(model, model_path)

    # Save summary file.
    metrics = {
        'accuracy': test_metrics['accuracy'],
        'macro_f1': test_metrics['macro_f1'],
    }
    summary_path = TABLES_DIR / 'model_comparison.csv'
    update_model_comparison(summary_path, metrics, test_metrics['per_class_auc'])

    print('\nSaved artifacts:')
    print(f"- {model_path}")
    print(f"- {importance_path}")
    print(f"- {plot_path}")
    print(f"- {cm_fig}")
    print(f"- {per_class_path}")
    print(f"- {summary_path}")


if __name__ == '__main__':
    main()
