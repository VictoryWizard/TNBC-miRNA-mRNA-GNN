"""Shared test-set classification metrics and reporting for baseline scripts."""

from pathlib import Path
from typing import Dict, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

DEFAULT_CLASS_NAMES = ('normal', 'primary', 'metastatic')
DEFAULT_CLASS_IDS = (0, 1, 2)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
    class_ids: Sequence[int] = DEFAULT_CLASS_IDS,
) -> Dict:
    """Compute accuracy, macro F1, per-class F1, OvR AUC, and confusion matrix.

    NOTE: ``y_prob`` columns MUST be ordered to match ``class_ids`` (i.e.
    column j holds P(class == class_ids[j])). Most sklearn estimators emit
    probabilities in ``estimator.classes_`` order; if that differs from
    ``class_ids`` the caller must reorder columns BEFORE calling this function.
    The assertion below guards the shape but cannot detect a silent column
    permutation, so reorder upstream.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    classes = np.array(class_ids)

    if y_prob.ndim != 2 or y_prob.shape[1] != len(classes):
        raise ValueError(
            f'y_prob must have shape (n_samples, {len(classes)}) with columns '
            f'aligned to class_ids={tuple(class_ids)}; got shape {y_prob.shape}.'
        )
    if y_prob.shape[0] != y_true.shape[0]:
        raise ValueError(
            f'y_prob has {y_prob.shape[0]} rows but y_true has {y_true.shape[0]}.'
        )

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    per_class_f1_vals = f1_score(
        y_true, y_pred, average=None, labels=classes, zero_division=0
    )
    per_class_f1 = {
        name: float(score) for name, score in zip(class_names, per_class_f1_vals)
    }

    y_true_bin = label_binarize(y_true, classes=classes)
    per_class_auc = {}
    for idx, name in enumerate(class_names):
        if y_true_bin[:, idx].sum() in (0, len(y_true_bin)):
            per_class_auc[name] = float('nan')
        else:
            per_class_auc[name] = float(roc_auc_score(y_true_bin[:, idx], y_prob[:, idx]))

    cm = confusion_matrix(y_true, y_pred, labels=classes)

    return {
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'per_class_f1': per_class_f1,
        'per_class_auc': per_class_auc,
        'confusion_matrix': cm,
        'class_names': list(class_names),
    }


def print_test_metrics(metrics: Dict, title: str = 'Held-out test set') -> None:
    """Print confusion matrix, per-class F1, and per-class AUROC."""
    class_names = metrics['class_names']
    cm = metrics['confusion_matrix']

    print(f'\n{title}')
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1:  {metrics['macro_f1']:.4f}")

    print('\nPer-class F1:')
    for name in class_names:
        print(f"  {name}: {metrics['per_class_f1'][name]:.4f}")

    print('\nPer-class AUROC (one-vs-rest):')
    for name in class_names:
        print(f"  {name}: {metrics['per_class_auc'][name]:.4f}")

    print('\nConfusion matrix (rows=true, cols=pred):')
    header = '          ' + '  '.join(f'{n:>10}' for n in class_names)
    print(header)
    for i, name in enumerate(class_names):
        row = '  '.join(f'{v:10d}' for v in cm[i])
        print(f'{name:>10}  {row}')


def save_test_metrics(
    metrics: Dict,
    model_slug: str,
    figures_dir: Path,
    tables_dir: Path,
) -> Tuple[Path, Path]:
    """Save confusion-matrix figure and per-class metrics CSV."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    class_names = metrics['class_names']
    cm = metrics['confusion_matrix']

    fig_path = figures_dir / f'{model_slug}_confusion_matrix.png'
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
        title=f'{model_slug} — confusion matrix (test)',
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], 'd'),
                ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black',
            )
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    per_class_rows = []
    for name in class_names:
        per_class_rows.append({
            'class': name,
            'f1': metrics['per_class_f1'][name],
            'auroc_ovr': metrics['per_class_auc'][name],
        })
    per_class_df = pd.DataFrame(per_class_rows)
    metrics_path = tables_dir / f'{model_slug}_per_class_metrics.csv'
    per_class_df.to_csv(metrics_path, index=False)

    summary_path = tables_dir / f'{model_slug}_test_summary.csv'
    pd.DataFrame([{
        'accuracy': metrics['accuracy'],
        'macro_f1': metrics['macro_f1'],
    }]).to_csv(summary_path, index=False)

    return fig_path, metrics_path
