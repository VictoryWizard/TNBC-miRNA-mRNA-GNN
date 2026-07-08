#!/usr/bin/env python3
"""Naive baselines (majority-class & stratified-random) on the held-out test set.

Gives the reader a floor for the model metrics: with imbalanced classes a trivial
rule can score deceptively high on accuracy, so 0.94 only means something relative
to these baselines. Reports accuracy and macro-F1 for:
  * majority_class  : always predict the most frequent TRAINING class
  * stratified_random : predict ~ training class proportions (expectation over seeds)
  * uniform_random  : predict each class with equal probability (expectation)

Outputs results/tables/naive_baselines.csv

Run from repo root:  python notebooks/30_naive_baselines.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

PROCESSED = Path('data/processed')
TABLES = Path('results/tables'); TABLES.mkdir(parents=True, exist_ok=True)
CLASSES = ['normal', 'primary', 'metastatic']
N_TRIALS = 2000
RNG = np.random.default_rng(42)


def _metrics(y_true, y_pred):
    return (accuracy_score(y_true, y_pred),
            f1_score(y_true, y_pred, labels=CLASSES, average='macro', zero_division=0))


def main():
    y_train = pd.read_csv(PROCESSED / 'train_labels.csv', index_col=0)['tissue_group']
    y_test = pd.read_csv(PROCESSED / 'test_labels.csv', index_col=0)['tissue_group'].values
    n = len(y_test)

    train_counts = y_train.value_counts()
    majority = train_counts.idxmax()
    priors = (train_counts.reindex(CLASSES).fillna(0) / len(y_train)).values

    print(f'Test samples: {n} | class counts: {dict(pd.Series(y_test).value_counts())}')
    print(f'Majority training class: {majority}')

    rows = []

    # 1) majority class (deterministic)
    acc, f1 = _metrics(y_test, np.array([majority] * n))
    rows.append({'baseline': 'majority_class', 'accuracy': acc, 'macro_f1': f1,
                 'accuracy_std': 0.0, 'macro_f1_std': 0.0})

    # 2) stratified-random and 3) uniform-random (expectation over trials)
    for name, p in [('stratified_random', priors),
                    ('uniform_random', np.ones(len(CLASSES)) / len(CLASSES))]:
        accs, f1s = [], []
        for _ in range(N_TRIALS):
            pred = RNG.choice(CLASSES, size=n, p=p)
            a, f = _metrics(y_test, pred)
            accs.append(a); f1s.append(f)
        rows.append({'baseline': name,
                     'accuracy': float(np.mean(accs)), 'macro_f1': float(np.mean(f1s)),
                     'accuracy_std': float(np.std(accs)), 'macro_f1_std': float(np.std(f1s))})

    df = pd.DataFrame(rows)
    out = TABLES / 'naive_baselines.csv'
    df.to_csv(out, index=False)
    print('\nNaive baselines (held-out test set):')
    print(df.to_string(index=False))
    print(f'\nSaved {out}')
    print('\nUse these as the floor when reporting XGBoost/GAT/TabNet accuracy & macro-F1.')


if __name__ == '__main__':
    main()
