#!/usr/bin/env python3
"""Print and save the model performance summary."""

from pathlib import Path

import pandas as pd


TABLES_DIR = Path('results') / 'tables'
RESULTS_DIR = Path('results')


def main():
    mc = pd.read_csv(TABLES_DIR / 'model_comparison.csv')

    L = []
    L.append('=' * 80)
    L.append('MODEL PERFORMANCE SUMMARY')
    L.append('=' * 80)
    L.append('')
    L.append(f'{"Model":<22} {"Accuracy":>10} {"Macro F1":>10} {"AUC-ROC Normal":>15} {"AUC-ROC Primary":>16} {"AUC-ROC Metastatic":>19}')
    L.append('-' * 80)

    for _, row in mc.iterrows():
        L.append(
            f'{row["model"]:<22} {row["accuracy"]:>10.4f} {row["macro_f1"]:>10.4f} '
            f'{row["auc_roc_normal"]:>15.4f} {row["auc_roc_primary"]:>16.4f} '
            f'{row["auc_roc_metastatic"]:>19.4f}'
        )

    L.append('-' * 80)
    L.append('')

    text = '\n'.join(L)
    print(text)

    out = RESULTS_DIR / 'model_performance_summary.txt'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f'Saved to {out}')


if __name__ == '__main__':
    main()
