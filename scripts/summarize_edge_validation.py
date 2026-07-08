#!/usr/bin/env python3
"""Reproduce the cross-modal edge / miRNA-target overlap reported in the paper
(Sections 2.2 and 2.7: "~7% miRDB, 3% miRTarBase").

NOTE ON PROVENANCE
------------------
The per-edge database membership (the `in_mirdb` and `in_mirtarbase` columns)
was annotated once, when the top-150 cross-modal edges of each tissue network
were checked against miRDB v6.0 (data/raw/miRDB_v6.0_prediction_result.txt.gz,
included) and miRTarBase (external; the resulting membership flags are stored in
the shipped table rather than the raw miRTarBase download). Those annotations
live in:

    results/tables/edge_validation_db.csv        (per-edge flags)
    results/tables/edge_validation_hypergeom.csv (per-tissue hypergeometric test)

This script recomputes the summary overlap percentages the paper reports from
that shipped table, so the headline numbers are reproducible from the repo.

Usage:  python scripts/summarize_edge_validation.py
"""
from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd

# locate the per-edge validation table relative to the repo root
HERE = Path(__file__).resolve().parent
CANDIDATES = [
    HERE.parent / 'results' / 'tables' / 'edge_validation_db.csv',
    Path('results/tables/edge_validation_db.csv'),
]


def main() -> None:
    path = next((p for p in CANDIDATES if p.exists()), None)
    if path is None:
        sys.exit('edge_validation_db.csv not found under results/tables/.')
    df = pd.read_csv(path)
    df = df.drop_duplicates(['mirna', 'mrna'])  # unique cross-modal pairs

    mirdb = 100 * df['in_mirdb'].mean()
    mtb = 100 * df['in_mirtarbase'].mean()
    print(f'Cross-modal miRNA-mRNA edges checked: {len(df)}')
    print(f'Overlap with miRDB v6.0 : {mirdb:.1f}%   (paper: ~7%)')
    print(f'Overlap with miRTarBase : {mtb:.1f}%   (paper: ~3%)')
    print()
    print('By tissue:')
    full = pd.read_csv(path)
    for tissue, g in full.groupby('tissue'):
        print(f'  {tissue:10s} n={len(g):4d}  '
              f'miRDB {100 * g["in_mirdb"].mean():4.1f}%  '
              f'miRTarBase {100 * g["in_mirtarbase"].mean():4.1f}%')


if __name__ == '__main__':
    main()
