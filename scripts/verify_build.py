#!/usr/bin/env python3
"""Build guard: fail loudly if data/processed is NOT the paper's 118-feature,
patient-clean build. This is the check that would have caught the 146-build run.
Edit EXPECTED_* only if the panel legitimately changes."""
import sys, pathlib
import pandas as pd

EXPECTED_MIRNA, EXPECTED_MRNA = 86, 32        # 118 features total
EXPECTED_TRAIN, EXPECTED_TEST = 184, 51        # patient-clean split

P = pathlib.Path('data/processed')
ncols = lambda f: pd.read_csv(P / f, index_col=0, nrows=0).shape[1]
nrows = lambda f: sum(1 for _ in open(P / f)) - 1

mi, mr = ncols('train_mirna.csv'), ncols('train_mrna.csv')
ntr, nte = nrows('train_labels.csv'), nrows('test_labels.csv')
print(f"BUILD: {mi} miRNA + {mr} mRNA = {mi+mr} features | {ntr} train / {nte} test")

if not (mi == EXPECTED_MIRNA and mr == EXPECTED_MRNA and ntr == EXPECTED_TRAIN and nte == EXPECTED_TEST):
    sys.exit(f"!!! WRONG BUILD: expected {EXPECTED_MIRNA} miRNA + {EXPECTED_MRNA} mRNA "
             f"({EXPECTED_MIRNA+EXPECTED_MRNA}) and {EXPECTED_TRAIN}/{EXPECTED_TEST} split. "
             "Halting. Re-run notebook 01 — a stale/leaky build is in data/processed.")
print("BUILD OK — 118-feature patient-clean build.")
