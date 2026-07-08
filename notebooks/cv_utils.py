"""Patient-aware cross-validation helpers for TNBC baseline scripts."""

from pathlib import Path
from typing import List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PROCESSED_DIR = Path('data/processed')
PATIENT_IDS_PATH = PROCESSED_DIR / 'patient_ids.csv'


def load_patient_groups(sample_ids: Sequence[str]) -> pd.Series:
    """Return patient_id for each sample_id (index-aligned Series)."""
    if not PATIENT_IDS_PATH.exists():
        raise FileNotFoundError(
            f'Missing {PATIENT_IDS_PATH}. Re-run notebooks/01_preprocessing_DE.py.'
        )
    patients = pd.read_csv(PATIENT_IDS_PATH)
    if 'sample_id' in patients.columns:
        patients = patients.set_index('sample_id')
    groups = patients.loc[list(sample_ids), 'patient_id']
    return groups


def patient_holdout_val_split(
    sample_ids: Sequence[str],
    y: Union[pd.Series, np.ndarray, dict],
    groups: Union[pd.Series, np.ndarray],
    *,
    n_splits: int = 5,
    val_fold: int = 0,
    random_state: int = 42,
) -> Tuple[List[str], List[str]]:
    """Split sample_ids into train/val with disjoint patients (one SGKF fold)."""
    sample_ids_arr = np.asarray(list(sample_ids))
    n = len(sample_ids_arr)

    # Both y and groups are explicitly aligned to sample_ids when they are
    # label-keyed (dict) or index-keyed (Series). When passed as raw ndarrays
    # they are assumed to be POSITIONALLY aligned with sample_ids; we assert the
    # length to catch the most common misalignment bug rather than silently
    # desyncing labels/groups from samples.
    if isinstance(y, dict):
        y_arr = np.array([y[sid] for sid in sample_ids_arr])
    elif isinstance(y, pd.Series):
        y_arr = y.loc[sample_ids_arr].values
    else:
        y_arr = np.asarray(y)
        if y_arr.shape[0] != n:
            raise ValueError(
                f'y has length {y_arr.shape[0]} but sample_ids has length {n}; '
                'pass y as a pd.Series/dict to align by sample_id, or ensure '
                'positional alignment.'
            )

    if isinstance(groups, pd.Series):
        group_arr = groups.loc[sample_ids_arr].values
    else:
        group_arr = np.asarray(groups)
        if group_arr.shape[0] != n:
            raise ValueError(
                f'groups has length {group_arr.shape[0]} but sample_ids has '
                f'length {n}; pass groups as a pd.Series to align by sample_id, '
                'or ensure positional alignment.'
            )

    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    for fold_i, (tr_idx, val_idx) in enumerate(
        sgkf.split(sample_ids_arr, y_arr, group_arr)
    ):
        if fold_i == val_fold:
            train_patients = set(group_arr[tr_idx])
            val_patients = set(group_arr[val_idx])
            assert train_patients.isdisjoint(val_patients), 'PATIENT LEAKAGE in val split'
            return (
                sample_ids_arr[tr_idx].tolist(),
                sample_ids_arr[val_idx].tolist(),
            )

    raise ValueError(f'val_fold={val_fold} is out of range for n_splits={n_splits}')
