#!/usr/bin/env python3
"""Held-out test statistical comparisons for XGBoost, GAT, and TabNet (notebook 14).

Generates missing test prediction CSVs from saved model artifacts, validates
alignment across models, computes bootstrap confidence intervals (sample-level
and patient-cluster), pairwise metric differences, and McNemar tests.

Multiple-testing: Benjamini-Hochberg FDR is applied across the 3 pairwise
McNemar tests, and across the pairwise bootstrap difference comparisons (within
each bootstrap_type x metric family). Adjusted p-values are reported alongside
the raw p-values / CIs.
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from statsmodels.stats.multitest import multipletests

_NOTEBOOK_DIR = Path(__file__).resolve().parent
if str(_NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOK_DIR))

warnings.filterwarnings('ignore')

PROCESSED_DIR = Path('data/processed')
TABLES_DIR = Path('results/tables')
FIGURES_DIR = Path('results/figures')
MODELS_DIR = Path('models')

CLASS_NAMES = ['normal', 'primary', 'metastatic']
CLASS_IDS = [0, 1, 2]
LABEL_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_LABEL = {idx: name for name, idx in LABEL_TO_IDX.items()}

MODEL_FILES = {
    'XGBoost': TABLES_DIR / 'xgboost_test_predictions.csv',
    'Multi-network GAT': TABLES_DIR / 'gat_test_predictions.csv',
    'TabNet': TABLES_DIR / 'tabnet_test_predictions.csv',
}

MODEL_ARTIFACTS = {
    'XGBoost': MODELS_DIR / 'xgboost_model.pkl',
    'Multi-network GAT': MODELS_DIR / 'gat_baseline.pt',
    'TabNet': MODELS_DIR / 'tabnet_model.zip',
}

N_BOOTSTRAP = 10_000
RANDOM_STATE = 42
PROB_TOL = 1e-3


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_patient_map() -> pd.Series:
    patients = pd.read_csv(PROCESSED_DIR / 'patient_ids.csv')
    return patients.set_index('sample_id')['patient_id']


def predictions_to_dataframe(
    sample_ids: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    patient_map: pd.Series,
) -> pd.DataFrame:
    rows = []
    for sample_id, true_idx, pred_idx, probs in zip(sample_ids, y_true, y_pred, y_prob):
        row = {
            'sample_id': sample_id,
            'patient_id': patient_map.loc[sample_id],
            'true_label': IDX_TO_LABEL[int(true_idx)],
            'pred_label': IDX_TO_LABEL[int(pred_idx)],
        }
        for class_i, class_name in enumerate(CLASS_NAMES):
            row[f'prob_{class_name}'] = float(probs[class_i])
        rows.append(row)
    return pd.DataFrame(rows)


def generate_xgboost_predictions(path: Path, patient_map: pd.Series) -> pd.DataFrame:
    xgb_mod = _load_module('xgb_baseline', _NOTEBOOK_DIR / '04_xgboost_baseline.py')
    _, X_test, _, y_test, _ = xgb_mod.load_data()
    model = pd.read_pickle(MODEL_ARTIFACTS['XGBoost'])
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    df = predictions_to_dataframe(
        sample_ids=X_test.index.tolist(),
        y_true=y_test.values,
        y_pred=np.asarray(y_pred),
        y_prob=np.asarray(y_prob),
        patient_map=patient_map,
    )
    df.to_csv(path, index=False)
    return df


def generate_gat_predictions(path: Path, patient_map: pd.Series) -> pd.DataFrame:
    gat_mod = _load_module('gat_baseline', _NOTEBOOK_DIR / '05_gat_baseline.py')
    checkpoint = torch.load(
        MODEL_ARTIFACTS['Multi-network GAT'],
        map_location=gat_mod.DEVICE,
        weights_only=False,
    )
    nodes = list(checkpoint['nodes'])
    best_params = dict(checkpoint['best_params'])

    x_all, y_all, train_ids, test_ids = gat_mod.load_data()
    train_mirna_cols = pd.read_csv(PROCESSED_DIR / 'train_mirna.csv', index_col=0).columns.tolist()
    node_types = gat_mod.infer_node_types(nodes, train_mirna_cols)
    edge_index_by_network, edge_attr_by_network = gat_mod.load_graphs(nodes)

    dataset_by_id, train_ids, test_ids, _ = gat_mod.build_dataset(
        x_all=x_all,
        y_all=y_all,
        train_ids=train_ids,
        test_ids=test_ids,
        nodes=nodes,
    )
    dataset_by_id = {sid: data.to(gat_mod.DEVICE) for sid, data in dataset_by_id.items()}
    node_types = node_types.to(gat_mod.DEVICE)
    edge_index_by_network = {k: v.to(gat_mod.DEVICE) for k, v in edge_index_by_network.items()}
    edge_attr_by_network = {k: v.to(gat_mod.DEVICE) for k, v in edge_attr_by_network.items()}

    model = gat_mod.make_model(
        best_params,
        len(nodes),
        node_types,
        edge_index_by_network,
        edge_attr_by_network,
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    metrics = gat_mod.evaluate_ids(model=model, dataset_by_id=dataset_by_id, ids=test_ids)
    df = predictions_to_dataframe(
        sample_ids=list(test_ids),
        y_true=metrics['y_true'],
        y_pred=metrics['y_pred'],
        y_prob=metrics['y_prob'],
        patient_map=patient_map,
    )
    df.to_csv(path, index=False)
    return df


def generate_tabnet_predictions(path: Path, patient_map: pd.Series) -> pd.DataFrame:
    tabnet_mod = _load_module('tabnet_baseline', _NOTEBOOK_DIR / '06_tabnet_baseline.py')
    from pytorch_tabnet.tab_model import TabNetClassifier

    X_train, X_test, _, y_test = tabnet_mod.load_data()
    X_test_np = X_test.fillna(X_train.median()).values.astype(np.float32)

    model = TabNetClassifier()
    zip_path = MODELS_DIR / 'tabnet_model.zip'
    if not zip_path.exists():
        raise FileNotFoundError(f'Missing TabNet artifact: {zip_path}')
    model.load_model(str(zip_path))

    y_pred = model.predict(X_test_np)
    y_prob = model.predict_proba(X_test_np)
    df = predictions_to_dataframe(
        sample_ids=X_test.index.tolist(),
        y_true=y_test.values,
        y_pred=np.asarray(y_pred),
        y_prob=np.asarray(y_prob),
        patient_map=patient_map,
    )
    df.to_csv(path, index=False)
    return df


def ensure_predictions(patient_map: pd.Series) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    preds: Dict[str, pd.DataFrame] = {}
    status: Dict[str, str] = {}
    generators = {
        'XGBoost': generate_xgboost_predictions,
        'Multi-network GAT': generate_gat_predictions,
        'TabNet': generate_tabnet_predictions,
    }
    for model_name, path in MODEL_FILES.items():
        if path.exists():
            preds[model_name] = pd.read_csv(path)
            if 'patient_id' not in preds[model_name].columns:
                preds[model_name]['patient_id'] = preds[model_name]['sample_id'].map(patient_map)
                preds[model_name].to_csv(path, index=False)
            status[model_name] = 'loaded existing'
        else:
            preds[model_name] = generators[model_name](path, patient_map)
            status[model_name] = 'generated from saved model'
    return preds, status


def validate_predictions(preds: Dict[str, pd.DataFrame]) -> None:
    model_names = list(preds.keys())
    base_name = model_names[0]
    base = preds[base_name].sort_values('sample_id').reset_index(drop=True)

    for name in model_names[1:]:
        other = preds[name].sort_values('sample_id').reset_index(drop=True)
        if set(base['sample_id']) != set(other['sample_id']):
            missing_a = set(base['sample_id']) - set(other['sample_id'])
            missing_b = set(other['sample_id']) - set(base['sample_id'])
            raise ValueError(
                f'sample_id mismatch between {base_name} and {name}: '
                f'missing in {name}={sorted(missing_a)}; missing in {base_name}={sorted(missing_b)}'
            )
        if not base['true_label'].equals(other.sort_values('sample_id').reset_index(drop=True)['true_label']):
            raise ValueError(f'true_label mismatch between {base_name} and {name}')

    for name, df in preds.items():
        sorted_df = df.sort_values('sample_id').reset_index(drop=True)
        if sorted_df['true_label'].nunique() < 1:
            raise ValueError(f'{name}: no true labels found')
        invalid = set(sorted_df['true_label'].unique()) - set(CLASS_NAMES)
        if invalid:
            raise ValueError(f'{name}: invalid labels {invalid}')
        prob_cols = [f'prob_{c}' for c in CLASS_NAMES]
        missing_probs = [c for c in prob_cols if c not in sorted_df.columns]
        if missing_probs:
            raise ValueError(f'{name}: missing probability columns {missing_probs}')
        prob_sums = sorted_df[prob_cols].sum(axis=1)
        if not np.allclose(prob_sums, 1.0, atol=PROB_TOL):
            bad = sorted_df.loc[~np.isclose(prob_sums, 1.0, atol=PROB_TOL), 'sample_id'].tolist()
            raise ValueError(f'{name}: probabilities do not sum to 1 for samples {bad[:5]}')


def labels_to_array(labels: pd.Series) -> np.ndarray:
    return labels.map(LABEL_TO_IDX).to_numpy()


def compute_accuracy_macro_f1(df: pd.DataFrame) -> Tuple[float, float]:
    y_true = labels_to_array(df['true_label'])
    y_pred = labels_to_array(df['pred_label'])
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return float(acc), float(macro_f1)


def compute_macro_auroc_arrays(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(
            roc_auc_score(
                y_true,
                y_prob,
                labels=CLASS_IDS,
                multi_class='ovr',
                average='macro',
            )
        )
    except ValueError:
        return None


def _prepare_model_arrays(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    df = df.sort_values('sample_id').reset_index(drop=True)
    prob_cols = [f'prob_{c}' for c in CLASS_NAMES]
    return {
        'df': df,
        'y_true': labels_to_array(df['true_label']),
        'y_pred': labels_to_array(df['pred_label']),
        'y_prob': df[prob_cols].to_numpy(dtype=float),
        'patient_ids': df['patient_id'].to_numpy(),
    }


def _metrics_from_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Tuple[float, float, Optional[float]]:
    acc = float((y_true == y_pred).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    auc = compute_macro_auroc_arrays(y_true, y_prob) if y_prob is not None else None
    return acc, macro_f1, auc


def _patient_index_map(patient_ids: np.ndarray) -> Dict[str, np.ndarray]:
    mapping: Dict[str, List[int]] = {}
    for idx, patient in enumerate(patient_ids):
        mapping.setdefault(str(patient), []).append(idx)
    return {patient: np.array(idxs, dtype=int) for patient, idxs in mapping.items()}


def _bootstrap_indices_sample(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=(N_BOOTSTRAP, n))


def _patient_cluster_setup(patient_ids: np.ndarray, rng: np.random.Generator) -> Tuple[List[np.ndarray], np.ndarray]:
    uniques, codes = np.unique(patient_ids, return_inverse=True)
    idx_lists = [np.flatnonzero(codes == i) for i in range(len(uniques))]
    sampled_codes = rng.integers(0, len(uniques), size=(N_BOOTSTRAP, len(uniques)))
    return idx_lists, sampled_codes


def _index_for_patient_draw(idx_lists: List[np.ndarray], draw: np.ndarray) -> np.ndarray:
    return np.concatenate([idx_lists[c] for c in draw])


def _metric_values_from_indices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    idx_lists: List[np.ndarray],
    sampled_codes: np.ndarray,
    include_auc: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    acc_vals = np.empty(N_BOOTSTRAP, dtype=float)
    f1_vals = np.empty(N_BOOTSTRAP, dtype=float)
    auc_vals = []
    for i in range(N_BOOTSTRAP):
        idx = _index_for_patient_draw(idx_lists, sampled_codes[i])
        acc_vals[i] = float((y_true[idx] == y_pred[idx]).mean())
        f1_vals[i] = float(f1_score(y_true[idx], y_pred[idx], average='macro', zero_division=0))
        if include_auc:
            auc = compute_macro_auroc_arrays(y_true[idx], y_prob[idx])
            if auc is not None:
                auc_vals.append(auc)
    return acc_vals, f1_vals, np.array(auc_vals, dtype=float), len(auc_vals)


def _metric_values_from_sample_indices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    boot_idx: np.ndarray,
    include_auc: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    acc_vals = (y_pred[boot_idx] == y_true[boot_idx]).mean(axis=1)
    f1_vals = np.empty(N_BOOTSTRAP, dtype=float)
    auc_vals = []
    for i in range(N_BOOTSTRAP):
        idx = boot_idx[i]
        f1_vals[i] = f1_score(y_true[idx], y_pred[idx], average='macro', zero_division=0)
        if include_auc:
            auc = compute_macro_auroc_arrays(y_true[idx], y_prob[idx])
            if auc is not None:
                auc_vals.append(auc)
    return acc_vals, f1_vals, np.array(auc_vals, dtype=float), len(auc_vals)


def bootstrap_metric_cis(
    preds: Dict[str, pd.DataFrame],
    sample_boot_idx: np.ndarray,
    patient_idx_lists: List[np.ndarray],
    patient_sampled_codes: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Tuple[float, float, float]]]]:
    rows = []
    main_cis: Dict[str, Dict[str, Tuple[float, float, float]]] = {
        model: {} for model in preds
    }

    reference = _prepare_model_arrays(next(iter(preds.values())))
    n_samples = len(reference['y_true'])
    n_patients = len(patient_idx_lists)

    for model_name, df in preds.items():
        print(f'  Bootstrap metrics for {model_name}...', flush=True)
        arrays = _prepare_model_arrays(df)
        y_true, y_pred, y_prob = arrays['y_true'], arrays['y_pred'], arrays['y_prob']
        point_acc, point_f1, point_auc = _metrics_from_arrays(y_true, y_pred, y_prob)

        bootstrap_runs = {
            'sample': _metric_values_from_sample_indices(
                y_true, y_pred, y_prob, sample_boot_idx, include_auc=False
            ),
            'patient_cluster': _metric_values_from_indices(
                y_true,
                y_pred,
                y_prob,
                patient_idx_lists,
                patient_sampled_codes,
                include_auc=True,
            ),
        }

        for bootstrap_type, (acc_vals, f1_vals, auc_vals, valid_auc) in bootstrap_runs.items():
            for metric, values, point in (
                ('accuracy', acc_vals, point_acc),
                ('macro_f1', f1_vals, point_f1),
            ):
                lower, upper = np.percentile(values, [2.5, 97.5])
                rows.append({
                    'model': model_name,
                    'bootstrap_type': bootstrap_type,
                    'metric': metric,
                    'point_estimate': point,
                    'ci_lower_95': float(lower),
                    'ci_upper_95': float(upper),
                    'n_resamples': N_BOOTSTRAP,
                    'n_samples': n_samples,
                    'n_patients': n_patients,
                })
                if bootstrap_type == 'patient_cluster':
                    main_cis[model_name][metric] = (point, float(lower), float(upper))

            if bootstrap_type == 'patient_cluster' and point_auc is not None and valid_auc > 0:
                lower, upper = np.percentile(auc_vals, [2.5, 97.5])
                rows.append({
                    'model': model_name,
                    'bootstrap_type': bootstrap_type,
                    'metric': 'macro_auroc',
                    'point_estimate': point_auc,
                    'ci_lower_95': float(lower),
                    'ci_upper_95': float(upper),
                    'n_resamples': valid_auc,
                    'n_samples': n_samples,
                    'n_patients': n_patients,
                })
                main_cis[model_name]['macro_auroc'] = (point_auc, float(lower), float(upper))

    return pd.DataFrame(rows), main_cis


def bootstrap_metric_differences(
    preds: Dict[str, pd.DataFrame],
    sample_boot_idx: np.ndarray,
    patient_idx_lists: List[np.ndarray],
    patient_sampled_codes: np.ndarray,
) -> pd.DataFrame:
    arrays = {name: _prepare_model_arrays(df) for name, df in preds.items()}
    pairs = [
        ('XGBoost', 'Multi-network GAT'),
        ('XGBoost', 'TabNet'),
        ('Multi-network GAT', 'TabNet'),
    ]
    rows = []

    for model_a, model_b in pairs:
        print(f'  Paired differences: {model_a} vs {model_b}...', flush=True)
        a = arrays[model_a]
        b = arrays[model_b]
        point_acc_a, point_f1_a, _ = _metrics_from_arrays(a['y_true'], a['y_pred'], a['y_prob'])
        point_acc_b, point_f1_b, _ = _metrics_from_arrays(b['y_true'], b['y_pred'], b['y_prob'])
        point_diffs = {
            'accuracy': point_acc_a - point_acc_b,
            'macro_f1': point_f1_a - point_f1_b,
        }

        for bootstrap_type, use_patient in (
            ('sample', False),
            ('patient_cluster', True),
        ):
            acc_diffs = np.empty(N_BOOTSTRAP, dtype=float)
            f1_diffs = np.empty(N_BOOTSTRAP, dtype=float)
            for i in range(N_BOOTSTRAP):
                idx = (
                    _index_for_patient_draw(patient_idx_lists, patient_sampled_codes[i])
                    if use_patient
                    else sample_boot_idx[i]
                )
                acc_a = float((a['y_true'][idx] == a['y_pred'][idx]).mean())
                acc_b = float((b['y_true'][idx] == b['y_pred'][idx]).mean())
                f1_a = float(f1_score(a['y_true'][idx], a['y_pred'][idx], average='macro', zero_division=0))
                f1_b = float(f1_score(b['y_true'][idx], b['y_pred'][idx], average='macro', zero_division=0))
                acc_diffs[i] = acc_a - acc_b
                f1_diffs[i] = f1_a - f1_b

            for metric, diffs in (('accuracy', acc_diffs), ('macro_f1', f1_diffs)):
                lower, upper = np.percentile(diffs, [2.5, 97.5])
                diff = point_diffs[metric]
                boot_p = _bootstrap_two_sided_p(diffs)
                if lower <= 0 <= upper:
                    note = 'CI includes 0; difference not clearly different from zero'
                elif diff > 0:
                    note = f'{model_a} higher than {model_b}'
                else:
                    note = f'{model_b} higher than {model_a}'
                rows.append({
                    'model_a': model_a,
                    'model_b': model_b,
                    'bootstrap_type': bootstrap_type,
                    'metric': metric,
                    'difference_a_minus_b': diff,
                    'ci_lower_95': float(lower),
                    'ci_upper_95': float(upper),
                    'bootstrap_p': boot_p,
                    'n_resamples': N_BOOTSTRAP,
                    'interpretation_note': note,
                })
    out = pd.DataFrame(rows)
    # Benjamini-Hochberg across the pairwise difference comparisons. Each
    # (bootstrap_type, metric) is its own family of 3 model-pair comparisons, so
    # the bootstrap p-values are corrected within that family. CI-based reading
    # is kept; the FDR column adds a multiplicity-aware significance flag.
    if not out.empty:
        out['bootstrap_p_fdr_bh'] = np.nan
        out['difference_significant_fdr_0_05'] = False
        adjusted_parts = []
        for _, grp_idx in out.groupby(['bootstrap_type', 'metric']).groups.items():
            sub = out.loc[list(grp_idx)]
            sub = _add_bh_fdr(sub, 'bootstrap_p', 'bootstrap_p_fdr_bh', 'difference_significant_fdr_0_05')
            adjusted_parts.append(sub)
        out = pd.concat(adjusted_parts).sort_index()
    return out


def _add_bh_fdr(df: pd.DataFrame, p_col: str, fdr_col: str, sig_col: str) -> pd.DataFrame:
    """Add Benjamini-Hochberg FDR-adjusted p-values for one family of tests.

    NaN p-values are left out of the correction (and stay NaN). Significance flag
    uses the adjusted p < 0.05.
    """
    out = df.copy()
    out[fdr_col] = np.nan
    if p_col not in out.columns or out.empty:
        out[sig_col] = False
        return out
    p = out[p_col].to_numpy(dtype=float)
    valid = np.isfinite(p)
    if valid.sum() > 0:
        _, padj, _, _ = multipletests(p[valid], alpha=0.05, method='fdr_bh')
        adj = np.full(len(out), np.nan, dtype=float)
        adj[np.flatnonzero(valid)] = padj
        out[fdr_col] = adj
    out[sig_col] = out[fdr_col] < 0.05
    return out


def _bootstrap_two_sided_p(diffs: np.ndarray) -> float:
    """Two-sided bootstrap p-value for H0: difference == 0.

    p = 2 * min(P(diff <= 0), P(diff >= 0)), clipped to [1/(B+1), 1], using the
    bootstrap distribution of the paired metric difference.
    """
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    n = len(diffs)
    if n == 0:
        return float('nan')
    prop_le = float((diffs <= 0).mean())
    prop_ge = float((diffs >= 0).mean())
    p = 2.0 * min(prop_le, prop_ge)
    return float(min(1.0, max(p, 1.0 / (n + 1))))


def run_mcnemar_tests(preds: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    aligned = {
        name: df.sort_values('sample_id').reset_index(drop=True)
        for name, df in preds.items()
    }
    pairs = [
        ('XGBoost', 'Multi-network GAT'),
        ('XGBoost', 'TabNet'),
        ('Multi-network GAT', 'TabNet'),
    ]
    rows = []
    for model_a, model_b in pairs:
        df_a = aligned[model_a]
        df_b = aligned[model_b]
        correct_a = (df_a['pred_label'] == df_a['true_label']).to_numpy()
        correct_b = (df_b['pred_label'] == df_b['true_label']).to_numpy()

        both = int(np.sum(correct_a & correct_b))
        a_only = int(np.sum(correct_a & ~correct_b))
        b_only = int(np.sum(~correct_a & correct_b))
        neither = int(np.sum(~correct_a & ~correct_b))

        discordant = a_only + b_only
        if discordant == 0:
            p_value = 1.0
            test_type = 'exact_binomial'
            note = 'No discordant pairs; models agree on all samples'
        else:
            result = binomtest(a_only, n=discordant, p=0.5, alternative='two-sided')
            p_value = float(result.pvalue)
            test_type = 'exact_binomial'
            if p_value < 0.05:
                winner = model_a if a_only > b_only else model_b
                note = f'Significant discordance; {winner} correct more often when models disagree'
            else:
                note = 'No significant discordance between model correctness'

        rows.append({
            'model_a': model_a,
            'model_b': model_b,
            'n_both_correct': both,
            'n_a_correct_b_wrong': a_only,
            'n_a_wrong_b_correct': b_only,
            'n_both_wrong': neither,
            'mcnemar_p': p_value,
            'test_type': test_type,
            'interpretation_note': note,
        })
    out = pd.DataFrame(rows)
    # Benjamini-Hochberg correction across the 3 pairwise McNemar tests (one
    # family). Report adjusted p alongside raw; the FDR column is the basis for
    # significance, the raw mcnemar_p is retained for transparency.
    out = _add_bh_fdr(out, 'mcnemar_p', 'mcnemar_p_fdr_bh', 'mcnemar_significant_fdr_0_05')
    return out


def plot_metric_cis(ci_df: pd.DataFrame, metric: str, path: Path, title: str) -> None:
    sub = ci_df[
        (ci_df['bootstrap_type'] == 'patient_cluster') & (ci_df['metric'] == metric)
    ].copy()
    sub = sub.sort_values('point_estimate', ascending=False)
    x = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        x,
        sub['point_estimate'],
        yerr=[
            sub['point_estimate'] - sub['ci_lower_95'],
            sub['ci_upper_95'] - sub['point_estimate'],
        ],
        fmt='o',
        color='#4C72B0',
        ecolor='#4C72B0',
        capsize=5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(sub['model'], rotation=15, ha='right')
    ax.set_ylabel(metric.replace('_', ' '))
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_pairwise_differences(diff_df: pd.DataFrame, path: Path) -> None:
    sub = diff_df[
        (diff_df['bootstrap_type'] == 'patient_cluster') & (diff_df['metric'] == 'macro_f1')
    ].copy()
    labels = [f"{row['model_a']} - {row['model_b']}" for _, row in sub.iterrows()]
    y = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(
        sub['difference_a_minus_b'],
        y,
        xerr=[
            sub['difference_a_minus_b'] - sub['ci_lower_95'],
            sub['ci_upper_95'] - sub['difference_a_minus_b'],
        ],
        fmt='o',
        color='#55A868',
        ecolor='#55A868',
        capsize=5,
    )
    ax.axvline(0.0, color='gray', linestyle='--', linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Macro F1 difference (model_a - model_b)')
    ax.set_title('Pairwise macro F1 differences (patient-cluster bootstrap 95% CI)')
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    patient_map = load_patient_map()
    preds, pred_status = ensure_predictions(patient_map)
    print('Validated prediction alignment across models.', flush=True)
    validate_predictions(preds)

    print('Running bootstrap confidence intervals...', flush=True)
    rng = np.random.default_rng(RANDOM_STATE)
    reference = _prepare_model_arrays(next(iter(preds.values())))
    sample_boot_idx = _bootstrap_indices_sample(len(reference['y_true']), rng)
    patient_idx_lists, patient_sampled_codes = _patient_cluster_setup(reference['patient_ids'], rng)
    bootstrap_df, main_cis = bootstrap_metric_cis(
        preds, sample_boot_idx, patient_idx_lists, patient_sampled_codes
    )
    print('Running paired metric differences...', flush=True)
    diff_df = bootstrap_metric_differences(
        preds, sample_boot_idx, patient_idx_lists, patient_sampled_codes
    )
    print('Running McNemar tests...', flush=True)
    mcnemar_df = run_mcnemar_tests(preds)

    bootstrap_path = TABLES_DIR / 'bootstrap_metric_cis.csv'
    diff_path = TABLES_DIR / 'model_metric_differences.csv'
    mcnemar_path = TABLES_DIR / 'mcnemar_tests.csv'

    bootstrap_df.to_csv(bootstrap_path, index=False)
    diff_df.to_csv(diff_path, index=False)
    mcnemar_df.to_csv(mcnemar_path, index=False)

    fig_acc = FIGURES_DIR / 'model_performance_ci_accuracy.png'
    fig_f1 = FIGURES_DIR / 'model_performance_ci_macro_f1.png'
    fig_diff = FIGURES_DIR / 'model_pairwise_differences.png'
    plot_metric_cis(bootstrap_df, 'accuracy', fig_acc, 'Held-out test accuracy (patient-cluster 95% CI)')
    plot_metric_cis(bootstrap_df, 'macro_f1', fig_f1, 'Held-out test macro F1 (patient-cluster 95% CI)')
    plot_pairwise_differences(diff_df, fig_diff)

    print('=' * 72)
    print('HELD-OUT TEST STATISTICAL COMPARISON')
    print('=' * 72)

    print('\nPrediction CSV status:')
    for model_name, status in pred_status.items():
        print(f'  {model_name:<22} {status} -> {MODEL_FILES[model_name]}')

    print('\nPoint estimates (accuracy / macro F1):')
    for model_name, df in preds.items():
        acc, f1 = compute_accuracy_macro_f1(df.sort_values('sample_id'))
        print(f'  {model_name:<22} accuracy={acc:.4f}  macro_f1={f1:.4f}')

    print('\nPatient-cluster bootstrap 95% CIs:')
    for model_name in preds:
        acc = main_cis[model_name].get('accuracy')
        f1 = main_cis[model_name].get('macro_f1')
        if acc:
            print(
                f'  {model_name:<22} accuracy={acc[0]:.4f} [{acc[1]:.4f}, {acc[2]:.4f}]'
            )
        if f1:
            print(
                f'  {model_name:<22} macro_f1={f1[0]:.4f} [{f1[1]:.4f}, {f1[2]:.4f}]'
            )

    print('\nMcNemar tests (raw p and BH-FDR across the 3 pairwise tests):')
    for _, row in mcnemar_df.iterrows():
        print(
            f"  {row['model_a']} vs {row['model_b']}: "
            f"p={row['mcnemar_p']:.4g} p_fdr={row['mcnemar_p_fdr_bh']:.4g} "
            f"sig_fdr={bool(row['mcnemar_significant_fdr_0_05'])} "
            f"(b={row['n_a_correct_b_wrong']}, c={row['n_a_wrong_b_correct']})"
        )

    print('\nPairwise macro F1 differences (patient-cluster 95% CI):')
    pair_sub = diff_df[
        (diff_df['bootstrap_type'] == 'patient_cluster') & (diff_df['metric'] == 'macro_f1')
    ]
    for _, row in pair_sub.iterrows():
        print(
            f"  {row['model_a']} - {row['model_b']}: "
            f"diff={row['difference_a_minus_b']:.4f} "
            f"[{row['ci_lower_95']:.4f}, {row['ci_upper_95']:.4f}] "
            f"p={row['bootstrap_p']:.4g} p_fdr={row['bootstrap_p_fdr_bh']:.4g} "
            f"sig_fdr={bool(row['difference_significant_fdr_0_05'])}"
        )

    print('\nSaved files:')
    for path in (
        *MODEL_FILES.values(),
        bootstrap_path,
        diff_path,
        mcnemar_path,
        fig_acc,
        fig_f1,
        fig_diff,
    ):
        print(f'  {path}')


if __name__ == '__main__':
    main()
