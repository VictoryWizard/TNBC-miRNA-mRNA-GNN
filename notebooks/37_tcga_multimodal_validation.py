#!/usr/bin/env python3
"""Combined miRNA + mRNA TCGA-BRCA external transfer (the honest multimodal check).

Why this exists:
  The manuscript's external validation reports the mRNA signature (AUROC ~0.88)
  and the miRNA classifier (AUROC ~0.99) as two separate unimodal transfers, and
  notebook 19's pretrained_full_model path feeds the full model with the mRNA half
  imputed at zero. Neither is an honest multimodal transfer. This script restricts
  to TCGA samples that have BOTH miRNA-seq and mRNA-seq, aligns each modality to
  the GSE45498 training scale (reference-batch ComBat + train z-score, reusing
  notebooks 10 and 19 verbatim), and runs the full 118-feature models on the
  matched cohort.

  Both requested models are evaluated on the SAME matched samples:
    * XGBoost  (models/xgboost_model.pkl)   -- matches how the paper validates features
    * multi-network GAT (models/gat_baseline.pt)

  For each model we report multimodal (both modalities) plus each modality alone
  on the identical matched cohort, so "does adding the second modality help the
  transfer?" is answered apples-to-apples.

External task is 2-class (normal vs basal-primary); TCGA-BRCA has no metastatic
class, so 3-class predictions collapse metastatic -> primary (as in notebooks 10/19).

Run AFTER a GPU runtime is selected. First run downloads TCGA miRNA-seq from GDC.
Requires: xgboost, torch, torch_geometric, combat (and inmoose for frozen ComBat).
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

_NOTEBOOK_DIR = Path(__file__).resolve().parent
if str(_NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOK_DIR))


def _load_module(name: str, filename: str):
    path = _NOTEBOOK_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gat = _load_module('gat_baseline', '05_gat_baseline.py')
tcga_mrna = _load_module('tcga_mrna_validation', '10_tcga_classifier_validation.py')
tcga_mirna = _load_module('tcga_mirna_validation', '19_tcga_mirna_classifier_validation.py')

# Route the mRNA ComBat through notebook 19's robust implementation. Notebook 10's
# run_combat only catches TypeError, so the combat.pycombat reference-batch bug (an
# IndexError on the reference batch) crashes it. Notebook 19's version prefers
# inmoose.pycombat_norm and falls back to joint ComBat on IndexError. Both modalities
# are then harmonized by the identical method, which is also cleaner for the paper.
tcga_mrna.run_combat = tcga_mirna.run_combat

PROCESSED_DIR = gat.PROCESSED_DIR
TABLES_DIR = gat.TABLES_DIR
FIGURES_DIR = gat.FIGURES_DIR
MODELS_DIR = gat.MODELS_DIR
DEVICE = gat.DEVICE
BATCH_SIZE = gat.BATCH_SIZE
GAT_CKPT_PATH = MODELS_DIR / 'gat_baseline.pt'
BEST_PARAMS_FALLBACK = {'num_heads': 2, 'hidden_size': 32, 'dropout': 0.2, 'lr': 0.001}


@dataclass
class TransferResult:
    model: str
    modality: str
    metrics: Dict
    n_normal: int
    n_tumor: int
    n_features_used: int
    notes: str


# --- matched multimodal cohort ----------------------------------------------
def build_matched_cohort() -> Dict:
    """Align both modalities to GSE train scale and intersect to matched TCGA samples."""
    # training matrix / feature lists (from the processed GSE45498 data)
    X_train_full, _, train_ids = tcga_mrna.load_gse45498_training_data()
    gene_panel = tcga_mrna.load_gene_panel()
    mrna_features = [c for c in X_train_full.columns if c in gene_panel]
    mirna_panel = tcga_mirna.load_gse_mirna_panel()

    # mRNA side (reuse notebook 10 exactly)
    print('\n[mRNA] loading GSE + TCGA mRNA and harmonizing (reference-batch ComBat)...')
    gse_mrna_log2 = tcga_mrna.load_gse_mrna_log2()
    tcga_mrna_log2, mrna_labels = tcga_mrna.load_tcga_validation_cohort()
    tcga_mrna_z, mrna_overlap = tcga_mrna.prepare_tcga_mrna_features(
        gse_mrna_log2, tcga_mrna_log2, train_ids, mrna_features,
    )
    tcga_mrna_z.index = tcga_mrna_z.index.astype(str)
    mrna_labels.index = mrna_labels.index.astype(str)

    # miRNA side (reuse notebook 19 exactly; first run downloads from GDC)
    print('\n[miRNA] loading GSE + TCGA miRNA and harmonizing (reference-batch ComBat)...')
    gse_mirna_log2 = tcga_mirna.load_gse_log2_mirna(train_ids, mirna_panel)
    tcga_mirna_log2, mirna_labels = tcga_mirna.load_tcga_mirna_cohort()
    tcga_mirna_z, mirna_overlap = tcga_mirna.prepare_tcga_mirna_features(
        gse_mirna_log2, tcga_mirna_log2, train_ids, mirna_panel,
    )
    tcga_mirna_z.index = tcga_mirna_z.index.astype(str)
    mirna_labels.index = mirna_labels.index.astype(str)

    # matched samples (present in BOTH modalities)
    matched = sorted(set(tcga_mrna_z.index) & set(tcga_mirna_z.index))
    if len(matched) < 4:
        raise RuntimeError(
            f'Only {len(matched)} TCGA samples have both modalities; cannot do a '
            'multimodal transfer. Check that the GDC miRNA download and the mRNA '
            'matrices use the same 15-char barcode style.'
        )

    # Labels must agree across modalities on matched samples (same barcode = same class).
    lab_mrna = mrna_labels.loc[matched]
    lab_mirna = mirna_labels.loc[matched]
    agree = lab_mrna == lab_mirna
    if not agree.all():
        n_bad = int((~agree).sum())
        print(f'  WARNING: {n_bad} matched samples had disagreeing labels across '
              'modalities; dropping them.')
        matched = [s for s in matched if bool(agree.loc[s])]
    labels = mrna_labels.loc[matched].astype(int)

    n_normal = int((labels == 0).sum())
    n_tumor = int((labels == 1).sum())
    print(f'\nMatched multimodal cohort: {len(matched)} samples '
          f'({n_normal} normal + {n_tumor} basal-primary)')
    print(f'  mRNA overlap features: {len(mrna_overlap)} | '
          f'miRNA overlap features: {len(mirna_overlap)}')

    return {
        'matched': matched,
        'labels': labels,
        'tcga_mrna_z': tcga_mrna_z,
        'tcga_mirna_z': tcga_mirna_z,
        'mrna_overlap': mrna_overlap,
        'mirna_overlap': mirna_overlap,
        'train_ids': train_ids,
        'n_normal': n_normal,
        'n_tumor': n_tumor,
    }


def assemble_feature_frame(
    columns: Sequence[str],
    matched: Sequence[str],
    tcga_mrna_z: pd.DataFrame,
    tcga_mirna_z: pd.DataFrame,
    use_mrna: bool = True,
    use_mirna: bool = True,
) -> pd.DataFrame:
    """Build an (n_matched x len(columns)) matrix; unavailable/omitted features -> 0 (train mean)."""
    frame = pd.DataFrame(0.0, index=list(matched), columns=list(columns))
    for col in columns:
        if use_mirna and col in tcga_mirna_z.columns:
            frame[col] = tcga_mirna_z.loc[matched, col].values
        elif use_mrna and col in tcga_mrna_z.columns:
            frame[col] = tcga_mrna_z.loc[matched, col].values
    return frame


def _count_features(columns: Sequence[str], cohort: Dict, use_mrna: bool, use_mirna: bool) -> int:
    n = 0
    if use_mirna:
        n += len([c for c in columns if c in cohort['tcga_mirna_z'].columns])
    if use_mrna:
        n += len([c for c in columns if c in cohort['tcga_mrna_z'].columns])
    return n


# --- XGBoost transfer --------------------------------------------------------
def run_xgboost(cohort: Dict) -> List[TransferResult]:
    X_train_full, y_train, _ = tcga_mrna.load_gse45498_training_data()
    model, _ = tcga_mrna.load_or_train_full_model(X_train_full, y_train)
    columns = list(model.feature_names_in_)
    y_true = cohort['labels'].values

    results = []
    for modality, (use_mrna, use_mirna) in [
        ('multimodal', (True, True)),
        ('mrna_only', (True, False)),
        ('mirna_only', (False, True)),
    ]:
        X = assemble_feature_frame(
            columns, cohort['matched'], cohort['tcga_mrna_z'], cohort['tcga_mirna_z'],
            use_mrna=use_mrna, use_mirna=use_mirna,
        )
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)
        metrics = tcga_mrna.compute_tcga_binary_metrics(y_true, y_pred, y_prob)
        note = ('Full 118-feature XGBoost on samples with BOTH modalities present.'
                if modality == 'multimodal'
                else 'Full XGBoost on matched samples; other modality imputed at train-mean (z=0).')
        results.append(TransferResult(
            model='xgboost', modality=modality, metrics=metrics,
            n_normal=cohort['n_normal'], n_tumor=cohort['n_tumor'],
            n_features_used=_count_features(columns, cohort, use_mrna, use_mirna),
            notes=note,
        ))
    return results


# --- GAT transfer ------------------------------------------------------------
def _best_params_from_grid() -> Dict:
    grid_path = TABLES_DIR / 'gat_grid_search.csv'
    if grid_path.exists():
        grid = pd.read_csv(grid_path)
        if not grid.empty:
            top = grid.sort_values('mean_val_macro_f1', ascending=False).iloc[0]
            params = {k: top[k] for k in ('num_heads', 'hidden_size', 'dropout', 'lr') if k in top}
            if params:
                return params
    return dict(BEST_PARAMS_FALLBACK)


def train_gat_fallback() -> Tuple[torch.nn.Module, List[str]]:
    """Train a single GAT if no saved checkpoint exists (keeps the notebook self-sufficient).

    A fresh retrain is a different random seed than the manuscript's saved model, so
    external numbers may differ slightly from the paper's exact GAT. Upload the real
    models/gat_baseline.pt for numbers consistent with the paper.
    """
    print('  Training a fresh GAT (no saved checkpoint found)...')
    x_all, y_all, train_ids, test_ids = gat.load_data()
    nodes = x_all.columns.tolist()
    train_mirna_cols = pd.read_csv(PROCESSED_DIR / 'train_mirna.csv', index_col=0).columns.tolist()
    node_types = gat.infer_node_types(nodes, train_mirna_cols).to(DEVICE)
    edge_index_by_network, edge_attr_by_network = gat.load_graphs(nodes)
    edge_index_by_network = {k: v.to(DEVICE) for k, v in edge_index_by_network.items()}
    edge_attr_by_network = {k: v.to(DEVICE) for k, v in edge_attr_by_network.items()}
    dataset_by_id, train_ids, test_ids, y_numeric = gat.build_dataset(
        x_all=x_all, y_all=y_all, train_ids=train_ids, test_ids=test_ids, nodes=nodes,
    )
    dataset_by_id = {sid: data.to(DEVICE) for sid, data in dataset_by_id.items()}

    best_params = _best_params_from_grid()
    groups = gat.load_patient_groups(train_ids)
    tr_ids, val_ids = gat.patient_holdout_val_split(
        train_ids, y=y_numeric, groups=groups, n_splits=5, val_fold=0,
        random_state=gat.RANDOM_STATE,
    )
    gat.set_seed()
    model = gat.make_model(best_params, len(nodes), node_types,
                           edge_index_by_network, edge_attr_by_network)
    class_weights = gat.compute_class_weights(tr_ids, y_numeric).to(DEVICE)
    model, _, _ = gat.train_with_early_stopping(
        model=model, dataset_by_id=dataset_by_id, tr_ids=tr_ids, val_ids=val_ids,
        lr=float(best_params['lr']), class_weights=class_weights, verbose=False,
    )
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'nodes': nodes,
                'class_names': gat.CLASS_NAMES, 'best_params': best_params}, GAT_CKPT_PATH)
    print(f'  Saved freshly trained GAT to {GAT_CKPT_PATH}')
    model.eval()
    return model, nodes


def load_gat_model() -> Tuple[torch.nn.Module, List[str]]:
    if not GAT_CKPT_PATH.exists():
        print(f'{GAT_CKPT_PATH} not found -- falling back to training a GAT in-notebook.')
        return train_gat_fallback()
    ckpt = torch.load(GAT_CKPT_PATH, map_location=DEVICE, weights_only=False)
    nodes = list(ckpt['nodes'])
    best_params = ckpt['best_params']
    train_mirna_cols = pd.read_csv(PROCESSED_DIR / 'train_mirna.csv', index_col=0).columns.tolist()
    node_types = gat.infer_node_types(nodes, train_mirna_cols).to(DEVICE)
    edge_index_by_network, edge_attr_by_network = gat.load_graphs(nodes)
    edge_index_by_network = {k: v.to(DEVICE) for k, v in edge_index_by_network.items()}
    edge_attr_by_network = {k: v.to(DEVICE) for k, v in edge_attr_by_network.items()}
    model = gat.make_model(best_params, len(nodes), node_types,
                           edge_index_by_network, edge_attr_by_network)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, nodes


def gat_predict_proba(model: torch.nn.Module, X: pd.DataFrame) -> np.ndarray:
    """3-class probabilities for an (n x num_nodes) expression matrix (columns in node order)."""
    values = X.to_numpy(dtype=np.float32)
    probs = []
    with torch.no_grad():
        for start in range(0, len(values), BATCH_SIZE):
            xb = torch.tensor(values[start:start + BATCH_SIZE], device=DEVICE).unsqueeze(-1)
            logits, _ = model(xb)
            probs.append(F.softmax(logits, dim=-1).cpu().numpy())
    return np.vstack(probs)


def run_gat(cohort: Dict) -> List[TransferResult]:
    model, nodes = load_gat_model()
    y_true = cohort['labels'].values

    results = []
    for modality, (use_mrna, use_mirna) in [
        ('multimodal', (True, True)),
        ('mrna_only', (True, False)),
        ('mirna_only', (False, True)),
    ]:
        X = assemble_feature_frame(
            nodes, cohort['matched'], cohort['tcga_mrna_z'], cohort['tcga_mirna_z'],
            use_mrna=use_mrna, use_mirna=use_mirna,
        )
        y_prob = gat_predict_proba(model, X)
        y_pred = np.argmax(y_prob, axis=1)
        metrics = tcga_mrna.compute_tcga_binary_metrics(y_true, y_pred, y_prob)
        note = ('Multi-network GAT on samples with BOTH modalities present (same 3 graphs).'
                if modality == 'multimodal'
                else 'Multi-network GAT on matched samples; other modality imputed at train-mean (z=0).')
        results.append(TransferResult(
            model='gat', modality=modality, metrics=metrics,
            n_normal=cohort['n_normal'], n_tumor=cohort['n_tumor'],
            n_features_used=_count_features(nodes, cohort, use_mrna, use_mirna),
            notes=note,
        ))
    return results


# --- output ------------------------------------------------------------------
def results_to_frame(results: List[TransferResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        m = r.metrics
        cm = m['confusion_matrix']
        rows.append({
            'model': r.model,
            'modality': r.modality,
            'accuracy': m['accuracy'],
            'macro_f1': m['macro_f1'],
            'auc_normal': m['per_class_auc']['normal'],
            'auc_primary': m['per_class_auc']['primary'],
            'f1_normal': m['per_class_f1']['normal'],
            'f1_primary': m['per_class_f1']['primary'],
            'n_normal': r.n_normal,
            'n_tumor': r.n_tumor,
            'n_features_used': r.n_features_used,
            'n_metastatic_predictions': m['n_metastatic_predictions'],
            'cm_normal_normal': int(cm[0, 0]),
            'cm_normal_primary': int(cm[0, 1]),
            'cm_primary_normal': int(cm[1, 0]),
            'cm_primary_primary': int(cm[1, 1]),
            'notes': r.notes,
        })
    return pd.DataFrame(rows)


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print('=' * 72)
    print('COMBINED miRNA + mRNA TCGA-BRCA MULTIMODAL TRANSFER')
    print('=' * 72)
    print(f'Using device: {DEVICE}')

    cohort = build_matched_cohort()

    print('\n--- XGBoost transfer ---')
    xgb_results = run_xgboost(cohort)
    print('\n--- GAT transfer ---')
    gat_results = run_gat(cohort)

    results = xgb_results + gat_results
    frame = results_to_frame(results)

    print('\n' + '=' * 72)
    print('MULTIMODAL TRANSFER SUMMARY (matched cohort, normal vs basal-primary)')
    print('=' * 72)
    with pd.option_context('display.max_columns', None, 'display.width', 160):
        print(frame[['model', 'modality', 'accuracy', 'macro_f1',
                     'auc_normal', 'auc_primary', 'n_features_used']].to_string(index=False))

    out_csv = TABLES_DIR / 'tcga_multimodal_validation.csv'
    frame.to_csv(out_csv, index=False)
    print(f'\nSaved {out_csv}')
    print('\nInterpretation guide for the manuscript:')
    print('  * Compare each model\'s "multimodal" row against its "mrna_only" and')
    print('    "mirna_only" rows on the SAME matched samples. If multimodal does not')
    print('    exceed the better single modality, the honest conclusion is that the')
    print('    second modality adds no transferable signal externally.')


if __name__ == '__main__':
    main()
