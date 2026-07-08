#!/usr/bin/env python3
"""Optuna fine-tuning for the multi-network GAT TNBC classifier."""

import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

_NOTEBOOK_DIR = Path(__file__).resolve().parent
if str(_NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOK_DIR))

_BASELINE_PATH = _NOTEBOOK_DIR / '05_gat_baseline.py'
_BASELINE_SPEC = importlib.util.spec_from_file_location('gat_baseline', _BASELINE_PATH)
gat_baseline = importlib.util.module_from_spec(_BASELINE_SPEC)
assert _BASELINE_SPEC.loader is not None
sys.modules['gat_baseline'] = gat_baseline
_BASELINE_SPEC.loader.exec_module(gat_baseline)


PROCESSED_DIR = gat_baseline.PROCESSED_DIR
RESULTS_DIR = gat_baseline.RESULTS_DIR
TABLES_DIR = gat_baseline.TABLES_DIR
FIGURES_DIR = gat_baseline.FIGURES_DIR
MODELS_DIR = gat_baseline.MODELS_DIR

CLASS_NAMES = gat_baseline.CLASS_NAMES
NETWORK_NAMES = gat_baseline.NETWORK_NAMES
RANDOM_STATE = gat_baseline.RANDOM_STATE
DEVICE = gat_baseline.DEVICE

N_TRIALS = 50
N_FOLDS = 3
MODEL_SLUG = 'gat_finetuned'


def suggest_params(trial: optuna.Trial) -> Dict[str, float]:
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    return {
        'learning_rate': learning_rate,
        'lr': learning_rate,
        'hidden_size': trial.suggest_categorical('hidden_size', [16, 32, 64, 128]),
        'num_heads': trial.suggest_categorical('num_heads', [1, 2, 4, 8]),
        'dropout': trial.suggest_float('dropout', 0.05, 0.3),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
        'num_layers': trial.suggest_categorical('num_layers', [1, 2, 3]),
    }


def run_optuna_cv(
    dataset_by_id: Dict[str, gat_baseline.Data],
    train_ids: List[str],
    y_numeric: Dict[str, int],
    train_groups: pd.Series,
    num_nodes: int,
    node_types: torch.Tensor,
    edge_index_by_network: Dict[str, torch.Tensor],
    edge_attr_by_network: Dict[str, torch.Tensor],
    n_trials: int = N_TRIALS,
    n_folds: int = N_FOLDS,
) -> Tuple[Dict[str, float], pd.DataFrame, optuna.Study]:
    y_train = pd.Series([y_numeric[sid] for sid in train_ids], index=train_ids)
    group_arr = train_groups.loc[train_ids].values
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        print(f'\nTrial {trial.number + 1}/{n_trials}: {trial.params}')

        fold_scores = []
        fold_epochs = []
        for fold_i, (tr_idx, val_idx) in enumerate(sgkf.split(train_ids, y_train.values, group_arr)):
            tr_ids = [train_ids[i] for i in tr_idx]
            vl_ids = [train_ids[i] for i in val_idx]

            gat_baseline.set_seed(RANDOM_STATE + trial.number * n_folds + fold_i)
            model = gat_baseline.make_model(
                params,
                num_nodes,
                node_types,
                edge_index_by_network,
                edge_attr_by_network,
            )
            class_weights = gat_baseline.compute_class_weights(tr_ids, y_numeric).to(DEVICE)
            _, best_score, best_epoch = gat_baseline.train_with_early_stopping(
                model=model,
                dataset_by_id=dataset_by_id,
                tr_ids=tr_ids,
                val_ids=vl_ids,
                lr=float(params['learning_rate']),
                weight_decay=float(params['weight_decay']),
                class_weights=class_weights,
                verbose=False,
            )
            fold_scores.append(best_score)
            fold_epochs.append(best_epoch)
            print(f'  fold {fold_i + 1}/{n_folds} val_macro_f1={best_score:.4f} best_epoch={best_epoch}')

        mean_score = float(np.mean(fold_scores))
        std_score = float(np.std(fold_scores))
        trial.set_user_attr('fold_scores', fold_scores)
        trial.set_user_attr('mean_best_epoch', float(np.mean(fold_epochs)))
        trial.set_user_attr('std_val_macro_f1', std_score)
        print(f'  => mean_val_macro_f1={mean_score:.4f} +/- {std_score:.4f}')
        return mean_score

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE, multivariate=True)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    print(f'Optuna search: {n_trials} trials x {n_folds}-fold patient-aware CV')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    trials_df = study.trials_dataframe(attrs=('number', 'value', 'params', 'user_attrs', 'state'))
    trials_df = trials_df.sort_values('value', ascending=False).reset_index(drop=True)
    best_params = dict(study.best_trial.params)
    best_params['lr'] = best_params['learning_rate']

    return best_params, trials_df, study


def save_confusion_matrix_csv(metrics: Dict, path: Path) -> None:
    pd.DataFrame(
        metrics['confusion_matrix'],
        index=metrics['class_names'],
        columns=metrics['class_names'],
    ).to_csv(path)


def save_predictions(metrics: Dict, ids: List[str], path: Path) -> None:
    id_to_label = {idx: name for idx, name in enumerate(CLASS_NAMES)}
    rows = []
    for sample_id, true_idx, pred_idx, probs in zip(
        ids,
        metrics['y_true'],
        metrics['y_pred'],
        metrics['y_prob'],
    ):
        row = {
            'sample_id': sample_id,
            'true_label': id_to_label[int(true_idx)],
            'pred_label': id_to_label[int(pred_idx)],
        }
        for class_i, class_name in enumerate(CLASS_NAMES):
            row[f'prob_{class_name}'] = float(probs[class_i])
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def update_comparison_csv(
    path: Path,
    accuracy: float,
    macro_f1: float,
    per_class_auc: Dict[str, float],
) -> None:
    row = {
        'model': 'Fine-tuned Multi-network GAT',
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'auc_roc_normal': per_class_auc.get('normal', float('nan')),
        'auc_roc_primary': per_class_auc.get('primary', float('nan')),
        'auc_roc_metastatic': per_class_auc.get('metastatic', float('nan')),
    }

    row_df = pd.DataFrame([row])
    if path.exists():
        existing = pd.read_csv(path)
        existing = existing[existing['model'] != row['model']]
        updated = pd.concat([existing, row_df], ignore_index=True)
    else:
        updated = row_df

    columns = [
        'model',
        'accuracy',
        'macro_f1',
        'auc_roc_normal',
        'auc_roc_primary',
        'auc_roc_metastatic',
    ]
    updated.reindex(columns=columns).to_csv(path, index=False)


def print_study_summary(study: optuna.Study, trials_df: pd.DataFrame) -> None:
    print('\nOptuna study summary')
    print(f'Completed trials: {len(study.trials)}')
    print(f'Best trial: {study.best_trial.number}')
    print(f'Best mean validation macro F1: {study.best_value:.4f}')
    print('Best params:')
    for key, value in study.best_trial.params.items():
        print(f'  {key}: {value}')

    top_cols = [
        'number',
        'value',
        'params_learning_rate',
        'params_hidden_size',
        'params_num_heads',
        'params_dropout',
        'params_weight_decay',
        'params_num_layers',
        'user_attrs_std_val_macro_f1',
        'user_attrs_mean_best_epoch',
    ]
    top_cols = [col for col in top_cols if col in trials_df.columns]
    print('\nTop 10 trials:')
    print(trials_df.loc[:9, top_cols].to_string(index=False))


def main() -> None:
    gat_baseline.set_seed()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f'Using device: {DEVICE}')
    if DEVICE.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        print(f'  GPU: {props.name}')
        print(f'  VRAM: {props.total_memory / 1e9:.1f} GB')

    x_all, y_all, train_ids, test_ids = gat_baseline.load_data()
    nodes = x_all.columns.tolist()
    train_mirna_cols = pd.read_csv(PROCESSED_DIR / 'train_mirna.csv', index_col=0).columns.tolist()
    node_types = gat_baseline.infer_node_types(nodes, train_mirna_cols)
    edge_index_by_network, edge_attr_by_network = gat_baseline.load_graphs(nodes)

    dataset_by_id, train_ids, test_ids, y_numeric = gat_baseline.build_dataset(
        x_all=x_all,
        y_all=y_all,
        train_ids=train_ids,
        test_ids=test_ids,
        nodes=nodes,
    )
    dataset_by_id = {sid: data.to(DEVICE) for sid, data in dataset_by_id.items()}
    node_types = node_types.to(DEVICE)
    edge_index_by_network = {k: v.to(DEVICE) for k, v in edge_index_by_network.items()}
    edge_attr_by_network = {k: v.to(DEVICE) for k, v in edge_attr_by_network.items()}

    print(f'Feature nodes: {len(nodes)} ({len(train_mirna_cols)} miRNAs, {len(nodes) - len(train_mirna_cols)} mRNAs)')
    print(f'Train samples: {len(train_ids)}, Test samples: {len(test_ids)}')

    train_groups = gat_baseline.load_patient_groups(train_ids)
    best_params, trials_df, study = run_optuna_cv(
        dataset_by_id=dataset_by_id,
        train_ids=train_ids,
        y_numeric=y_numeric,
        train_groups=train_groups,
        num_nodes=len(nodes),
        node_types=node_types,
        edge_index_by_network=edge_index_by_network,
        edge_attr_by_network=edge_attr_by_network,
    )
    print_study_summary(study, trials_df)

    trials_path = TABLES_DIR / f'{MODEL_SLUG}_optuna_trials.csv'
    trials_df.to_csv(trials_path, index=False)

    best_params_path = TABLES_DIR / f'{MODEL_SLUG}_best_params.json'
    with best_params_path.open('w', encoding='utf-8') as f:
        json.dump({k: v for k, v in best_params.items() if k != 'lr'}, f, indent=2)

    tr_ids, val_ids = gat_baseline.patient_holdout_val_split(
        train_ids,
        y=y_numeric,
        groups=train_groups,
        n_splits=5,
        val_fold=0,
        random_state=RANDOM_STATE,
    )
    print(f'Final train/val split: {len(tr_ids)} train samples, {len(val_ids)} val samples')

    gat_baseline.set_seed()
    model = gat_baseline.make_model(
        best_params,
        len(nodes),
        node_types,
        edge_index_by_network,
        edge_attr_by_network,
    )
    class_weights = gat_baseline.compute_class_weights(tr_ids, y_numeric).to(DEVICE)
    model, final_val_macro_f1, best_epoch = gat_baseline.train_with_early_stopping(
        model=model,
        dataset_by_id=dataset_by_id,
        tr_ids=tr_ids,
        val_ids=val_ids,
        lr=float(best_params['learning_rate']),
        weight_decay=float(best_params['weight_decay']),
        class_weights=class_weights,
        verbose=True,
    )
    print(f'Final validation macro F1: {final_val_macro_f1:.4f} at epoch {best_epoch}')

    test_metrics = gat_baseline.evaluate_ids(model=model, dataset_by_id=dataset_by_id, ids=test_ids)
    gat_baseline.print_test_metrics(test_metrics)
    cm_fig, per_class_path = gat_baseline.save_test_metrics(
        test_metrics,
        MODEL_SLUG,
        FIGURES_DIR,
        TABLES_DIR,
    )

    cm_csv_path = TABLES_DIR / f'{MODEL_SLUG}_confusion_matrix.csv'
    save_confusion_matrix_csv(test_metrics, cm_csv_path)

    predictions_path = TABLES_DIR / f'{MODEL_SLUG}_test_predictions.csv'
    save_predictions(test_metrics, test_ids, predictions_path)

    attention = gat_baseline.extract_attention(model=model, dataset_by_id=dataset_by_id, test_ids=test_ids, nodes=nodes)
    attention_path = TABLES_DIR / f'{MODEL_SLUG}_attention.csv'
    attention.to_csv(attention_path, index=False)

    branch_scores = gat_baseline.extract_network_branch_scores(model=model, dataset_by_id=dataset_by_id, ids=test_ids)
    branch_scores_path = TABLES_DIR / f'{MODEL_SLUG}_network_branch_scores.csv'
    branch_scores.to_csv(branch_scores_path, index=False)

    branch_summary = gat_baseline.summarize_network_branch_scores(branch_scores)
    branch_summary_path = TABLES_DIR / f'{MODEL_SLUG}_network_branch_summary.csv'
    branch_summary.to_csv(branch_summary_path, index=False)

    model_path = MODELS_DIR / f'{MODEL_SLUG}.pt'
    torch.save({
        'state_dict': model.state_dict(),
        'nodes': nodes,
        'class_names': CLASS_NAMES,
        'network_names': NETWORK_NAMES,
        'best_params': best_params,
        'optuna_best_value': study.best_value,
        'final_val_macro_f1': final_val_macro_f1,
        'best_epoch': best_epoch,
    }, model_path)

    comparison_path = TABLES_DIR / f'{MODEL_SLUG}_model_comparison.csv'
    update_comparison_csv(
        comparison_path,
        test_metrics['accuracy'],
        test_metrics['macro_f1'],
        test_metrics['per_class_auc'],
    )

    print('Saved artifacts:')
    print(f'- {model_path}')
    print(f'- {trials_path}')
    print(f'- {best_params_path}')
    print(f'- {predictions_path}')
    print(f'- {cm_csv_path}')
    print(f'- {attention_path}')
    print(f'- {branch_scores_path}')
    print(f'- {branch_summary_path}')
    print(f'- {cm_fig}')
    print(f'- {per_class_path}')
    print(f'- {comparison_path}')


if __name__ == '__main__':
    main()
