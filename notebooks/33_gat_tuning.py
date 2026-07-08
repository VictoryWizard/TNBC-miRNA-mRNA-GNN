#!/usr/bin/env python3
"""Run #2b: full GAT tuning. (1) expanded grid search incl. num_layers (5-fold CV),
(2) 50-trial Optuna over a wider space (CV), then evaluate each CV-winner on the
held-out test set over 10 seeds. Depth is CV-selected (no test leakage)."""
import importlib.util
from pathlib import Path
import numpy as np, pandas as pd
import optuna
from sklearn.model_selection import StratifiedGroupKFold

ND = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("gat_baseline", ND / "05_gat_baseline.py")
gb = importlib.util.module_from_spec(spec); spec.loader.exec_module(gb)
optuna.logging.set_verbosity(optuna.logging.WARNING)

x_all, y_all, train_ids, test_ids = gb.load_data()
nodes = x_all.columns.tolist()
mirna_cols = pd.read_csv(gb.PROCESSED_DIR / 'train_mirna.csv', index_col=0).columns.tolist()
node_types = gb.infer_node_types(nodes, mirna_cols).to(gb.DEVICE)
eix, eattr = gb.load_graphs(nodes)
eix = {k: v.to(gb.DEVICE) for k, v in eix.items()}; eattr = {k: v.to(gb.DEVICE) for k, v in eattr.items()}
dataset_by_id, train_ids, test_ids, y_numeric = gb.build_dataset(x_all, y_all, train_ids, test_ids, nodes)
dataset_by_id = {sid: d.to(gb.DEVICE) for sid, d in dataset_by_id.items()}
train_groups = gb.load_patient_groups(train_ids)

# ---- 1) expanded grid search (CV, includes num_layers) ----
gb.PARAM_GRID = {'num_heads': [2, 4], 'hidden_size': [32, 64], 'dropout': [0.1, 0.2], 'lr': [0.001], 'num_layers': [1, 2]}
best_grid, grid_df = gb.grid_search_cv(dataset_by_id, train_ids, y_numeric, train_groups,
                                       len(nodes), node_types, eix, eattr, n_folds=5)
print("\nGRID best:", best_grid)

# ---- 2) Optuna (CV) ----
def cv_score(params, n_folds=3):
    yv = pd.Series([y_numeric[s] for s in train_ids], index=train_ids)
    ga = train_groups.loc[train_ids].values
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=gb.RANDOM_STATE)
    sc = []
    for fi, (tr, vl) in enumerate(sgkf.split(train_ids, yv.values, ga)):
        tri = [train_ids[i] for i in tr]; vli = [train_ids[i] for i in vl]
        gb.set_seed(gb.RANDOM_STATE + fi)
        model = gb.make_model(params, len(nodes), node_types, eix, eattr)
        cw = gb.compute_class_weights(tri, y_numeric).to(gb.DEVICE)
        _, bs, _ = gb.train_with_early_stopping(model, dataset_by_id, tri, vli, lr=params['lr'],
                                                class_weights=cw, verbose=False)
        sc.append(bs)
    return float(np.mean(sc))

def objective(trial):
    p = {'num_heads': trial.suggest_categorical('num_heads', [1, 2, 4, 8]),
         'hidden_size': trial.suggest_categorical('hidden_size', [16, 32, 64, 128]),
         'dropout': trial.suggest_float('dropout', 0.0, 0.5),
         'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
         'num_layers': trial.suggest_int('num_layers', 1, 3)}
    return cv_score(p)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
print("OPTUNA best:", study.best_params, "cv_macroF1", round(study.best_value, 4))

# ---- 3) evaluate each CV-winner on TEST over seeds ----
def test_dist(params, n=10):
    tr_ids, val_ids = gb.patient_holdout_val_split(train_ids, y=y_numeric, groups=train_groups,
                                                   n_splits=5, val_fold=0, random_state=gb.RANDOM_STATE)
    f = []
    for s in range(n):
        gb.set_seed(s)
        model = gb.make_model(params, len(nodes), node_types, eix, eattr)
        cw = gb.compute_class_weights(tr_ids, y_numeric).to(gb.DEVICE)
        model, _, _ = gb.train_with_early_stopping(model, dataset_by_id, tr_ids, val_ids, lr=params['lr'],
                                                   class_weights=cw, verbose=False)
        f.append(gb.evaluate_ids(model, dataset_by_id, test_ids)['macro_f1'])
    return np.array(f)

out_rows = []
for name, raw in [('grid', dict(best_grid)), ('optuna', dict(study.best_params))]:
    p = {'num_heads': int(raw['num_heads']), 'hidden_size': int(raw['hidden_size']),
         'dropout': float(raw['dropout']), 'lr': float(raw['lr']), 'num_layers': int(raw['num_layers'])}
    f = test_dist(p, 10)
    print(f"\n{name.upper()} winner {p}\n  -> TEST macroF1 mean {f.mean():.4f}  SD {f.std(ddof=1):.4f}  (10 seeds)")
    out_rows.append({'search': name, **p, 'test_macro_f1_mean': round(f.mean(), 4), 'test_macro_f1_sd': round(f.std(ddof=1), 4)})

res = pd.DataFrame(out_rows)
outp = ND.parent / 'results' / 'tables' / 'gat_tuning_results.csv'
outp.parent.mkdir(parents=True, exist_ok=True); res.to_csv(outp, index=False)
print("\n" + res.to_string(index=False))
print("Reference: XGBoost ~0.939; untuned 1-layer 20-seed 0.911, 2-layer 0.897")
print(f"saved -> {outp}")
