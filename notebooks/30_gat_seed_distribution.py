#!/usr/bin/env python3
"""Run #1: GAT test macro-F1 distribution over N seeds (fixed best params,
fixed held-out split). Answers whether 0.96 is typical or a lucky seed, and
gives the mean +/- SD / SEM to report consistently. XGBoost is ~deterministic
at this setting, so its single value (~0.939) is the comparator."""
import importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd

ND = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("gat_baseline", ND / "05_gat_baseline.py")
gb = importlib.util.module_from_spec(spec); spec.loader.exec_module(gb)

N_SEEDS = 20
BEST = {'num_heads': 2, 'hidden_size': 32, 'dropout': 0.2, 'lr': 0.001, 'num_layers': 2}

print(f"Device: {gb.DEVICE}")
x_all, y_all, train_ids, test_ids = gb.load_data()
nodes = x_all.columns.tolist()
mirna_cols = pd.read_csv(gb.PROCESSED_DIR / 'train_mirna.csv', index_col=0).columns.tolist()
node_types = gb.infer_node_types(nodes, mirna_cols)
eix, eattr = gb.load_graphs(nodes)
dataset_by_id, train_ids, test_ids, y_numeric = gb.build_dataset(x_all, y_all, train_ids, test_ids, nodes)
dataset_by_id = {sid: d.to(gb.DEVICE) for sid, d in dataset_by_id.items()}
node_types = node_types.to(gb.DEVICE)
eix = {k: v.to(gb.DEVICE) for k, v in eix.items()}
eattr = {k: v.to(gb.DEVICE) for k, v in eattr.items()}
train_groups = gb.load_patient_groups(train_ids)
tr_ids, val_ids = gb.patient_holdout_val_split(train_ids, y=y_numeric, groups=train_groups,
                                               n_splits=5, val_fold=0, random_state=gb.RANDOM_STATE)
print(f"Train {len(tr_ids)} / Val {len(val_ids)} / Test {len(test_ids)} | nodes {len(nodes)}")

rows = []
for s in range(N_SEEDS):
    gb.set_seed(s)
    model = gb.make_model(BEST, len(nodes), node_types, eix, eattr)
    cw = gb.compute_class_weights(tr_ids, y_numeric).to(gb.DEVICE)
    model, _, _ = gb.train_with_early_stopping(model, dataset_by_id, tr_ids, val_ids,
                                               lr=BEST['lr'], class_weights=cw, verbose=False)
    m = gb.evaluate_ids(model, dataset_by_id, test_ids)
    print(f"seed {s:2d}: acc={m['accuracy']:.4f}  macroF1={m['macro_f1']:.4f}")
    rows.append({'seed': s, 'accuracy': m['accuracy'], 'macro_f1': m['macro_f1']})

df = pd.DataFrame(rows)
f1 = df['macro_f1'].to_numpy(); ac = df['accuracy'].to_numpy()
out = ND.parent / 'results' / 'tables' / 'gat_seed_distribution.csv'
out.parent.mkdir(parents=True, exist_ok=True); df.to_csv(out, index=False)
print("\n================ GAT over", N_SEEDS, "seeds ================")
print(f"macro-F1: mean {f1.mean():.4f}  SD {f1.std(ddof=1):.4f}  SEM {f1.std(ddof=1)/np.sqrt(N_SEEDS):.4f}  min {f1.min():.4f}  max {f1.max():.4f}")
print(f"accuracy: mean {ac.mean():.4f}  SD {ac.std(ddof=1):.4f}  min {ac.min():.4f}  max {ac.max():.4f}")
print(f"XGBoost comparator (deterministic at fixed params): macro-F1 ~0.939")
print(f"saved -> {out}")
