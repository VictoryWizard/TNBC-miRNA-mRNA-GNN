#!/usr/bin/env python3
"""Edge-level GAT attention analysis (Section 2.7).

Recomputes, on the CURRENT build, the edge-level attention claims:
  - cross-modal share of the top-25 highest-attention edges vs all edges
  - Spearman(mean attention, across-seed CV)   [attention vs seed-stability]
  - Spearman(mean attention, endpoint XGBoost-SHAP importance)
  - the single highest-attention edge (mean, CV)

Inputs (produced upstream):
  results/tables/gat_attention_stability_graphml_edges.csv   (nb29, 20-seed)
  results/tables/xgboost_importance.csv                      (nb04)
  data/processed/train_mirna.csv                             (nb01)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

T = Path('results/tables')


def main():
    edges = pd.read_csv(T / 'gat_attention_stability_graphml_edges.csv')
    mirna = set(pd.read_csv('data/processed/train_mirna.csv', index_col=0, nrows=0).columns)
    is_mi = lambda n: n in mirna

    def etype(u, v):
        a, b = is_mi(u), is_mi(v)
        return 'cross' if a != b else ('mi-mi' if a else 'mr-mr')

    edges['etype'] = [etype(u, v) for u, v in zip(edges.node_u, edges.node_v)]

    top25 = edges.nlargest(25, 'mean_attention')
    pct_top = 100 * (top25.etype == 'cross').mean()
    pct_all = 100 * (edges.etype == 'cross').mean()

    rho_cv, _ = spearmanr(edges.mean_attention, edges.cv_attention)

    imp = pd.read_csv(T / 'xgboost_importance.csv')
    impd = dict(zip(imp.iloc[:, 0], imp.iloc[:, 1]))
    endp = [max(impd.get(u, 0.0), impd.get(v, 0.0)) for u, v in zip(edges.node_u, edges.node_v)]
    rho_imp, _ = spearmanr(edges.mean_attention, endp)

    top = edges.loc[edges.mean_attention.idxmax()]

    print('=' * 60)
    print('EDGE-LEVEL ATTENTION  (Section 2.7)')
    print('=' * 60)
    print(f'edges total: {len(edges)}')
    print(f'cross-modal share:  top-25 = {pct_top:.0f}%   |   all edges = {pct_all:.0f}%')
    print(f'Spearman(attention, seed-CV)           = {rho_cv:+.2f}   (~0 -> unrelated to seed-stability)')
    print(f'Spearman(attention, endpoint XGB-SHAP) = {rho_imp:+.2f}')
    print(f'highest-attention edge: {top.node_u}-{top.node_v}  '
          f'mean={top.mean_attention:.3f}  CV={top.cv_attention:.3f}')
    print('=' * 60)


if __name__ == '__main__':
    main()
