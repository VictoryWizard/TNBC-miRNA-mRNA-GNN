#!/usr/bin/env python3
"""What does the GAT surface that XGBoost does not (and vice versa)?

With 'consensus' dropped, the point of comparing the structure-aware GAT against
the structure-blind XGBoost is the DIFFERENCES: which features each model uniquely
elevates. We summarize agreement with a single, directly interpretable measure --
the Jaccard overlap of the two models' top-K features (per modality) -- and list
the molecules each model promotes that the other buries.

Why Jaccard of the top-K (and not Spearman/Kendall/RBO): the question is simply
"do the models pick the same top biomarkers?", which is exactly set overlap of the
top features. It needs no distributional assumptions, is interpretable to a
clinical reader ("they share N of their top K"), and ignores the long tail of
unimportant features that whole-list rank correlations would fold in.

Inputs:
  results/tables/xgboost_importance.csv              (feature, mean_abs_shap)
  results/tables/gat_attention_stability_graphml.csv (node, mean_attention, modality)
Outputs:
  results/tables/gat_vs_xgboost_agreement.csv      (Jaccard per modality)
  results/tables/gat_vs_xgboost_differences.csv    (per-feature ranks + flags)
  results/figures/gat_vs_xgboost_rank_scatter.png

Run from repo root:  python notebooks/32_gat_vs_xgboost_differences.py
"""

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

TABLES = Path('results/tables')
FIGURES = Path('results/figures'); FIGURES.mkdir(parents=True, exist_ok=True)
TOP_K = 25  # standardized top-25 across analyses


def modality_of(name):
    n = str(name).lower()
    return 'miRNA' if n.startswith('hsa-') or n.startswith('mir') else 'mRNA'


def jaccard_at_k(list_a, list_b, k):
    a, b = set(list_a[:k]), set(list_b[:k])
    return (len(a & b) / len(a | b)) if (a or b) else float('nan')


def main():
    xgb = pd.read_csv(TABLES / 'xgboost_importance.csv').rename(
        columns={'feature': 'molecule', 'mean_abs_shap': 'xgb_importance'})
    gat = pd.read_csv(TABLES / 'gat_attention_stability_graphml.csv')
    # 29 writes both 'node' and 'molecule' (a copy); avoid creating a duplicate
    # 'molecule' column when normalizing.
    if 'molecule' in gat.columns and 'node' in gat.columns:
        gat = gat.drop(columns=['node'])
    elif 'molecule' not in gat.columns:
        gat = gat.rename(columns={'node': 'molecule'})
    gat = gat.rename(columns={'mean_attention': 'gat_importance'})
    keep = ['molecule', 'gat_importance'] + (['modality'] if 'modality' in gat.columns else [])
    gat = gat[keep]

    df = pd.merge(xgb[['molecule', 'xgb_importance']], gat, on='molecule', how='outer')
    if 'modality' not in df.columns:
        df['modality'] = df['molecule'].map(modality_of)
    df['modality'] = df['modality'].fillna(df['molecule'].map(modality_of))
    df['xgb_importance'] = df['xgb_importance'].fillna(0.0)
    df['gat_importance'] = df['gat_importance'].fillna(0.0)
    df['xgb_rank'] = df['xgb_importance'].rank(ascending=False, method='min').astype(int)
    df['gat_rank'] = df['gat_importance'].rank(ascending=False, method='min').astype(int)

    # --- single agreement metric: top-K Jaccard, overall and per modality ---
    rows = []
    for mod in ['all', 'miRNA', 'mRNA']:
        sub = df if mod == 'all' else df[df['modality'] == mod]
        xl = sub.sort_values('xgb_importance', ascending=False)['molecule'].tolist()
        gl = sub.sort_values('gat_importance', ascending=False)['molecule'].tolist()
        shared = sorted(set(xl[:TOP_K]) & set(gl[:TOP_K]))
        rows.append({'scope': mod, 'n_features': len(sub), 'top_k': TOP_K,
                     f'jaccard_top{TOP_K}': round(jaccard_at_k(xl, gl, TOP_K), 3),
                     'n_shared_topk': len(shared), 'shared': '; '.join(shared)})
    agree = pd.DataFrame(rows)
    agree.to_csv(TABLES / 'gat_vs_xgboost_agreement.csv', index=False)
    print(f'Top-{TOP_K} agreement (Jaccard) between GAT attention and XGBoost SHAP:')
    print(agree[['scope', 'n_features', f'jaccard_top{TOP_K}', 'n_shared_topk', 'shared']].to_string(index=False))
    print('  (low Jaccard => the models disagree on the top features = GAT offers a different view)')

    gat_unique = df[(df['gat_rank'] <= TOP_K) & (df['xgb_rank'] > TOP_K)].sort_values('gat_rank')
    xgb_unique = df[(df['xgb_rank'] <= TOP_K) & (df['gat_rank'] > TOP_K)].sort_values('xgb_rank')
    print(f'\nGAT-elevated (top-{TOP_K} in GAT, not XGBoost):')
    print(gat_unique[['molecule', 'modality', 'gat_rank', 'xgb_rank']].to_string(index=False))
    print(f'\nXGBoost-elevated (top-{TOP_K} in XGBoost, not GAT):')
    print(xgb_unique[['molecule', 'modality', 'xgb_rank', 'gat_rank']].to_string(index=False))

    df['gat_elevated'] = (df['gat_rank'] <= TOP_K) & (df['xgb_rank'] > TOP_K)
    df['xgb_elevated'] = (df['xgb_rank'] <= TOP_K) & (df['gat_rank'] > TOP_K)
    df.sort_values('gat_rank').to_csv(TABLES / 'gat_vs_xgboost_differences.csv', index=False)

    fig, ax = plt.subplots(figsize=(6, 6))
    colors = df['modality'].map({'miRNA': 'tab:blue', 'mRNA': 'tab:orange'})
    ax.scatter(df['xgb_rank'], df['gat_rank'], c=colors, alpha=0.6, s=20)
    lim = max(df['xgb_rank'].max(), df['gat_rank'].max())
    ax.plot([1, lim], [1, lim], 'k--', lw=0.8, alpha=0.5)
    ax.axvline(TOP_K, color='gray', ls=':', lw=0.7); ax.axhline(TOP_K, color='gray', ls=':', lw=0.7)
    jac = float(agree.loc[agree.scope == 'all', f'jaccard_top{TOP_K}'].iloc[0])
    ax.set_xlabel('XGBoost SHAP rank'); ax.set_ylabel('GAT attention rank')
    ax.set_title(f'Importance ranks (top-{TOP_K} Jaccard = {jac:.2f})')
    fig.tight_layout(); fig.savefig(FIGURES / 'gat_vs_xgboost_rank_scatter.png', dpi=300); plt.close(fig)
    print(f'\nSaved gat_vs_xgboost_agreement.csv, gat_vs_xgboost_differences.csv, and the rank scatter.')


if __name__ == '__main__':
    main()
