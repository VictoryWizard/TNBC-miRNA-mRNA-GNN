#!/usr/bin/env python3
"""Plain-text report: GAT attention stability vs XGBoost SHAP overlap."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

TABLES_DIR = Path('results/tables')
STABILITY_PATH = TABLES_DIR / 'gat_attention_stability_graphml.csv'
XGB_PATH = TABLES_DIR / 'xgboost_importance.csv'

NOTE_SUFFIX = '(+++ See note below)'
DEPRECATED_MIRNAS = {'hsa-miR-1975', 'hsa-miR-1308', 'hsa-miR-1979'}


def clean_feature(name: str) -> str:
    text = str(name).strip()
    if NOTE_SUFFIX in text:
        text = text.replace(NOTE_SUFFIX, '').strip()
    return text


def normalize_modality(value: str) -> str:
    """Map common modality spelling variants to canonical 'mirna' / 'mrna'."""
    v = str(value).strip().lower()
    if v in ('mirna', 'mir', 'mi-rna', 'micrornas'):
        return 'mirna'
    if v in ('mrna', 'mr', 'messenger'):
        return 'mrna'
    return v


def edge_type_label(edge_type: str) -> str:
    et = str(edge_type)
    if et == 'miRNA-miRNA':
        return 'miRNA–miRNA'
    if et == 'mRNA-mRNA':
        return 'mRNA–mRNA'
    if et == 'miRNA-mRNA':
        return 'cross-modal'
    return et


def flag_deprecated(source: str, target: str) -> str:
    hits = []
    for node in (clean_feature(source), clean_feature(target)):
        base = node.split('+')[0].strip()
        if base in DEPRECATED_MIRNAS or node in DEPRECATED_MIRNAS:
            hits.append(node)
    if not hits:
        return ''
    return f'  [FLAG: deprecated miRNA endpoint: {", ".join(sorted(set(hits)))}]'


def rank_series(df: pd.DataFrame, score_col: str, feature_col: str) -> pd.DataFrame:
    out = df[[feature_col, score_col]].copy()
    out[feature_col] = out[feature_col].map(clean_feature)
    out = out.sort_values(score_col, ascending=False).drop_duplicates(feature_col, keep='first')
    out['rank'] = range(1, len(out) + 1)
    return out.reset_index(drop=True)


def find_rank(ranked: pd.DataFrame, feature_col: str, query: str) -> str:
    q = clean_feature(query)
    match = ranked[ranked[feature_col].str.lower() == q.lower()]
    if match.empty:
        # partial match for miR-542-3p style
        partial = ranked[ranked[feature_col].str.contains(re.escape(q.split('/')[-1]), case=False, regex=True)]
        if len(partial) == 1:
            row = partial.iloc[0]
            return f"{int(row['rank'])} ({row[feature_col]})"
        return 'not found'
    return str(int(match.iloc[0]['rank']))


def print_node_table(title: str, df: pd.DataFrame) -> None:
    print(title)
    if df.empty:
        print('  (none)')
        return
    for _, row in df.iterrows():
        print(
            f"  {row['feature']}\t"
            f"mean_attention={row['mean_attention']:.6f}\t"
            f"CV={row['cv_attention']:.6f}"
        )


def main() -> None:
    stability = pd.read_csv(STABILITY_PATH)
    xgb = pd.read_csv(XGB_PATH)

    # --- column names used ---
    print('COLUMN NAMES USED')
    print(f'  stability: entity_type, molecule, modality, mean_attention, cv_attention')
    print(f'  xgboost:   feature, mean_abs_shap')
    print()

    nodes = stability[stability['entity_type'] == 'node'].copy()
    nodes['feature'] = nodes['molecule'].map(clean_feature)
    nodes['modality_norm'] = nodes['modality'].map(normalize_modality)

    edges = stability[stability['entity_type'] == 'edge'].copy()

    xgb_prep = xgb.copy()
    xgb_prep['feature'] = xgb_prep['feature'].map(clean_feature)
    xgb_prep = xgb_prep[~xgb_prep['feature'].isin(DEPRECATED_MIRNAS)]
    xgb_prep = xgb_prep.sort_values('mean_abs_shap', ascending=False).drop_duplicates('feature', keep='first')

    # --- 1. Top 15 nodes by mean attention per modality ---
    print('=' * 72)
    print('NODES: TOP 15 BY MEAN ATTENTION (mRNA)')
    top_mrna = nodes[nodes['modality_norm'] == 'mrna'].nlargest(15, 'mean_attention')
    print_node_table('', top_mrna.rename(columns={'feature': 'feature'}))

    print()
    print('NODES: TOP 15 BY MEAN ATTENTION (miRNA)')
    top_mirna = nodes[nodes['modality_norm'] == 'mirna'].nlargest(15, 'mean_attention')
    print_node_table('', top_mirna)

    # --- 2. Steadiest nodes (lowest CV) ---
    print()
    print('=' * 72)
    print('STEADY NODES: 5 LOWEST CV')
    steady = nodes.nsmallest(5, 'cv_attention')
    for _, row in steady.iterrows():
        print(f"  {row['feature']}\tCV={row['cv_attention']:.6f}")

    pthlh_node = nodes[nodes['feature'].str.upper() == 'PTHLH']
    print()
    if pthlh_node.empty:
        print('PTHLH node CV: not found')
    else:
        cv = float(pthlh_node.iloc[0]['cv_attention'])
        print(f'PTHLH node CV: {cv:.6f}')

    # --- 3. Overlap: global top 15 attention nodes vs global top 15 SHAP ---
    print()
    print('=' * 72)
    print('SHAP vs ATTENTION OVERLAP (global top 15 each)')
    att_top15 = set(nodes.nlargest(15, 'mean_attention')['feature'])
    shap_top15 = set(xgb_prep.head(15)['feature'])
    overlap = sorted(att_top15 & shap_top15)
    if overlap:
        for feat in overlap:
            print(f'  {feat}')
    else:
        print('  (none)')

    # --- 4. Specific ranks ---
    print()
    print('=' * 72)
    print('RANKS')
    xgb_ranked = rank_series(xgb_prep, 'mean_abs_shap', 'feature')
    att_ranked = rank_series(nodes, 'mean_attention', 'feature')

    print(f"  PTHLH — XGBoost SHAP rank: {find_rank(xgb_ranked, 'feature', 'PTHLH')}")
    print(f"  miR-542-3p — GAT attention rank: {find_rank(att_ranked, 'feature', 'hsa-miR-542-3p')}")
    print(f"  miR-421 — XGBoost SHAP rank: {find_rank(xgb_ranked, 'feature', 'hsa-miR-421')}")
    print(f"  miR-421 — GAT attention rank: {find_rank(att_ranked, 'feature', 'hsa-miR-421')}")

    # --- 5. Steadiest edges (lowest CV) ---
    print()
    print('=' * 72)
    print('EDGES: 6 LOWEST CV (steadiest)')
    steady_edges = edges.nsmallest(6, 'cv_attention')
    for _, row in steady_edges.iterrows():
        flag = flag_deprecated(row['source'], row['target'])
        print(
            f"  {row['network']}: {row['source']} — {row['target']}\t"
            f"type={edge_type_label(row['edge_type'])}\t"
            f"mean_attention={row['mean_attention']:.6f}\t"
            f"CV={row['cv_attention']:.6f}{flag}"
        )


if __name__ == '__main__':
    main()
