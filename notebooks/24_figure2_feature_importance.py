#!/usr/bin/env python3
"""Figure 2: per-model feature importance (2×3 panels).

Panels (top row miRNA, bottom row mRNA):
  XGBoost mean |SHAP|  |  TabNet importance  |  GAT mean attention (20-seed stability)

GAT panels use results/tables/gat_attention_stability_graphml.csv (notebook 29,
top-150 networks, 20-seed).

Run from repo root:
  python notebooks/24_figure2_feature_importance.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

TABLES_DIR = Path('results/tables')
FIGURES_DIR = Path('results/figures')
PROCESSED_DIR = Path('data/processed')

OUTPUT_PATH = FIGURES_DIR / 'figure2_feature_importance.png'

XGB_PATH = TABLES_DIR / 'xgboost_importance.csv'
TABNET_PATH = TABLES_DIR / 'tabnet_importance.csv'
GAT_STABILITY_PATHS = (
    TABLES_DIR / 'gat_attention_stability_graphml.csv',   # notebook 29 (top-150 networks)
)

TOP_K = 8
NOTE_SUFFIX = '(+++ See note below)'
DEPRECATED_MIRNAS = {
    'hsa-miR-1979',
    'hsa-miR-1975',
    'hsa-miR-1308',
}

# Model colors (unchanged from prior Figure 2 styling in this project).
MODEL_COLORS = {
    'xgboost': '#4C72B0',
    'tabnet': '#DD8452',
    'gat': '#55A868',
}

X_LABELS = {
    'xgboost': 'mean |SHAP|',
    'tabnet': 'importance',
    'gat': 'mean attention',
}


def clean_feature_label(name: str) -> str:
    """Strip NanoString note suffix and surrounding whitespace."""
    text = str(name).strip()
    if NOTE_SUFFIX in text:
        text = text.replace(NOTE_SUFFIX, '').strip()
    return text


def is_deprecated_mirna(name: str) -> bool:
    cleaned = clean_feature_label(name)
    base = cleaned.split('+')[0].strip()
    return base in DEPRECATED_MIRNAS or cleaned in DEPRECATED_MIRNAS


def load_mirna_panel() -> set[str]:
    path = PROCESSED_DIR / 'train_mirna.csv'
    if not path.exists():
        return set()
    return set(pd.read_csv(path, index_col=0, nrows=0).columns)


def classify_modality(feature: str, mirna_panel: set[str]) -> str:
    label = clean_feature_label(feature)
    if label in mirna_panel:
        return 'mirna'
    if label.lower().startswith('hsa-') or label.lower().startswith('hsa-let-'):
        return 'mirna'
    return 'mrna'


def prepare_score_table(
    df: pd.DataFrame,
    feature_col: str,
    score_col: str,
    mirna_panel: set[str],
) -> pd.DataFrame:
    out = df[[feature_col, score_col]].copy()
    out.columns = ['feature', 'score']
    out['feature'] = out['feature'].map(clean_feature_label)
    out = out[~out['feature'].map(is_deprecated_mirna)]
    out['modality'] = out['feature'].map(lambda f: classify_modality(f, mirna_panel))
    out = out.sort_values('score', ascending=False)
    out = out.drop_duplicates('feature', keep='first')
    return out.reset_index(drop=True)


def top_k_by_modality(prepared: pd.DataFrame, modality: str, k: int = TOP_K) -> pd.DataFrame:
    sub = prepared[prepared['modality'] == modality].nlargest(k, 'score')
    return sub.sort_values('score', ascending=True)


def load_xgboost_tables(mirna_panel: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(XGB_PATH)
    prepared = prepare_score_table(df, 'feature', 'mean_abs_shap', mirna_panel)
    return (
        top_k_by_modality(prepared, 'mirna'),
        top_k_by_modality(prepared, 'mrna'),
    )


def load_tabnet_tables(mirna_panel: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(TABNET_PATH)
    prepared = prepare_score_table(df, 'feature', 'importance', mirna_panel)
    return (
        top_k_by_modality(prepared, 'mirna'),
        top_k_by_modality(prepared, 'mrna'),
    )


def resolve_gat_stability_path() -> Path:
    for path in GAT_STABILITY_PATHS:
        if path.exists():
            return path
    raise FileNotFoundError(
        'GAT stability CSV not found. Run notebooks/29_attention_stability_graphml.py first '
        f'(expected one of: {", ".join(str(p) for p in GAT_STABILITY_PATHS)}).'
    )


def load_gat_tables(mirna_panel: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = resolve_gat_stability_path()
    df = pd.read_csv(path)
    nodes = df[df['entity_type'] == 'node'].copy()
    if nodes.empty:
        raise ValueError(f'No node rows in {path}')

    feature_col = 'molecule' if 'molecule' in nodes.columns else 'entity_key'
    nodes['feature'] = nodes[feature_col].map(clean_feature_label)
    nodes = nodes[~nodes['feature'].map(is_deprecated_mirna)]

    if 'modality' in nodes.columns:
        nodes['modality'] = nodes['modality'].str.lower().replace({'mirna': 'mirna', 'mrna': 'mrna'})
    else:
        nodes['modality'] = nodes['feature'].map(lambda f: classify_modality(f, mirna_panel))

    nodes = nodes.rename(columns={'mean_attention': 'score'})
    nodes = nodes.sort_values('score', ascending=False).drop_duplicates('feature', keep='first')
    return (
        top_k_by_modality(nodes, 'mirna'),
        top_k_by_modality(nodes, 'mrna'),
    )


def plot_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    color: str,
    title: str,
    xlabel: str,
) -> None:
    if data.empty:
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.text(0.5, 0.5, 'No features', ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
        return

    ax.barh(data['feature'], data['score'], color=color, edgecolor='none')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=9)
    ax.tick_params(axis='y', labelsize=8)
    ax.tick_params(axis='x', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def build_figure(
    xgb_mirna: pd.DataFrame,
    xgb_mrna: pd.DataFrame,
    tab_mirna: pd.DataFrame,
    tab_mrna: pd.DataFrame,
    gat_mirna: pd.DataFrame,
    gat_mrna: pd.DataFrame,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.5), constrained_layout=True)

    plot_panel(
        axes[0, 0], xgb_mirna, MODEL_COLORS['xgboost'],
        'XGBoost — top miRNA', X_LABELS['xgboost'],
    )
    plot_panel(
        axes[0, 1], tab_mirna, MODEL_COLORS['tabnet'],
        'TabNet — top miRNA', X_LABELS['tabnet'],
    )
    plot_panel(
        axes[0, 2], gat_mirna, MODEL_COLORS['gat'],
        'GAT — top miRNA', X_LABELS['gat'],
    )
    plot_panel(
        axes[1, 0], xgb_mrna, MODEL_COLORS['xgboost'],
        'XGBoost — top mRNA', X_LABELS['xgboost'],
    )
    plot_panel(
        axes[1, 1], tab_mrna, MODEL_COLORS['tabnet'],
        'TabNet — top mRNA', X_LABELS['tabnet'],
    )
    plot_panel(
        axes[1, 2], gat_mrna, MODEL_COLORS['gat'],
        'GAT — top mRNA', X_LABELS['gat'],
    )

    return fig


def print_panel_summary(name: str, mirna_df: pd.DataFrame, mrna_df: pd.DataFrame) -> None:
    print(f'\n{name} — top miRNA:')
    for _, row in mirna_df.sort_values('score', ascending=False).iterrows():
        print(f"  {row['feature']:<30} {row['score']:.4f}")
    print(f'{name} — top mRNA:')
    for _, row in mrna_df.sort_values('score', ascending=False).iterrows():
        print(f"  {row['feature']:<30} {row['score']:.4f}")


def main() -> None:
    if not XGB_PATH.exists():
        raise FileNotFoundError(f'Missing {XGB_PATH}. Run notebooks/04_xgboost_baseline.py.')
    if not TABNET_PATH.exists():
        raise FileNotFoundError(f'Missing {TABNET_PATH}. Run notebooks/06_tabnet_baseline.py.')

    mirna_panel = load_mirna_panel()
    gat_path = resolve_gat_stability_path()
    print(f'GAT source: {gat_path.resolve()}')

    xgb_mirna, xgb_mrna = load_xgboost_tables(mirna_panel)
    tab_mirna, tab_mrna = load_tabnet_tables(mirna_panel)
    gat_mirna, gat_mrna = load_gat_tables(mirna_panel)

    for label, df in (
        ('XGBoost miRNA', xgb_mirna),
        ('XGBoost mRNA', xgb_mrna),
        ('TabNet miRNA', tab_mirna),
        ('TabNet mRNA', tab_mrna),
        ('GAT miRNA', gat_mirna),
        ('GAT mRNA', gat_mrna),
    ):
        bad = [f for f in df['feature'] if is_deprecated_mirna(f)]
        if bad:
            raise AssertionError(f'Deprecated miRNA still present in {label}: {bad}')
        note_bad = [f for f in df['feature'] if NOTE_SUFFIX in f]
        if note_bad:
            raise AssertionError(f'Unstripped note suffix in {label}: {note_bad}')

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_figure(xgb_mirna, xgb_mrna, tab_mirna, tab_mrna, gat_mirna, gat_mrna)
    out_path = OUTPUT_PATH.resolve()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print_panel_summary('XGBoost', xgb_mirna, xgb_mrna)
    print_panel_summary('TabNet', tab_mirna, tab_mrna)
    print_panel_summary('GAT (20-seed stability)', gat_mirna, gat_mrna)

    print(f'\nSaved Figure 2 -> {out_path}')


if __name__ == '__main__':
    main()
