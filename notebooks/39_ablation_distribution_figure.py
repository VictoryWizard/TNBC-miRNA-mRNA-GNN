#!/usr/bin/env python3
"""Network-ablation permutation-null figure (Figure 2).

Histogram of the 1,000-permutation null distribution of held-out macro F1, with
the real network, the edge-removed network, and the null mean marked as reference
lines and a boxed key. No on-figure title and few x-ticks (title/p belong in the
caption, per reviewer note); the empirical p is computed and printed for the caption.

Reads the permutation values from the edge-ablation run
(network_ablation_graphml_permutations.csv) and the condition means from the
ablation summary (network_ablation_graphml.csv); falls back to the paper's
hidden-32 constants if the summary is absent.

Overwrites results/figures/network_ablation_graphml.png (the Figure 2 file), so
run this after 28_network_ablation_graphml.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

TABLES = Path('results/tables')
FIGS = Path('results/figures')

# Fallbacks (paper hidden-32 build) if the summary table is unavailable.
REAL_FALLBACK = 0.9358
EDGE_REMOVED_FALLBACK = 0.9395

PERM_CANDIDATES = [
    TABLES / 'network_ablation_graphml_permutations.csv',
    Path('network_ablation_graphml_permutations.csv'),
]
SUMMARY_CANDIDATES = [
    TABLES / 'network_ablation_graphml.csv',
    Path('network_ablation_graphml.csv'),
]


def load_permutations() -> np.ndarray:
    for p in PERM_CANDIDATES:
        if not p.exists():
            continue
        df = pd.read_csv(p)
        print(f'loaded {p}  columns={list(df.columns)}  rows={len(df)}')
        for name in ('macro_f1', 'permuted_macro_f1', 'permuted', 'value', 'score'):
            if name in df.columns and df[name].notna().sum() >= 50:
                return df[name].dropna().to_numpy(dtype=float)
    raise FileNotFoundError(
        'No permutation CSV found. Expected network_ablation_graphml_permutations.csv '
        'in results/tables/ (produced by 28_network_ablation_graphml.py).')


def load_conditions() -> tuple[float, float]:
    """Return (real, edges_removed) macro-F1 from the ablation summary, else fallbacks."""
    for p in SUMMARY_CANDIDATES:
        if not p.exists():
            continue
        df = pd.read_csv(p)
        cols = {c.lower(): c for c in df.columns}
        cond_col = next((cols[c] for c in cols if c in ('condition', 'setting', 'name')), None)
        val_col = next((cols[c] for c in cols
                        if c in ('macro_f1_mean', 'macro_f1', 'mean', 'value')), None)
        if cond_col and val_col:
            m = {str(r[cond_col]).lower(): float(r[val_col]) for _, r in df.iterrows()}
            real = m.get('real', m.get('real_network', REAL_FALLBACK))
            edges = m.get('no_edges', m.get('edges_removed', EDGE_REMOVED_FALLBACK))
            return real, edges
    return REAL_FALLBACK, EDGE_REMOVED_FALLBACK


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    perms = load_permutations()
    real, no_edges = load_conditions()
    null_mean = float(perms.mean())
    ge = int((perms >= real).sum())
    p_emp = (1 + ge) / (1 + len(perms))
    print(f'{len(perms)} permuted values; null mean {null_mean:.3f} +/- {perms.std():.3f}; '
          f'real {real:.3f}; edges-removed {no_edges:.3f}; {ge} >= real; '
          f'empirical p = {p_emp:.3f}')

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12})
    fig, ax = plt.subplots(figsize=(7.6, 4.7))
    ax.hist(perms, bins=30, color='#c9c9c9', edgecolor='white', linewidth=0.7, zorder=1)
    ax.axvline(real, color='#c0392b', lw=2.6, zorder=4)
    ax.axvline(no_edges, color='#2c7fb8', lw=2.2, ls='--', zorder=4)
    ax.axvline(null_mean, color='#7a7a7a', lw=1.5, ls=':', zorder=3)

    ax.set_xlabel('Held-out macro F1', fontsize=12.5, labelpad=8)
    ax.set_ylabel('Number of permutations', fontsize=12.5, labelpad=8)
    ax.set_xticks([0.80, 0.85, 0.90, 0.95, 1.00])
    ax.set_xlim(0.795, 1.005)

    handles = [
        Line2D([0], [0], color='#c0392b', lw=2.6, label=f'Real network   {real:.3f}'),
        Line2D([0], [0], color='#2c7fb8', lw=2.2, ls='--', label=f'Edges removed   {no_edges:.3f}'),
        Line2D([0], [0], color='#7a7a7a', lw=1.5, ls=':', label=f'Null mean   {null_mean:.3f}'),
        Patch(facecolor='#c9c9c9', edgecolor='white', label='Permuted null'),
    ]
    leg = ax.legend(handles=handles, title=f'n = {len(perms):,} permutations',
                    loc='upper left', frameon=True, fontsize=10.5, title_fontsize=10.5,
                    borderpad=0.8, labelspacing=0.6, handlelength=1.9)
    leg.get_frame().set_edgecolor('#cccccc')
    leg.get_frame().set_linewidth(0.8)
    leg.get_frame().set_facecolor('white')
    leg._legend_box.align = 'left'

    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.spines['left'].set_color('#999')
    ax.spines['bottom'].set_color('#999')
    ax.tick_params(colors='#444', length=4)

    fig.tight_layout()
    out = FIGS / 'network_ablation_graphml.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print('saved', out, f'(empirical p = {p_emp:.3f})')


if __name__ == '__main__':
    main()
