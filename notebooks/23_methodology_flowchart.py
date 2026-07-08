#!/usr/bin/env python3
"""Figure 5 - analysis pipeline flowchart (matplotlib; no external Graphviz).

Horizontal preprocessing chain -> the three classifiers (highlighted hub) ->
the three downstream analyses. Numbers reflect the 118-feature, patient-clean
build (184/51 split, 86 miRNA + 32 mRNA, 118 nodes).

Run from repo root:
  python notebooks/23_methodology_flowchart.py
Output: results/figures/methodology_flowchart.png
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIGURES_DIR = Path('results/figures')

TOP_FILL, TOP_EDGE = '#dbe8f7', '#5a9bd4'
CLS_FILL, CLS_EDGE = '#fdf4ea', '#e2933f'
BR_FILL, BR_EDGE = '#efefef', '#9a9a9a'
TXT, ARROW = '#232a31', '#6b6b6b'


def _box(ax, cx, cy, w, h, title, detail, fill, edge, lw=1.3, tsz=9.5, dsz=8.4):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10", linewidth=lw,
        edgecolor=edge, facecolor=fill))
    if detail:
        ax.text(cx, cy + h * 0.20, title, ha='center', va='center',
                fontsize=tsz, fontweight='bold', color=TXT)
        ax.text(cx, cy - h * 0.22, detail, ha='center', va='center',
                fontsize=dsz, color=TXT, linespacing=1.25)
    else:
        ax.text(cx, cy, title, ha='center', va='center',
                fontsize=tsz, fontweight='bold', color=TXT)


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
        mutation_scale=13, lw=1.2, color=ARROW, shrinkA=2, shrinkB=2))


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12.6, 5.4))
    ax.set_xlim(0, 14); ax.set_ylim(0, 6); ax.axis('off')

    ty, tw, th = 5.05, 3.0, 1.15
    tx = [1.85, 5.28, 8.71, 12.14]
    top = [
        ("GSE45498", "235 samples · 173 patients\nmatched miRNA + mRNA"),
        ("Patient-grouped split", "184 train / 51 test"),
        ("Differential expression", "118 molecules\n(86 miRNA, 32 mRNA)"),
        ("Three tissue networks", "top-150 edges / type\n(normal, primary, met)"),
    ]
    for (t, d), x in zip(top, tx):
        _box(ax, x, ty, tw, th, t, d, TOP_FILL, TOP_EDGE)
    for i in range(3):
        _arrow(ax, tx[i] + tw / 2, ty, tx[i + 1] - tw / 2, ty)

    cx, cy, cw, ch = 7.0, 3.0, 4.6, 1.15
    _box(ax, cx, cy, cw, ch, "Three classifiers",
         "XGBoost · multi-network GAT · TabNet", CLS_FILL, CLS_EDGE, lw=1.8, tsz=10)
    _arrow(ax, tx[3], ty - th / 2, cx + cw / 2 - 0.3, cy + ch / 2 - 0.15)

    by, bw, bh = 0.78, 3.55, 1.2
    bx = [2.15, 7.0, 11.85]
    br = [
        ("Network ablation", "edges add nothing\n(p = 0.426)"),
        ("Feature importance", "SHAP vs. attention diverge"),
        ("TCGA-BRCA validation", "mRNA AUROC 0.88\nmiRNA AUROC 0.99"),
    ]
    for (t, d), x in zip(br, bx):
        _box(ax, x, by, bw, bh, t, d, BR_FILL, BR_EDGE)
    for x in bx:
        _arrow(ax, cx, cy - ch / 2, x, by + bh / 2)

    out = FIGURES_DIR / 'methodology_flowchart.png'
    fig.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.08)
    print(f"Saved {out}")


if __name__ == '__main__':
    main()
