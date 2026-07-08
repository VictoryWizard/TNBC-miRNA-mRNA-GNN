#!/usr/bin/env python3
"""Edge-stability analysis: how reproducible are the per-stage network edges?

Bootstraps each stage's TRAINING samples (resample with replacement, B times),
rebuilds the top-N edge set each time using the exact functions from
notebooks/03_network_construction.py, and measures how often each real-network
edge reappears. Low reproducibility means the network was never stable enough to
carry classification signal -- which explains why permuting/removing edges barely
changes performance (see notebook 28).

Per stage and edge type it reports:
  * mean_selection_freq : average fraction of bootstraps in which a real edge recurs
  * mean_jaccard        : mean Jaccard(bootstrap edge set, real edge set)

Outputs:
  results/tables/edge_stability.csv
  results/figures/edge_stability.png

Run from repo root:  python notebooks/31_edge_stability.py
"""

import importlib.util as _ilu
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_DIR = Path(__file__).resolve().parent
_spec = _ilu.spec_from_file_location('nb03', _DIR / '03_network_construction.py')
nb03 = _ilu.module_from_spec(_spec); _spec.loader.exec_module(nb03)

TABLES = Path('results/tables'); TABLES.mkdir(parents=True, exist_ok=True)
FIGURES = Path('results/figures'); FIGURES.mkdir(parents=True, exist_ok=True)
N_BOOT = 200
RNG = np.random.default_rng(42)
TISSUES = ('normal', 'primary', 'metastatic')


def edge_keys(edges):
    """Set of undirected edge keys from a list of edge dicts."""
    return {frozenset((e['source'], e['target'])) for e in edges}


def build_all(mirna_g, mrna_g, tissue, mirdb):
    """Return the three edge-key sets for a (sub)sample of a tissue."""
    mi = edge_keys(nb03.build_intramodal_edges(mirna_g, tissue, 'miRNA-miRNA'))
    mr = edge_keys(nb03.build_intramodal_edges(mrna_g, tissue, 'mRNA-mRNA'))
    cr = edge_keys(nb03.build_crossmodal_edges(mirna_g, mrna_g, tissue, mirdb))
    return {'miRNA-miRNA': mi, 'mRNA-mRNA': mr, 'miRNA-mRNA(cross)': cr}


def main():
    mirna, mrna, labels, mirdb = nb03.load_inputs()
    rows = []
    for tissue in TISSUES:
        samples = labels[labels == tissue].index.tolist()
        mi_g = mirna.loc[samples]; mr_g = mrna.loc[samples]
        ref = build_all(mi_g, mr_g, tissue, mirdb)
        # per-edge recurrence counts
        recur = {et: defaultdict(int) for et in ref}
        jacc = {et: [] for et in ref}
        for b in range(N_BOOT):
            idx = RNG.choice(len(samples), size=len(samples), replace=True)
            bs = [samples[i] for i in idx]
            bsets = build_all(mirna.loc[bs], mrna.loc[bs], tissue, mirdb)
            for et in ref:
                for e in bsets[et]:
                    if e in ref[et]:
                        recur[et][e] += 1
                inter = len(ref[et] & bsets[et]); union = len(ref[et] | bsets[et]) or 1
                jacc[et].append(inter / union)
        for et in ref:
            n_ref = len(ref[et]) or 1
            freqs = [recur[et][e] / N_BOOT for e in ref[et]]
            rows.append({
                'tissue': tissue, 'edge_type': et, 'n_ref_edges': len(ref[et]),
                'mean_selection_freq': float(np.mean(freqs)) if freqs else float('nan'),
                'mean_jaccard': float(np.mean(jacc[et])),
                'n_samples': len(samples),
            })
        print(f'{tissue}: ' + ', '.join(
            f"{r['edge_type']} sel={r['mean_selection_freq']:.2f}/jac={r['mean_jaccard']:.2f}"
            for r in rows if r['tissue'] == tissue))

    df = pd.DataFrame(rows)
    out = TABLES / 'edge_stability.csv'
    df.to_csv(out, index=False)
    print('\nEdge stability (1.0 = perfectly reproducible, ~chance = unstable):')
    print(df.to_string(index=False))

    # figure: mean selection frequency per tissue/edge type
    fig, ax = plt.subplots(figsize=(8, 5))
    piv = df.pivot(index='tissue', columns='edge_type', values='mean_selection_freq')
    piv.plot(kind='bar', ax=ax)
    ax.set_ylabel('Mean edge selection frequency (bootstrap)')
    ax.set_ylim(0, 1); ax.set_title('Network edge reproducibility by stage')
    ax.legend(title='edge type', fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / 'edge_stability.png', dpi=300); plt.close(fig)
    print(f'\nSaved {out} and results/figures/edge_stability.png')
    print('Low values (esp. normal, the smallest stage) explain the null ablation result.')


if __name__ == '__main__':
    main()
