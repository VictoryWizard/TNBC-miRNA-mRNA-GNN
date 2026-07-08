#!/usr/bin/env python3
"""Network statistics summary for tissue-specific regulatory graphs (notebook 17).

Computes descriptive graph metrics from saved GraphML networks without retraining
models or rebuilding the full pipeline.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

RESULTS_DIR = Path('results')
TABLES_DIR = Path('results/tables')
FIGURES_DIR = Path('results/figures')
PROCESSED_DIR = Path('data/processed')

TISSUE_STAGES = ['normal', 'primary', 'metastatic']
EDGE_TYPES = ['miRNA-miRNA', 'mRNA-mRNA', 'miRNA-mRNA']


def load_reference_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load pair and expression tables for context (not used to rebuild graphs)."""
    pairs = pd.read_csv(PROCESSED_DIR / 'mirna_mrna_pairs.csv')
    train_mirna = pd.read_csv(PROCESSED_DIR / 'train_mirna.csv', index_col=0)
    train_mrna = pd.read_csv(PROCESSED_DIR / 'train_mrna.csv', index_col=0)
    return pairs, train_mirna, train_mrna


def count_edge_types(graph: nx.Graph) -> Dict[str, int]:
    counts = {edge_type: 0 for edge_type in EDGE_TYPES}
    for _, _, data in graph.edges(data=True):
        edge_type = data.get('edge_type', '')
        if edge_type in counts:
            counts[edge_type] += 1
    return counts


def count_node_types(graph: nx.Graph) -> Tuple[int, int]:
    n_mirna = sum(
        1 for _, data in graph.nodes(data=True) if data.get('node_type') == 'miRNA'
    )
    n_mrna = sum(
        1 for _, data in graph.nodes(data=True) if data.get('node_type') == 'mRNA'
    )
    return n_mirna, n_mrna


def top_nodes_by_degree(graph: nx.Graph, top_n: int = 10) -> str:
    degrees = dict(graph.degree())
    ranked = sorted(degrees.items(), key=lambda item: (-item[1], item[0]))[:top_n]
    return '; '.join(f'{node}({deg})' for node, deg in ranked)


def compute_graph_statistics(graph: nx.Graph, tissue_stage: str) -> Dict:
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    n_mirna, n_mrna = count_node_types(graph)
    edge_counts = count_edge_types(graph)

    degrees = np.array([deg for _, deg in graph.degree()], dtype=float)
    if degrees.size == 0:
        avg_degree = median_degree = max_degree = 0.0
    else:
        avg_degree = float(degrees.mean())
        median_degree = float(np.median(degrees))
        max_degree = float(degrees.max())

    density = float(nx.density(graph)) if n_nodes > 1 else 0.0
    components = list(nx.connected_components(graph))
    n_components = len(components)
    largest_component_size = max((len(c) for c in components), default=0)

    return {
        'tissue_stage': tissue_stage,
        'n_nodes': n_nodes,
        'n_edges_total': n_edges,
        'n_mirna_nodes': n_mirna,
        'n_mrna_nodes': n_mrna,
        'n_mirna_mirna_edges': edge_counts['miRNA-miRNA'],
        'n_mrna_mrna_edges': edge_counts['mRNA-mRNA'],
        'n_mirna_mrna_edges': edge_counts['miRNA-mRNA'],
        'average_degree': round(avg_degree, 4),
        'median_degree': median_degree,
        'max_degree': max_degree,
        'graph_density': round(density, 6),
        'n_connected_components': n_components,
        'largest_component_size': largest_component_size,
        'top_10_nodes_by_degree': top_nodes_by_degree(graph),
    }


def plot_edge_type_counts(stats_df: pd.DataFrame, path: Path) -> None:
    x = np.arange(len(TISSUE_STAGES))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 6))

    for idx, edge_type in enumerate(EDGE_TYPES):
        col = {
            'miRNA-miRNA': 'n_mirna_mirna_edges',
            'mRNA-mRNA': 'n_mrna_mrna_edges',
            'miRNA-mRNA': 'n_mirna_mrna_edges',
        }[edge_type]
        values = stats_df.set_index('tissue_stage').loc[TISSUE_STAGES, col].values
        ax.bar(x + (idx - 1) * width, values, width=width, label=edge_type)

    ax.set_xticks(x)
    ax.set_xticklabels(TISSUE_STAGES)
    ax.set_xlabel('Tissue stage')
    ax.set_ylabel('Edge count')
    ax.set_title('Edge counts by type across tissue-specific networks')
    ax.legend(title='Edge type')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_degree_distribution(graphs: Dict[str, nx.Graph], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, tissue in zip(axes, TISSUE_STAGES):
        degrees = [deg for _, deg in graphs[tissue].degree()]
        if degrees:
            bins = min(30, max(5, len(set(degrees))))
            ax.hist(degrees, bins=bins, color='#4C72B0', edgecolor='white', alpha=0.85)
        ax.set_title(tissue.capitalize())
        ax.set_xlabel('Node degree')
        ax.grid(axis='y', alpha=0.3)
    axes[0].set_ylabel('Number of nodes')
    fig.suptitle('Degree distributions by tissue-specific network', y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 72)
    print('NETWORK STATISTICS SUMMARY')
    print('=' * 72)

    pairs, train_mirna, train_mrna = load_reference_inputs()
    print(
        f'Reference inputs loaded: {len(pairs)} regulatory pairs, '
        f'{train_mirna.shape[1]} train miRNAs, {train_mrna.shape[1]} train mRNAs',
        flush=True,
    )

    graphs: Dict[str, nx.Graph] = {}
    rows: List[Dict] = []

    for tissue in TISSUE_STAGES:
        graph_path = RESULTS_DIR / f'{tissue}_network.graphml'
        print(f'Loading {graph_path}...', flush=True)
        graph = nx.read_graphml(graph_path)
        graphs[tissue] = graph
        rows.append(compute_graph_statistics(graph, tissue))

    stats_df = pd.DataFrame(rows)

    table_path = TABLES_DIR / 'network_statistics.csv'
    fig_edges_path = FIGURES_DIR / 'network_edge_type_counts.png'
    fig_degree_path = FIGURES_DIR / 'network_degree_distribution.png'

    stats_df.to_csv(table_path, index=False)
    plot_edge_type_counts(stats_df, fig_edges_path)
    plot_degree_distribution(graphs, fig_degree_path)

    display_cols = [
        'tissue_stage', 'n_nodes', 'n_edges_total', 'n_mirna_nodes', 'n_mrna_nodes',
        'n_mirna_mirna_edges', 'n_mrna_mrna_edges', 'n_mirna_mrna_edges',
        'average_degree', 'median_degree', 'max_degree', 'graph_density',
        'n_connected_components', 'largest_component_size',
    ]
    print('\nNetwork statistics table:')
    print(stats_df[display_cols].to_string(index=False))

    print('\nTop 10 nodes by degree:')
    for _, row in stats_df.iterrows():
        print(f"  {row['tissue_stage']}: {row['top_10_nodes_by_degree']}")

    print('\nSaved files:')
    for path in (table_path, fig_edges_path, fig_degree_path):
        print(f'  {path}')


if __name__ == '__main__':
    main()
