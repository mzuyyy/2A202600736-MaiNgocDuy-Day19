from __future__ import annotations

from pathlib import Path

import matplotlib
import networkx as nx

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def draw_graph(graph_path: Path, output_path: Path, *, max_nodes: int = 50) -> Path:
    graph = nx.read_graphml(graph_path, force_multigraph=True)
    selected = _top_degree_nodes(graph, max_nodes)
    view = graph.subgraph(selected).copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(16, 12))
    positions = nx.spring_layout(view, seed=42, k=0.8)
    degrees = dict(view.degree())
    node_sizes = [300 + (degrees[node] * 80) for node in view.nodes]

    nx.draw_networkx_nodes(
        view,
        positions,
        node_size=node_sizes,
        node_color="#9ecae1",
        edgecolors="#08519c",
        linewidths=1.2,
    )
    nx.draw_networkx_edges(
        view,
        positions,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=10,
        edge_color="#9e9e9e",
        alpha=0.55,
    )
    nx.draw_networkx_labels(view, positions, font_size=8)
    nx.draw_networkx_edge_labels(
        view,
        positions,
        edge_labels=_edge_labels(view),
        font_size=6,
        label_pos=0.55,
    )
    plt.title(
        f"GraphRAG knowledge graph: {view.number_of_nodes()} nodes, "
        f"{view.number_of_edges()} edges"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def _top_degree_nodes(graph: nx.MultiDiGraph[str], limit: int) -> list[str]:
    ranked = sorted(graph.degree, key=lambda item: item[1], reverse=True)
    return [str(node) for node, _degree in ranked[:limit]]


def _edge_labels(graph: nx.MultiDiGraph[str]) -> dict[tuple[str, str], str]:
    labels: dict[tuple[str, str], str] = {}
    for source, target, data in graph.edges(data=True):
        key = (str(source), str(target))
        labels.setdefault(key, str(data.get("relation", "")))
    return labels
