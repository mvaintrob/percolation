"""
embed.py — Embed a binary tree in R^d via independent Gaussian
displacements along edges.

Each edge e with length ℓ_e gets displacement ~ N(0, ℓ_e * I_d).
Node positions are cumulative sums from root to leaves.
"""

import torch
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TreeConfig:
    k: int = 10
    masses: Optional[torch.Tensor] = None  # default: uniform 1/k
    d: int = 11
    embed: bool = False
    seed: Optional[int] = None


def embed_tree(tree, d):
    """Embed a binary tree in R^d.

    Root is placed at the origin. Each edge gets an independent
    Gaussian displacement with variance = edge_length * I_d.

    Args:
        tree: dict from coalescent.py (must have parent, edge_lengths, heights)
        d: embedding dimension

    Returns:
        positions: (2k-1, d) tensor of all node positions
    """
    masses = tree["masses"]
    k = len(masses)
    n_nodes = 2 * k - 1
    parent = tree["parent"]

    # edge_lengths has size n_nodes - 1 (no edge for root)
    edge_lengths = tree["edge_lengths"]

    positions = torch.zeros((n_nodes, d), dtype=torch.float64)
    # Root = node 2k-2, at origin

    # Process nodes top-down: root first, then in reverse merge order
    # Parent of node j is parent[j]. Root has parent -1.
    # Internal nodes are k..2k-2 in merge order.
    # We need to process parents before children.

    # Build processing order: BFS from root
    root = n_nodes - 1
    order = [root]
    children_map = {i: [] for i in range(n_nodes)}
    for j in range(n_nodes - 1):
        p = parent[j].item()
        if p >= 0:
            children_map[p].append(j)

    head = 0
    while head < len(order):
        node = order[head]
        for c in children_map[node]:
            order.append(c)
        head += 1

    # Assign positions top-down
    for node in order:
        if node == root:
            continue  # root stays at origin
        p = parent[node].item()
        ell = edge_lengths[node].item()
        if ell > 0:
            displacement = torch.randn(d, dtype=torch.float64) * (ell ** 0.5)
        else:
            displacement = torch.zeros(d, dtype=torch.float64)
        positions[node] = positions[p] + displacement

    return positions


def project_2d(positions, k):
    """Project node positions to 2D for plotting.

    If d == 2: return as-is.
    If d == 1: pad with zeros.
    If d > 2: SVD on leaf positions (no centering, keeps origin at 0),
              project all nodes onto top 2 singular vectors.

    Args:
        positions: (n_nodes, d) tensor
        k: number of leaves (nodes 0..k-1)

    Returns:
        (n_nodes, 2) tensor
    """
    d = positions.shape[1]

    if d == 2:
        return positions.clone()

    if d == 1:
        xy = torch.zeros((positions.shape[0], 2), dtype=positions.dtype)
        xy[:, 0] = positions[:, 0]
        return xy

    # PCA directions from centered leaf data, but project uncentered
    # (so origin maps to origin)
    leaf_pos = positions[:k]  # (k, d)
    leaf_centered = leaf_pos - leaf_pos.mean(dim=0, keepdim=True)

    # SVD of centered data to get PC directions
    U, S, Vh = torch.linalg.svd(leaf_centered, full_matrices=False)
    pcs = Vh[:2]  # (2, d) — top 2 principal directions

    # Project all nodes WITHOUT centering
    projected = positions @ pcs.T  # (n_nodes, 2)

    return projected
