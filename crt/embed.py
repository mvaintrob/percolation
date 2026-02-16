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
    edge_lengths = tree["edge_lengths"]  # (n_nodes - 1,)

    positions = torch.zeros((n_nodes, d), dtype=torch.float64)
    root = n_nodes - 1

    # Generate all displacements at once: (n_nodes-1, d)
    noise = torch.randn(n_nodes - 1, d, dtype=torch.float64)
    # Scale by sqrt(edge_length): broadcast (n_nodes-1, 1)
    scales = edge_lengths.clamp(min=0).sqrt().unsqueeze(1)
    displacements = noise * scales  # (n_nodes-1, d)

    # Process in topological order (parents before children).
    # Internal nodes k..2k-2 are in merge order, so node k+s is
    # created at step s. Its children were created earlier.
    # Process order: root (2k-2), then internal nodes in reverse
    # (2k-3 down to k), then leaves (0..k-1).
    # But actually we just need: for each node, its parent is
    # already placed. The tree is indexed so that parent[j] > j
    # for all non-root j (parent is always a later internal node).
    # So processing in reverse order (2k-3 down to 0) works:
    # when we process j, parent[j] > j is already done.

    for j in range(n_nodes - 2, -1, -1):
        p = parent[j].item()
        positions[j] = positions[p] + displacements[j]

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


def project_2d_tsne(positions, k, perplexity=30.0):
    """Project node positions to 2D using t-SNE on leaves.

    Runs t-SNE on leaf positions, then places internal nodes
    at the average of their children's t-SNE coordinates (via
    the tree structure, if available) or by projecting with
    a learned linear map.

    Falls back to PCA if sklearn is not available.

    Args:
        positions: (n_nodes, d) tensor
        k: number of leaves (nodes 0..k-1)
        perplexity: t-SNE perplexity parameter

    Returns:
        (n_nodes, 2) tensor
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("sklearn not available for t-SNE, falling back to PCA")
        return project_2d(positions, k)

    n_nodes = positions.shape[0]
    leaf_pos = positions[:k].numpy()

    # Run t-SNE on leaves
    perp = min(perplexity, max(5.0, k / 4.0))  # clamp for small k
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
    leaf_2d = tsne.fit_transform(leaf_pos)

    # For internal nodes: fit a linear map from d-dim to 2D
    # using the leaf correspondences, then apply to all nodes.
    # This is a least-squares fit: leaf_2d ≈ leaf_pos @ W
    leaf_pos_t = torch.as_tensor(leaf_pos, dtype=torch.float64)
    leaf_2d_t = torch.as_tensor(leaf_2d, dtype=torch.float64)
    W = torch.linalg.lstsq(leaf_pos_t, leaf_2d_t).solution  # (d, 2)

    # Project all nodes
    all_2d = positions.double() @ W

    # Override leaf positions with exact t-SNE output
    all_2d[:k] = leaf_2d_t

    return all_2d


def project_2d_umap(positions, k, n_neighbors=15, min_dist=0.1):
    """Project node positions to 2D using UMAP on leaves.

    Runs UMAP on leaf positions, then places internal nodes
    via a least-squares linear map fit on the leaf correspondences.

    Falls back to PCA if umap-learn is not available.

    Args:
        positions: (n_nodes, d) tensor
        k: number of leaves (nodes 0..k-1)
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter

    Returns:
        (n_nodes, 2) tensor
    """
    try:
        from umap import UMAP
    except ImportError:
        print("umap-learn not available, falling back to PCA. "
              "Install with: pip install umap-learn")
        return project_2d(positions, k)

    n_nodes = positions.shape[0]
    leaf_pos = positions[:k].numpy()

    # Clamp n_neighbors for small k
    nn = min(n_neighbors, max(2, k - 1))
    reducer = UMAP(n_components=2, n_neighbors=nn,
                   min_dist=min_dist, random_state=42)
    leaf_2d = reducer.fit_transform(leaf_pos)

    # Least-squares linear map for internal nodes
    leaf_pos_t = torch.as_tensor(leaf_pos, dtype=torch.float64)
    leaf_2d_t = torch.as_tensor(leaf_2d, dtype=torch.float64)
    W = torch.linalg.lstsq(leaf_pos_t, leaf_2d_t).solution

    all_2d = positions.double() @ W
    all_2d[:k] = leaf_2d_t

    return all_2d
