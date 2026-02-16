"""
plot.py — Plot an embedded tree in 2D.

Vectorized: uses LineCollection for edges and single scatter calls
for node groups. Handles k=5000+ without issue.
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.collections as mcollections
import numpy as np


def plot_tree(tree, positions=None, d=None,
              show_labels=False, figsize=(10, 8), save=None,
              _projected=None, leaves_only=False):
    """Plot an embedded binary tree.

    Args:
        tree: dict from coalescent.py (needs parent, masses, merge_pairs)
        positions: (2k-1, d) tensor of node positions.
            If None, calls embed_tree automatically.
        d: embedding dimension (required if positions is None)
        show_labels: label leaves with their index
        figsize: matplotlib figure size
        save: if given, save figure to this path
        _projected: optional (2k-1, 2) tensor of pre-projected 2D positions
        leaves_only: if True, plot only the leaf point cloud

    Returns:
        matplotlib Figure
    """
    from .embed import embed_tree, project_2d

    masses = tree["masses"]
    k = len(masses)
    n_nodes = 2 * k - 1
    root = n_nodes - 1
    parent = tree["parent"]
    merge_pairs = tree["merge_pairs"]

    # Get or compute positions
    if positions is None:
        if d is None:
            d = 2
        positions = embed_tree(tree, d)

    # Project to 2D
    if _projected is not None:
        xy = _projected.numpy() if hasattr(_projected, 'numpy') else _projected
    else:
        xy = project_2d(positions, k).numpy()

    # Compute descendant masses for all nodes
    desc_mass = torch.zeros(n_nodes, dtype=torch.float64)
    desc_mass[:k] = masses
    for step in range(k - 1):
        j = k + step
        a = merge_pairs[step, 0].item()
        b = merge_pairs[step, 1].item()
        desc_mass[j] = desc_mass[a] + desc_mass[b]

    desc_mass_np = desc_mass.numpy()

    # Color map
    mass_min = desc_mass[desc_mass > 0].min().item()
    mass_max = desc_mass.max().item()
    norm = mcolors.LogNorm(vmin=mass_min, vmax=mass_max)
    cmap = plt.cm.YlOrRd

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if not leaves_only:
        # ── Edges (vectorized via LineCollection) ──────────
        parent_np = parent.numpy()
        mask = parent_np >= 0
        child_idx = np.where(mask)[0]
        parent_idx = parent_np[child_idx]

        segments = np.stack([
            xy[child_idx],
            xy[parent_idx],
        ], axis=1)

        edge_masses = np.clip(desc_mass_np[child_idx], mass_min, None)
        edge_colors = cmap(norm(edge_masses))

        lc = mcollections.LineCollection(
            segments, colors=edge_colors,
            linewidths=0.8, linestyles='--', alpha=0.5)
        ax.add_collection(lc)

        # ── Internal nodes (single scatter, excluding root) ────
        internal_idx = np.arange(k, n_nodes - 1)
        if len(internal_idx) > 0:
            internal_masses = desc_mass_np[internal_idx]
            internal_sizes = 5 + 50 * (internal_masses / mass_max) ** 0.5

            ax.scatter(xy[internal_idx, 0], xy[internal_idx, 1],
                       c='#333333', s=internal_sizes, zorder=3,
                       edgecolors='none', alpha=0.7)

        # ── Root (bright red) ──────────────────────────────
        ax.scatter(xy[root, 0], xy[root, 1], c='red',
                   s=130, zorder=5, edgecolors='darkred',
                   linewidths=0.8, alpha=0.9)

    # ── Leaves ─────────────────────────────────────────
    leaf_idx = np.arange(k)
    if leaves_only:
        # Color leaves by distance to origin
        leaf_dists = np.sqrt(xy[leaf_idx, 0]**2 + xy[leaf_idx, 1]**2)
        dist_norm = mcolors.Normalize(
            vmin=leaf_dists.min(), vmax=leaf_dists.max())
        leaf_cmap = plt.cm.cool  # cyan → magenta
        leaf_colors = leaf_cmap(dist_norm(leaf_dists))
        ax.scatter(xy[leaf_idx, 0], xy[leaf_idx, 1], s=12, zorder=4,
                   c=leaf_colors, edgecolors='none', alpha=0.8)
    else:
        ax.scatter(xy[leaf_idx, 0], xy[leaf_idx, 1], s=4, zorder=4,
                   facecolors='none', edgecolors='#333333',
                   linewidths=0.8)

    # Labels
    if show_labels:
        for j in range(k):
            ax.annotate(str(j), (xy[j, 0], xy[j, 1]),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=8, color='black')

    # Colorbar
    if leaves_only:
        leaf_dists = np.sqrt(xy[:k, 0]**2 + xy[:k, 1]**2)
        dist_norm = mcolors.Normalize(
            vmin=leaf_dists.min(), vmax=leaf_dists.max())
        sm = plt.cm.ScalarMappable(cmap=plt.cm.cool, norm=dist_norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label('distance to root', fontsize=10)
    else:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label('descendant mass', fontsize=10)

    # Autoscale since we used add_collection
    ax.autoscale_view()
    ax.set_aspect('equal')
    embed_d = positions.shape[1]
    ax.set_title(f'CRT embedding (k={k}, d={embed_d})', fontsize=12)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    if save:
        fig.savefig(save, dpi=150, bbox_inches='tight')

    plt.show()

    return fig
