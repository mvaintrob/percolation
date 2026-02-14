"""
plot.py — Plot an embedded tree in 2D.

Uses first two coordinates of the embedding (pads if d=1).
Colors edges by descendant mass. Internal nodes sized by mass.
Root is bright red and larger.
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def plot_tree(tree, positions=None, d=None,
              show_labels=False, figsize=(10, 8), save=None):
    """Plot an embedded binary tree.

    Args:
        tree: dict from coalescent.py (needs parent, masses, merge_pairs)
        positions: (2k-1, d) tensor of node positions.
            If None, calls embed_tree automatically.
        d: embedding dimension (required if positions is None)
        show_labels: label leaves with their index
        figsize: matplotlib figure size
        save: if given, save figure to this path

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
    xy = project_2d(positions, k).numpy()

    # Compute descendant masses for all nodes
    desc_mass = torch.zeros(n_nodes, dtype=torch.float64)
    desc_mass[:k] = masses
    for step in range(k - 1):
        j = k + step
        a = merge_pairs[step, 0].item()
        b = merge_pairs[step, 1].item()
        desc_mass[j] = desc_mass[a] + desc_mass[b]

    # Color map: YlOrRd, log-scaled actual mass
    mass_min = desc_mass[desc_mass > 0].min().item()
    mass_max = desc_mass.max().item()
    norm = mcolors.LogNorm(vmin=mass_min, vmax=mass_max)
    cmap = plt.cm.YlOrRd

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Draw edges (colored by child node's descendant mass)
    for j in range(n_nodes):
        p = parent[j].item()
        if p < 0:
            continue
        m = desc_mass[j].item()
        if m <= 0:
            m = mass_min
        color = cmap(norm(m))
        ax.plot([xy[j, 0], xy[p, 0]], [xy[j, 1], xy[p, 1]],
                color=color, linewidth=0.8, linestyle='--', alpha=0.5)

    # Draw internal nodes (size ∝ sqrt(descendant mass), uniform color)
    DOT_COLOR = '#333333'
    max_mass = desc_mass.max().item()
    for j in range(k, n_nodes):
        if j == root:
            continue  # draw root separately
        m = desc_mass[j].item()
        size = 5 + 50 * (m / max_mass) ** 0.5 if max_mass > 0 else 10
        ax.scatter(xy[j, 0], xy[j, 1], c=DOT_COLOR,
                   s=size, zorder=3, edgecolors='none', alpha=0.7)

    # Draw root (bright red, larger)
    root_size = 80 + 50 * 1.0  # always max relative size
    ax.scatter(xy[root, 0], xy[root, 1], c='red',
               s=root_size, zorder=5, edgecolors='darkred',
               linewidths=0.8, alpha=0.9)

    # Draw leaves (small hollow circles, uniform color)
    for j in range(k):
        ax.scatter(xy[j, 0], xy[j, 1], s=4, zorder=4,
                   facecolors='none', edgecolors=DOT_COLOR,
                   linewidths=0.8)

    # Labels on leaves
    if show_labels:
        for j in range(k):
            ax.annotate(str(j), (xy[j, 0], xy[j, 1]),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=8, color='black')

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('descendant mass', fontsize=10)

    ax.set_aspect('equal')
    embed_d = positions.shape[1]
    ax.set_title(f'CRT embedding (k={k}, d={embed_d})', fontsize=12)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    if save:
        fig.savefig(save, dpi=150, bbox_inches='tight')

    plt.show()

    return fig
