"""
plot_sequence.py — Generate a sequence of images showing progressive
expansion of a contracted tree.

Frame 0: corona (all internal edges collapsed, k leaves + 1 root)
Frame i: i internal edges revealed
Frame k-2: full binary tree (all k-2 internal edges visible)

Only INTERNAL edges (between internal nodes) are collapsed. Leaves
always remain visible. Each frame reveals one additional internal edge.

Edge ordering options:
  "merge": reveal in reverse merge order (= reverse height order,
           true coalescent inverse). DEFAULT.
  "length": reveal longest edges first (metric coarsening)
  or pass a list of internal node ids specifying custom order

Centroid mode for contracted clusters:
  "path": length-weighted edge midpoint average (default).
  "leaf": unweighted average of leaf descendant positions.
  "root": position of the root node of the collapsed sub-tree.
"""

import os
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


def _compute_path_centroid(members, k, xy_all, parent, edge_lengths):
    """Compute length-weighted edge midpoint centroid for a set of nodes."""
    member_set = set(members)
    total_weight = 0.0
    weighted_pos = np.zeros(2)

    for j in members:
        p = parent[j].item()
        if p < 0:
            continue
        if p not in member_set:
            continue
        ell = edge_lengths[j].item() if j < len(edge_lengths) else 0.0
        if ell <= 0:
            continue
        midpoint = (xy_all[j].numpy() + xy_all[p].numpy()) / 2.0
        weighted_pos += ell * midpoint
        total_weight += ell

    if total_weight > 0:
        return weighted_pos / total_weight
    else:
        coords = torch.stack([xy_all[m] for m in members])
        return coords.mean(dim=0).numpy()


def plot_contraction_sequence(tree, positions, output_dir="frames",
                               edge_order="merge",
                               centroid_mode="path",
                               show_labels=False, figsize=(10, 8),
                               make_gif=False, gif_path="tree.gif",
                               gif_duration=500,
                               make_pdf=True, pdf_path=None):
    """Generate PNG frames and optionally a PDF slideshow.

    Args:
        tree: dict from coalescent.py
        positions: (2k-1, d) tensor of all node positions from embedding
        output_dir: directory to save frames
        edge_order: "merge" (reverse merge/height order, default),
            "length" (longest first), or list of internal node ids
        centroid_mode: "path" (length-weighted edge midpoints, default),
            "leaf" (leaf average), "root" (root of collapsed subtree)
        show_labels: label leaves with index
        figsize: matplotlib figure size
        make_gif: if True, also combine frames into a GIF
        gif_path: path for the GIF
        gif_duration: ms per frame in GIF
        make_pdf: if True, generate a multi-page PDF (default True)
        pdf_path: path for the PDF (default: output_dir/../slides.pdf)

    Returns:
        list of frame file paths
    """
    masses = tree["masses"]
    k = len(masses)
    n_nodes = 2 * k - 1
    root = n_nodes - 1
    parent = tree["parent"]
    merge_pairs = tree["merge_pairs"]
    edge_lengths = tree["edge_lengths"]
    embed_d = positions.shape[1]

    # Internal edges
    internal_edges = list(range(k, n_nodes - 1))
    n_internal = len(internal_edges)

    # Determine reveal order
    if isinstance(edge_order, str):
        if edge_order == "merge":
            internal_edges.sort(reverse=True)
        elif edge_order == "length":
            internal_edges.sort(
                key=lambda j: edge_lengths[j].item(), reverse=True)
        else:
            raise ValueError(f"Unknown edge_order: {edge_order}")
    else:
        internal_edges = list(edge_order)

    # Project to 2D
    from .embed import project_2d
    xy_all = project_2d(positions, k)

    # Descendant masses
    desc_mass = torch.zeros(n_nodes, dtype=torch.float64)
    desc_mass[:k] = masses
    for step in range(k - 1):
        j = k + step
        a = merge_pairs[step, 0].item()
        b = merge_pairs[step, 1].item()
        desc_mass[j] = desc_mass[a] + desc_mass[b]

    # Color scale: YlOrRd, log-scaled with actual mass values
    mass_min = desc_mass[desc_mass > 0].min().item()
    mass_max = desc_mass.max().item()
    norm = mcolors.LogNorm(vmin=mass_min, vmax=mass_max)
    cmap = plt.cm.YlOrRd

    # Generate frames
    os.makedirs(output_dir, exist_ok=True)
    frame_paths = []

    # Open PDF if requested
    if pdf_path is None:
        pdf_path = os.path.join(
            os.path.dirname(output_dir.rstrip('/')), "slides.pdf")
    pdf = PdfPages(pdf_path) if make_pdf else None

    for level in range(n_internal + 1):
        collapsed = set(internal_edges[level:])

        # Build representative map
        rep = list(range(n_nodes))
        for j in collapsed:
            p = parent[j].item()
            if p >= 0:
                rep[j] = p

        changed = True
        while changed:
            changed = False
            for i in range(n_nodes):
                if rep[rep[i]] != rep[i]:
                    rep[i] = rep[rep[i]]
                    changed = True

        # Contracted nodes and members
        contracted_nodes = {}
        for i in range(n_nodes):
            r = rep[i]
            if r not in contracted_nodes:
                contracted_nodes[r] = []
            contracted_nodes[r].append(i)

        # Positions
        contracted_xy = {}
        contracted_mass = {}

        for r, members in contracted_nodes.items():
            contracted_mass[r] = desc_mass[r].item()

            if r < k:
                contracted_xy[r] = xy_all[r].numpy()
                continue

            if centroid_mode == "path":
                contracted_xy[r] = _compute_path_centroid(
                    members, k, xy_all, parent, edge_lengths)
            elif centroid_mode == "leaf":
                leaf_members = [m for m in members if m < k]
                if leaf_members:
                    coords = torch.stack([xy_all[m] for m in leaf_members])
                    contracted_xy[r] = coords.mean(dim=0).numpy()
                else:
                    coords = torch.stack([xy_all[m] for m in members])
                    contracted_xy[r] = coords.mean(dim=0).numpy()
            elif centroid_mode == "root":
                contracted_xy[r] = xy_all[r].numpy()
            else:
                raise ValueError(f"Unknown centroid_mode: {centroid_mode}")

        # Edges
        edges = set()
        for r in contracted_nodes:
            p = parent[r].item()
            if p < 0:
                continue
            p_rep = rep[p]
            if p_rep != r:
                edges.add((r, p_rep))

        for i in range(k):
            p = parent[i].item()
            if p >= 0:
                p_rep = rep[p]
                if p_rep != i:
                    edges.add((i, p_rep))

        edges = list(edges)

        # Find the root contracted node (rep of the actual root)
        root_rep = rep[root]

        # ── Plot ──────────────────────────────────────

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Edges colored by child descendant mass
        for a, b in edges:
            xy_a = contracted_xy.get(a)
            xy_b = contracted_xy.get(b)
            if xy_a is None or xy_b is None:
                continue
            child = a if desc_mass[a] < desc_mass[b] else b
            m_child = contracted_mass.get(child, mass_min)
            if m_child <= 0:
                m_child = mass_min
            color = cmap(norm(m_child))
            ax.plot([xy_a[0], xy_b[0]], [xy_a[1], xy_b[1]],
                    color=color, linewidth=0.8, linestyle='--', alpha=0.5)

        # Internal nodes (uniform color, size ∝ sqrt(mass))
        internal_masses = [contracted_mass[r] for r in contracted_nodes
                           if r >= k]
        max_mass = max(internal_masses) if internal_masses else 1.0

        DOT_COLOR = '#333333'
        for r in contracted_nodes:
            if r < k:
                continue
            if r == root_rep:
                continue  # draw root separately
            xy = contracted_xy[r]
            m = contracted_mass[r]
            size = 5 + 50 * (m / max_mass) ** 0.5 if max_mass > 0 else 10
            ax.scatter(xy[0], xy[1], c=DOT_COLOR, s=size, zorder=3,
                       edgecolors='none', alpha=0.7)

        # Root node (bright red, larger)
        if root_rep in contracted_xy:
            xy_root = contracted_xy[root_rep]
            root_size = 80 + 50 * 1.0
            ax.scatter(xy_root[0], xy_root[1], c='red',
                       s=root_size, zorder=5, edgecolors='darkred',
                       linewidths=0.8, alpha=0.9)

        # Leaves (small hollow circles)
        for i in range(k):
            xy = contracted_xy[i]
            ax.scatter(xy[0], xy[1], s=4, zorder=4,
                       facecolors='none', edgecolors=DOT_COLOR,
                       linewidths=0.8)

        # Labels
        if show_labels:
            for i in range(k):
                xy = contracted_xy[i]
                ax.annotate(str(i), xy, textcoords="offset points",
                            xytext=(5, 5), fontsize=8)

        # Colorbar with actual mass values
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label('descendant mass', fontsize=10)

        ax.set_aspect('equal')
        ax.set_title(f'Level {level}/{n_internal}: '
                     f'{level} internal edges revealed '
                     f'(k={k}, d={embed_d})',
                     fontsize=11)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        frame_path = os.path.join(output_dir, f"frame_{level:03d}.png")
        fig.savefig(frame_path, dpi=150, bbox_inches='tight')
        frame_paths.append(frame_path)

        if pdf is not None:
            pdf.savefig(fig, bbox_inches='tight')

        plt.close(fig)

    if pdf is not None:
        pdf.close()
        print(f"Saved PDF to {pdf_path}")

    print(f"Saved {len(frame_paths)} frames to {output_dir}/")

    if make_gif:
        try:
            from PIL import Image
            images = [Image.open(fp) for fp in frame_paths]
            images[0].save(gif_path, save_all=True,
                           append_images=images[1:],
                           duration=gif_duration, loop=0)
            print(f"Saved GIF to {gif_path}")
        except ImportError:
            print("PIL not available. Install: pip install Pillow")

    return frame_paths
