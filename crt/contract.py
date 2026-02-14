"""
contract.py — Collapse short edges to produce a non-binary tree.

Deterministic given the binary tree and epsilon.
"""

import torch


def contract(tree, epsilon):
    """Collapse all internal edges shorter than epsilon.

    Args:
        tree: dict from coalescent.py
        epsilon: float, collapse threshold

    Returns dict:
        surviving_nodes: (n,) int — node ids that survive
        children: list of lists — children[i] = list of child node ids
            for surviving node i (in terms of original node ids)
        node_masses: (n,) float — descendant mass of each surviving node
        node_heights: (n,) float — height of each surviving node
        surviving_merges: list of ints — indices into original merge_pairs
            for merges that survive (edge >= epsilon)
        epsilon: the threshold used
        original_tree: reference to input tree
    """
    merge_pairs = tree["merge_pairs"]
    merge_times = tree["merge_times"]
    heights = tree["heights"]
    masses = tree["masses"]
    parent = tree["parent"]
    k = len(masses)
    n_nodes = 2 * k - 1

    edge_lengths = tree["edge_lengths"]

    # An internal edge (from node j to parent[j]) is "short" if
    # edge_lengths[j] < epsilon AND j is an internal node (j >= k).
    # Leaf edges are never collapsed.

    # For each node, find its "representative" after collapsing:
    # follow parent pointers, skipping over short internal edges.
    rep = list(range(n_nodes))

    # Process internal nodes in merge order (bottom up).
    # If internal node j has a short edge to its parent, merge j
    # into its parent: all children of j become children of parent[j].
    collapsed = [False] * n_nodes
    for step in range(k - 1):
        j = k + step  # internal node
        if j == n_nodes - 1:
            continue  # root has no edge
        elen = edge_lengths[j].item()
        if elen < epsilon:
            collapsed[j] = True

    # Build representative map: follow collapsed edges upward
    for step in range(k - 1):
        j = k + step
        if collapsed[j]:
            p = parent[j].item()
            # rep[j] should point to rep[p]
            rep[j] = rep[p]

    # Fix up: need to iterate until stable since rep[p] might also
    # be collapsed
    changed = True
    while changed:
        changed = False
        for j in range(n_nodes):
            if rep[rep[j]] != rep[j]:
                rep[j] = rep[rep[j]]
                changed = True

    # Build children lists for surviving nodes
    surviving_set = set()
    children_map = {}

    for j in range(n_nodes):
        r = rep[j]
        surviving_set.add(r)
        if r not in children_map:
            children_map[r] = []

    # A surviving node r has as children all nodes j where:
    # - rep[j] != r (j is not collapsed into r), but
    # - rep[parent[j]] == r (j's parent IS collapsed into r, or is r)
    # OR j is a leaf whose parent collapses into r.
    for j in range(n_nodes):
        p = parent[j].item()
        if p < 0:
            continue  # root
        if rep[j] == j and rep[p] != j:
            # j survives and its parent maps to rep[p]
            r = rep[p]
            if r != j:
                children_map.setdefault(r, []).append(j)

    # For leaves: they always survive (rep[j] == j for j < k)
    # and connect to rep[parent[j]]

    # Surviving nodes sorted
    surviving_nodes = sorted(surviving_set)
    node_to_idx = {n: i for i, n in enumerate(surviving_nodes)}

    n_surviving = len(surviving_nodes)
    node_masses_out = torch.zeros(n_surviving, dtype=torch.float64)
    node_heights_out = torch.zeros(n_surviving, dtype=torch.float64)

    # Compute descendant masses
    # First compute descendant mass for all original nodes
    desc_mass = torch.zeros(n_nodes, dtype=torch.float64)
    desc_mass[:k] = masses
    for step in range(k - 1):
        j = k + step
        a = merge_pairs[step, 0].item()
        b = merge_pairs[step, 1].item()
        desc_mass[j] = desc_mass[a] + desc_mass[b]

    for i, n in enumerate(surviving_nodes):
        node_masses_out[i] = desc_mass[n]
        node_heights_out[i] = heights[n]

    # Surviving merges: merge step i survives if internal node k+i
    # is not collapsed
    surviving_merges = []
    for step in range(k - 1):
        j = k + step
        if not collapsed[j]:
            surviving_merges.append(step)

    # Build children as list of lists (using original node ids)
    children_out = []
    for n in surviving_nodes:
        ch = children_map.get(n, [])
        children_out.append(sorted(ch))

    return {
        "surviving_nodes": torch.tensor(surviving_nodes, dtype=torch.long),
        "children": children_out,
        "node_masses": node_masses_out,
        "node_heights": node_heights_out,
        "surviving_merges": surviving_merges,
        "epsilon": epsilon,
        "original_tree": tree,
    }
