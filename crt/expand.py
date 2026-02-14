"""
expand.py — Resolve non-binary nodes by running a constrained
forward coalescent.

The contracted tree defines groups of leaves that must merge
internally before merging across groups. We run the standard
additive coalescent but restrict eligible pairs at each step.
"""

import torch
import random


def expand(contracted_tree):
    """Expand a contracted tree back to a full binary tree.

    Runs a forward additive coalescent on all k leaves, constrained
    so that particles only merge when they belong to the same
    group in the contracted tree (or their groups have already
    fully merged and share a parent).

    Args:
        contracted_tree: dict from contract.py

    Returns dict (same format as coalescent.py):
        merge_pairs, merge_times, edge_lengths, heights, masses, parent
    """
    original_tree = contracted_tree["original_tree"]
    masses = original_tree["masses"]
    k = len(masses)

    if k <= 1:
        return {
            "merge_pairs": torch.zeros((0, 2), dtype=torch.long),
            "merge_times": torch.zeros(0, dtype=torch.float64),
            "edge_lengths": torch.zeros(0, dtype=torch.float64),
            "heights": torch.zeros(1, dtype=torch.float64),
            "masses": masses,
            "parent": torch.full((1,), -1, dtype=torch.long),
        }

    # Build the group structure from the contracted tree.
    # Each leaf belongs to a "group" = the lowest surviving
    # internal node that contains it.
    # Groups form a tree (the contracted tree itself).

    children_lists = contracted_tree["children"]
    surviving_nodes = contracted_tree["surviving_nodes"].tolist()
    node_to_idx = {n: i for i, n in enumerate(surviving_nodes)}

    # For each surviving node, find its leaf descendants
    def get_leaf_descendants(node_id):
        if node_id < k:
            return [node_id]
        idx = node_to_idx[node_id]
        leaves = []
        for ch in children_lists[idx]:
            leaves.extend(get_leaf_descendants(ch))
        return leaves

    # Build group hierarchy: for each surviving internal node,
    # record its children (in the contracted tree)
    # and the set of leaves under each child.
    # A merge is "allowed" if both particles are under the same
    # surviving internal node AND under the same child of that node,
    # OR if one of the children is fully merged (single particle).

    # Simpler approach: assign each leaf a "group path" — the
    # sequence of surviving ancestors. Two particles can merge
    # iff they share their lowest unresolved ancestor.

    # Even simpler: just track which surviving node each particle
    # currently "belongs to" (= lowest surviving node containing it).
    # When all particles under a surviving node merge into one,
    # that particle "moves up" to the parent surviving node.

    # For each leaf, find its immediate surviving parent
    # (the surviving node that directly contains it as a child)
    leaf_group = {}
    for idx, node_id in enumerate(surviving_nodes):
        for ch in children_lists[idx]:
            if ch < k:
                leaf_group[ch] = node_id
            # For internal children: their leaves are assigned
            # to that internal child, not to node_id
            # (they need to merge internally first)

    # For internal surviving nodes that are children of other
    # surviving nodes: their leaves belong to them
    def assign_groups(node_id):
        if node_id < k:
            return
        idx = node_to_idx[node_id]
        for ch in children_lists[idx]:
            if ch < k:
                leaf_group[ch] = node_id
            else:
                assign_groups(ch)

    # Find root of contracted tree (the surviving node with no
    # surviving parent)
    root_surviving = surviving_nodes[-1]  # highest node
    assign_groups(root_surviving)

    # Parent map in contracted tree
    contracted_parent = {}
    for idx, node_id in enumerate(surviving_nodes):
        for ch in children_lists[idx]:
            contracted_parent[ch] = node_id

    # Now run the coalescent.
    # State: active particles, each with a mass and a group.
    # Two particles can merge iff they have the same group.
    # When a group has only one particle left, reassign it
    # to the parent group.

    n_nodes = 2 * k - 1
    particle_mass = {i: masses[i].item() for i in range(k)}
    particle_group = dict(leaf_group)
    active = list(range(k))
    M = masses.sum().item()

    merge_pairs = torch.zeros((k - 1, 2), dtype=torch.long)
    merge_times = torch.zeros(k - 1, dtype=torch.float64)
    heights = torch.zeros(n_nodes, dtype=torch.float64)
    parent = torch.full((n_nodes,), -1, dtype=torch.long)

    t = 0.0
    next_id = k

    for step in range(k - 1):
        # Find eligible pairs: same group
        # Group particles by group
        groups = {}
        for p in active:
            g = particle_group[p]
            groups.setdefault(g, []).append(p)

        # Collect eligible pairs and their rates
        eligible = []
        for g, members in groups.items():
            if len(members) < 2:
                continue
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    a, b = members[i], members[j]
                    rate = particle_mass[a] + particle_mass[b]
                    eligible.append((rate, a, b))

        if not eligible:
            # This shouldn't happen if the contracted tree is valid
            raise RuntimeError(
                f"No eligible pairs at step {step}, "
                f"active={active}, groups={groups}")

        total_rate = sum(r for r, _, _ in eligible)
        dt = random.expovariate(total_rate)
        t += dt

        # Pick pair
        u = random.random() * total_rate
        cum = 0.0
        chosen_a, chosen_b = None, None
        for rate, a, b in eligible:
            cum += rate
            if cum >= u:
                chosen_a, chosen_b = a, b
                break

        new_node = next_id
        new_mass = particle_mass[chosen_a] + particle_mass[chosen_b]
        new_group = particle_group[chosen_a]  # same group

        # Record merge
        merge_pairs[step, 0] = chosen_a
        merge_pairs[step, 1] = chosen_b
        merge_times[step] = t
        heights[new_node] = t
        parent[chosen_a] = new_node
        parent[chosen_b] = new_node

        # Update active list
        active = [p for p in active if p not in (chosen_a, chosen_b)]
        active.append(new_node)
        particle_mass[new_node] = new_mass
        particle_group[new_node] = new_group
        next_id += 1

        # Check if this group is now fully merged (single particle)
        # If so, move it to the parent group
        _promote_if_singleton(new_node, particle_group,
                              active, contracted_parent)

    # Edge lengths
    edge_lengths = torch.zeros(n_nodes - 1, dtype=torch.float64)
    for j in range(n_nodes - 1):
        p = parent[j].item()
        if p >= 0:
            edge_lengths[j] = heights[p] - heights[j]

    return {
        "merge_pairs": merge_pairs,
        "merge_times": merge_times,
        "edge_lengths": edge_lengths,
        "heights": heights,
        "masses": masses,
        "parent": parent,
    }


def _promote_if_singleton(particle, particle_group, active,
                          contracted_parent):
    """If the particle's group has only this one particle in it,
    promote it to the parent group. Repeat until the group has
    multiple particles or we reach the root."""
    while True:
        g = particle_group[particle]
        # Count particles in this group
        count = sum(1 for p in active if particle_group.get(p) == g)
        if count > 1:
            break
        # Promote to parent group
        if g not in contracted_parent:
            break  # at root
        particle_group[particle] = contracted_parent[g]