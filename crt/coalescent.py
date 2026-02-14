"""
coalescent.py — Forward additive coalescent on weighted leaves.

Gillespie exact simulation: O(k^2).
Produces a binary tree with edge lengths and merge ordering.
"""

import torch
import random
import math


def coalescent(k=None, masses=None):
    """Run the forward additive coalescent.

    Args:
        k: number of leaves (default: len(masses))
        masses: (k,) tensor of leaf masses (default: uniform 1/k)

    Returns dict of torch tensors:
        merge_pairs: (k-1, 2) int — merged cluster ids per step
        merge_times: (k-1,) float — absolute time of each merge
        edge_lengths: (2k-2,) float — edge length for each non-root node
        heights: (2k-1,) float — height of each node
        masses: (k,) float — leaf masses
    """
    # Handle defaults
    if masses is not None:
        masses = torch.as_tensor(masses, dtype=torch.float64)
        if k is None:
            k = len(masses)
        else:
            assert k == len(masses)
    else:
        assert k is not None, "Must provide k or masses"
        masses = torch.full((k,), 1.0 / k, dtype=torch.float64)

    if k == 1:
        return {
            "merge_pairs": torch.zeros((0, 2), dtype=torch.long),
            "merge_times": torch.zeros(0, dtype=torch.float64),
            "edge_lengths": torch.zeros(0, dtype=torch.float64),
            "heights": torch.zeros(1, dtype=torch.float64),
            "masses": masses,
        }

    # Node indexing: 0..k-1 are leaves, k+i is the internal node
    # created at merge step i. Root = 2k-2.
    n_nodes = 2 * k - 1

    # Active clusters: maps cluster_id -> (mass, node_id)
    # Initially each leaf is its own cluster.
    cluster_mass = {}
    cluster_node = {}
    active = []
    for i in range(k):
        cluster_mass[i] = masses[i].item()
        cluster_node[i] = i
        active.append(i)

    M = masses.sum().item()  # total mass, conserved

    merge_pairs = torch.zeros((k - 1, 2), dtype=torch.long)
    merge_times = torch.zeros(k - 1, dtype=torch.float64)
    heights = torch.zeros(n_nodes, dtype=torch.float64)
    # heights[0..k-1] = 0 (leaves)

    t = 0.0
    next_cluster_id = k

    for step in range(k - 1):
        j = len(active)
        total_rate = (j - 1) * M

        # Sample waiting time
        dt = random.expovariate(total_rate)
        t += dt

        # Pick merging pair: sample a size-biased, then b uniform from rest
        # This gives pair (a,b) with prob (m_a + m_b) / ((j-1)*M)
        u = random.random() * M
        cum = 0.0
        a_idx = 0
        for i, cid in enumerate(active):
            cum += cluster_mass[cid]
            if cum >= u:
                a_idx = i
                break

        b_idx = random.randrange(j - 1)
        if b_idx >= a_idx:
            b_idx += 1

        a_cid = active[a_idx]
        b_cid = active[b_idx]

        a_node = cluster_node[a_cid]
        b_node = cluster_node[b_cid]
        new_node = k + step  # internal node id
        new_mass = cluster_mass[a_cid] + cluster_mass[b_cid]

        # Record
        merge_pairs[step, 0] = a_node
        merge_pairs[step, 1] = b_node
        merge_times[step] = t
        heights[new_node] = t

        # Update active list
        cluster_mass[next_cluster_id] = new_mass
        cluster_node[next_cluster_id] = new_node

        hi, lo = max(a_idx, b_idx), min(a_idx, b_idx)
        active.pop(hi)
        active.pop(lo)
        active.append(next_cluster_id)
        next_cluster_id += 1

    # Compute edge_lengths and parent array
    # parent[j] = the internal node that j merges into
    parent = torch.full((n_nodes,), -1, dtype=torch.long)
    for step in range(k - 1):
        internal_node = k + step
        parent[merge_pairs[step, 0]] = internal_node
        parent[merge_pairs[step, 1]] = internal_node

    # edge_lengths[j] = heights[parent[j]] - heights[j] for j != root
    edge_lengths = torch.zeros(n_nodes, dtype=torch.float64)
    for j in range(n_nodes):
        if parent[j] >= 0:
            edge_lengths[j] = heights[parent[j].item()] - heights[j]

    # Trim: root has no edge, so return 2k-2 edge lengths
    # (indexed by non-root nodes 0..2k-3)
    return {
        "merge_pairs": merge_pairs,
        "merge_times": merge_times,
        "edge_lengths": edge_lengths[:n_nodes - 1],  # exclude root
        "heights": heights,
        "masses": masses,
        "parent": parent,
    }
