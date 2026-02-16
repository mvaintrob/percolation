"""
coalescent.py — Forward additive coalescent on weighted leaves.

Gillespie exact simulation using a Fenwick tree for O(k log k) total.
Produces a binary tree with edge lengths and merge ordering.
"""

import torch
import random
import math


class _FenwickTree:
    """Binary indexed tree supporting prefix sums and point updates in O(log n).

    Indices are 0-based externally, 1-based internally.
    """

    def __init__(self, n):
        self.n = n
        self.tree = [0.0] * (n + 1)

    def update(self, i, delta):
        """Add delta to position i."""
        i += 1  # to 1-based
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)

    def prefix_sum(self, i):
        """Sum of elements 0..i (inclusive)."""
        s = 0.0
        i += 1  # to 1-based
        while i > 0:
            s += self.tree[i]
            i -= i & (-i)
        return s

    def total(self):
        """Sum of all elements."""
        return self.prefix_sum(self.n - 1)

    def find(self, target):
        """Find smallest i such that prefix_sum(i) >= target. O(log n)."""
        pos = 0
        bit_mask = 1
        while bit_mask <= self.n:
            bit_mask <<= 1
        bit_mask >>= 1

        cum = 0.0
        while bit_mask > 0:
            next_pos = pos + bit_mask
            if next_pos <= self.n and cum + self.tree[next_pos] < target:
                cum += self.tree[next_pos]
                pos = next_pos
            bit_mask >>= 1

        return pos  # 0-based index


def coalescent(k=None, masses=None):
    """Run the forward additive coalescent.

    Uses a Fenwick tree for O(log k) size-biased sampling per step,
    giving O(k log k) total time.

    Args:
        k: number of leaves (default: len(masses))
        masses: (k,) tensor of leaf masses (default: uniform 1/k)

    Returns dict of torch tensors:
        merge_pairs: (k-1, 2) int — merged cluster ids per step
        merge_times: (k-1,) float — absolute time of each merge
        edge_lengths: (2k-2,) float — edge length for each non-root node
        heights: (2k-1,) float — height of each node
        masses: (k,) float — leaf masses
        parent: (2k-1,) int — parent node (-1 for root)
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
            "parent": torch.full((1,), -1, dtype=torch.long),
        }

    n_nodes = 2 * k - 1

    # We use slots 0..2k-3 in the Fenwick tree: one slot per possible
    # active cluster. Initially slots 0..k-1 hold the leaf masses.
    # When clusters a and b merge, we zero out both slots and write
    # the combined mass into a new slot.

    max_slots = n_nodes  # more than enough
    fw = _FenwickTree(max_slots)

    # slot_node[i] = the tree node id for cluster in slot i
    slot_node = [0] * max_slots
    # active_slots = set of slots currently holding active clusters
    active_slots = []
    # slot_index[slot] = position in active_slots (for O(1) removal)
    slot_pos = {}

    for i in range(k):
        fw.update(i, masses[i].item())
        slot_node[i] = i
        slot_pos[i] = i
        active_slots.append(i)

    M = masses.sum().item()
    next_slot = k

    merge_pairs = torch.zeros((k - 1, 2), dtype=torch.long)
    merge_times = torch.zeros(k - 1, dtype=torch.float64)
    heights = torch.zeros(n_nodes, dtype=torch.float64)

    t = 0.0

    for step in range(k - 1):
        j = len(active_slots)
        total_rate = (j - 1) * M

        # Sample waiting time
        dt = random.expovariate(total_rate)
        t += dt

        # Size-biased pick of a: find slot with prefix_sum >= u
        u = random.random() * M
        a_slot = fw.find(u)

        # Uniform pick of b from remaining j-1 active slots
        # Find a's position in active_slots
        a_pos = slot_pos[a_slot]
        b_pos = random.randrange(j - 1)
        if b_pos >= a_pos:
            b_pos += 1
        b_slot = active_slots[b_pos]

        a_node = slot_node[a_slot]
        b_node = slot_node[b_slot]
        new_node = k + step

        # Get masses before zeroing
        # (prefix_sum(i) - prefix_sum(i-1) gives the value at i)
        a_mass = fw.prefix_sum(a_slot)
        if a_slot > 0:
            a_mass -= fw.prefix_sum(a_slot - 1)
        b_mass = fw.prefix_sum(b_slot)
        if b_slot > 0:
            b_mass -= fw.prefix_sum(b_slot - 1)
        new_mass = a_mass + b_mass

        # Record merge
        merge_pairs[step, 0] = a_node
        merge_pairs[step, 1] = b_node
        merge_times[step] = t
        heights[new_node] = t

        # Zero out old slots, write new slot
        fw.update(a_slot, -a_mass)
        fw.update(b_slot, -b_mass)
        fw.update(next_slot, new_mass)
        slot_node[next_slot] = new_node

        # Update active_slots: remove a and b, add new
        # Remove higher index first to avoid shifting
        hi_pos = max(a_pos, b_pos)
        lo_pos = min(a_pos, b_pos)

        # Swap-and-pop for O(1) removal
        # Remove hi first
        last = active_slots[-1]
        active_slots[hi_pos] = last
        slot_pos[last] = hi_pos
        active_slots.pop()

        # Now remove lo (list is one shorter)
        if lo_pos < len(active_slots):
            last = active_slots[-1]
            active_slots[lo_pos] = last
            slot_pos[last] = lo_pos
            active_slots.pop()
        else:
            active_slots.pop()

        # Clean up stale entries
        if a_slot in slot_pos:
            del slot_pos[a_slot]
        if b_slot in slot_pos:
            del slot_pos[b_slot]

        # Add new slot
        active_slots.append(next_slot)
        slot_pos[next_slot] = len(active_slots) - 1

        next_slot += 1

    # Compute parent array and edge lengths
    parent = torch.full((n_nodes,), -1, dtype=torch.long)
    for step in range(k - 1):
        internal_node = k + step
        parent[merge_pairs[step, 0]] = internal_node
        parent[merge_pairs[step, 1]] = internal_node

    edge_lengths = torch.zeros(n_nodes, dtype=torch.float64)
    for j in range(n_nodes):
        if parent[j] >= 0:
            edge_lengths[j] = heights[parent[j].item()] - heights[j]

    return {
        "merge_pairs": merge_pairs,
        "merge_times": merge_times,
        "edge_lengths": edge_lengths[:n_nodes - 1],
        "heights": heights,
        "masses": masses,
        "parent": parent,
    }
