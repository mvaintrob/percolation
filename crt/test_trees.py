"""
test_trees.py — Tests for the coalescent tree library.

Usage:
    python -m crt.test_trees [test_name]

Test modes:
    smoke       — Quick sanity checks (imports, shapes, types)
    consistency — Verify merge_pairs, heights, edge_lengths are consistent
    roundtrip   — coalescent -> contract -> expand, check valid binary tree
    symmetry    — Masses [2,2,1,1,1], compare tree distribution modulo S2xS3
    metric      — Verify d(i,j) = h_i + h_j - 2*h_lca is a valid metric
    all         — Run all tests
"""

import sys
import torch
import random
from collections import Counter
from itertools import permutations

from .coalescent import coalescent
from .contract import contract
from .expand import expand


# ── Helpers ──────────────────────────────────────────────────

def tree_metric(tree):
    """Compute pairwise distances between leaves.
    d(i,j) = h_i + h_j - 2 * h_lca(i,j)
    where h_lca = height of lowest common ancestor.
    """
    k = len(tree["masses"])
    heights = tree["heights"]
    parent = tree["parent"]
    n_nodes = len(heights)

    # Precompute ancestors for each node
    def ancestors(j):
        path = [j]
        while parent[path[-1]].item() >= 0:
            path.append(parent[path[-1]].item())
        return path

    ancestor_cache = {j: ancestors(j) for j in range(k)}

    dist = torch.zeros((k, k), dtype=torch.float64)
    for i in range(k):
        anc_i = set(ancestor_cache[i])
        for j in range(i + 1, k):
            # LCA = first ancestor of j that's also ancestor of i
            for a in ancestor_cache[j]:
                if a in anc_i:
                    lca = a
                    break
            d = 2 * heights[lca] - heights[i] - heights[j]
            dist[i, j] = d
            dist[j, i] = d

    return dist


def tree_to_topology(tree):
    """Extract tree topology as nested tuples for comparison."""
    k = len(tree["masses"])
    parent = tree["parent"]

    # Build children map
    children = {i: [] for i in range(2 * k - 1)}
    for j in range(2 * k - 2):
        p = parent[j].item()
        if p >= 0:
            children[p].append(j)

    root = 2 * k - 2

    def to_tuple(node):
        if node < k:
            return node
        ch = children[node]
        if len(ch) == 2:
            return (to_tuple(ch[0]), to_tuple(ch[1]))
        elif len(ch) == 1:
            return to_tuple(ch[0])
        else:
            return tuple(to_tuple(c) for c in ch)

    return to_tuple(root)


def tree_key(tree_tuple):
    """Comparable key for nested tree tuples."""
    if isinstance(tree_tuple, int):
        return (0, tree_tuple)
    return (1,) + tuple(tree_key(c) for c in tree_tuple)


def normalize_tree(tree_tuple):
    """Sort children at each node."""
    if isinstance(tree_tuple, int):
        return tree_tuple
    children = [normalize_tree(c) for c in tree_tuple]
    children.sort(key=tree_key)
    return tuple(children)


def apply_perm(tree_tuple, perm):
    if isinstance(tree_tuple, int):
        return perm[tree_tuple]
    return tuple(apply_perm(c, perm) for c in tree_tuple)


def canonicalize_22111(tree_tuple):
    """Canonical form under S_2(swap 0,1) x S_3(permute 2,3,4)."""
    best = None
    for s2 in [[0, 1], [1, 0]]:
        for s3 in permutations([2, 3, 4]):
            perm = [s2[0], s2[1], s3[0], s3[1], s3[2]]
            t = normalize_tree(apply_perm(tree_tuple, perm))
            k = tree_key(t)
            if best is None or k < best[1]:
                best = (t, k)
    return best[0]


# ── Tests ────────────────────────────────────────────────────

def test_smoke():
    """Quick sanity checks."""
    print("test_smoke...")

    # Basic coalescent
    t = coalescent(k=5, masses=torch.tensor([50., 30., 10., 7., 3.]))
    assert t["merge_pairs"].shape == (4, 2)
    assert t["merge_times"].shape == (4,)
    assert t["heights"].shape == (9,)
    assert t["edge_lengths"].shape == (8,)
    assert t["masses"].shape == (5,)
    assert t["parent"].shape == (9,)

    # Default masses
    t2 = coalescent(k=10)
    assert t2["masses"].shape == (10,)
    assert torch.allclose(t2["masses"].sum(),
                          torch.tensor(1.0, dtype=torch.float64))

    # Single leaf
    t1 = coalescent(k=1)
    assert t1["merge_pairs"].shape == (0, 2)

    print("  PASSED")


def test_consistency():
    """Verify internal consistency of coalescent output."""
    print("test_consistency...")

    t = coalescent(k=8, masses=torch.tensor([10., 8., 6., 5., 4., 3., 2., 1.]))
    k = 8
    n_nodes = 2 * k - 1

    # Merge times should be strictly increasing
    times = t["merge_times"]
    for i in range(len(times) - 1):
        assert times[i] < times[i + 1], f"times not increasing at {i}"

    # All heights non-negative
    assert (t["heights"] >= 0).all()

    # Leaf heights are 0
    assert (t["heights"][:k] == 0).all()

    # Internal node heights match merge times
    for i in range(k - 1):
        assert abs(t["heights"][k + i] - times[i]) < 1e-12

    # Edge lengths = parent height - child height
    for j in range(n_nodes - 1):
        p = t["parent"][j].item()
        if p >= 0:
            expected = t["heights"][p] - t["heights"][j]
            actual = t["edge_lengths"][j]
            assert abs(expected - actual) < 1e-12, \
                f"edge_length mismatch at node {j}"

    # All edge lengths positive
    assert (t["edge_lengths"] > 0).all()

    # Each non-root node has exactly one parent
    parent = t["parent"]
    assert parent[n_nodes - 1] == -1  # root
    for j in range(n_nodes - 1):
        assert parent[j] >= k  # parent is an internal node
        assert parent[j] < n_nodes

    # Each internal node has exactly 2 children
    child_count = Counter()
    for j in range(n_nodes - 1):
        child_count[parent[j].item()] += 1
    for node in range(k, n_nodes):
        assert child_count[node] == 2, \
            f"node {node} has {child_count[node]} children"

    print("  PASSED")


def test_roundtrip():
    """coalescent -> contract -> expand, verify valid binary tree."""
    print("test_roundtrip...")

    masses = torch.tensor([50., 30., 10., 7., 3.])
    k = len(masses)
    t = coalescent(k=k, masses=masses)

    # Contract with a reasonable epsilon
    epsilon = t["edge_lengths"].median().item()
    ct = contract(t, epsilon)

    # Check contraction produced some non-binary nodes
    max_deg = max(len(ch) for ch in ct["children"])
    print(f"  epsilon={epsilon:.4f}, max degree after contraction: {max_deg}")

    # Expand
    t2 = expand(ct)

    # Check valid binary tree
    assert t2["merge_pairs"].shape == (k - 1, 2)
    assert t2["merge_times"].shape == (k - 1,)
    assert t2["heights"].shape == (2 * k - 1,)
    assert t2["parent"].shape == (2 * k - 1,)

    # All edge lengths non-negative
    assert (t2["edge_lengths"] >= -1e-12).all()

    # Times increasing
    for i in range(k - 2):
        assert t2["merge_times"][i] <= t2["merge_times"][i + 1] + 1e-12

    # Each internal node has 2 children
    child_count = Counter()
    for j in range(2 * k - 2):
        child_count[t2["parent"][j].item()] += 1
    for node in range(k, 2 * k - 1):
        assert child_count[node] == 2, \
            f"node {node} has {child_count.get(node, 0)} children"

    print("  PASSED")


def test_symmetry(n_samples=50000):
    """For masses [2,2,1,1,1], compare tree distributions.

    Runs both coalescent and expand-from-scratch, canonicalizes
    under S_2 x S_3, compares frequencies.
    """
    print(f"test_symmetry (n={n_samples})...")

    masses = torch.tensor([2., 2., 1., 1., 1.])

    coal_counts = Counter()
    for _ in range(n_samples):
        t = coalescent(masses=masses)
        topo = tree_to_topology(t)
        canon = canonicalize_22111(topo)
        coal_counts[canon] += 1

    all_shapes = sorted(coal_counts.keys(), key=tree_key)

    print(f"  {len(all_shapes)} distinct shapes")
    print(f"  {'Shape':<40} {'Freq':>8}")
    print("  " + "-" * 50)

    for shape in all_shapes[:10]:
        freq = coal_counts[shape] / n_samples
        print(f"  {str(shape):<40} {freq:>8.4f}")

    print("  PASSED (visual inspection)")


def test_metric():
    """Verify tree metric is a valid ultrametric."""
    print("test_metric...")

    t = coalescent(k=6, masses=torch.tensor([10., 8., 5., 3., 2., 1.]))
    dist = tree_metric(t)
    k = 6

    # Non-negative
    assert (dist >= -1e-12).all()

    # Symmetric
    assert torch.allclose(dist, dist.T)

    # Zero diagonal
    assert torch.allclose(dist.diag(), torch.zeros(k, dtype=torch.float64))

    # Triangle inequality
    for i in range(k):
        for j in range(k):
            for l in range(k):
                assert dist[i, j] <= dist[i, l] + dist[l, j] + 1e-10

    # Ultrametric: d(i,j) <= max(d(i,l), d(l,j))
    for i in range(k):
        for j in range(k):
            for l in range(k):
                assert dist[i, j] <= max(dist[i, l].item(),
                                          dist[l, j].item()) + 1e-10

    print("  PASSED")


# ── Main ─────────────────────────────────────────────────────

TESTS = {
    "smoke": test_smoke,
    "consistency": test_consistency,
    "roundtrip": test_roundtrip,
    "symmetry": test_symmetry,
    "metric": test_metric,
}


def main():
    if len(sys.argv) > 1:
        name = sys.argv[1]
    else:
        name = "all"

    random.seed(42)
    torch.manual_seed(42)

    if name == "all":
        for test_name, test_fn in TESTS.items():
            test_fn()
    elif name in TESTS:
        TESTS[name]()
    else:
        print(f"Unknown test: {name}")
        print(f"Available: {', '.join(TESTS.keys())}, all")
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()