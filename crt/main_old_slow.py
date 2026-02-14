"""
Coalescent inverse (refinement) process on weighted trees.

Given a non-binary tree with masses on leaves, implements:
1. Splitting formula: P({S1,S2}) ∝ w_S1^{|S1|-1} * w_S2^{|S2|-1}
2. Vertex selection: exact order-statistic formula
3. Edge length: sampled from max of k-1 independent exponentials, 
   conditioned on < epsilon
4. Full iterative refinement

The splitting formula is exact for the additive coalescent.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, FrozenSet
from scipy import integrate
import random
import math
from collections import Counter


# ── Tree data structure ──────────────────────────────────────

@dataclass
class Node:
    """A node in the tree. Leaves carry mass directly; 
    internal nodes carry the sum of descendant masses."""
    mass: float
    children: list = field(default_factory=list)
    parent: Optional['Node'] = None
    edge_length: float = 0.0
    label: str = ""
    
    @property
    def is_leaf(self):
        return len(self.children) == 0
    
    @property
    def degree(self):
        return len(self.children)
    
    def recompute_mass(self):
        if self.is_leaf:
            return self.mass
        self.mass = sum(c.recompute_mass() for c in self.children)
        return self.mass


def print_tree(node, indent=0):
    prefix = "  " * indent
    if node.is_leaf:
        print(f"{prefix}Leaf({node.label}, mass={node.mass:.2f}, "
              f"edge_len={node.edge_length:.4f})")
    else:
        deg = node.degree
        print(f"{prefix}Node({node.label}, mass={node.mass:.2f}, "
              f"deg={deg}, edge_len={node.edge_length:.4f})")
        for c in node.children:
            print_tree(c, indent + 1)


def make_star_tree(masses, labels=None):
    """Create a star tree: one root with k leaf children."""
    if labels is None:
        labels = [f"L{i}({m})" for i, m in enumerate(masses)]
    children = []
    for m, lab in zip(masses, labels):
        leaf = Node(mass=m, label=lab)
        children.append(leaf)
    root = Node(mass=sum(masses), children=children, label="root")
    for c in children:
        c.parent = root
    return root


# ── Splitting distribution ───────────────────────────────────

def log_split_weight(child_masses, S1_indices):
    """Log probability weight of partition {S1, S2}.
    P({S1, S2}) ∝ w_S1^{|S1|-1} * w_S2^{|S2|-1}
    """
    k = len(child_masses)
    S2_indices = set(range(k)) - set(S1_indices)
    if len(S1_indices) == 0 or len(S2_indices) == 0:
        return -float('inf')
    w1 = sum(child_masses[i] for i in S1_indices)
    w2 = sum(child_masses[i] for i in S2_indices)
    k1 = len(S1_indices)
    k2 = len(S2_indices)
    return (k1 - 1) * math.log(w1) + (k2 - 1) * math.log(w2)


def enumerate_splits(k):
    """Enumerate all unordered bipartitions {S1, S2} of [k].
    Fix element 0 in S1 to avoid double-counting."""
    splits = []
    for mask in range(0, 2**(k-1)):
        S1 = {0}
        S2 = set()
        for bit in range(k - 1):
            if mask & (1 << bit):
                S1.add(bit + 1)
            else:
                S2.add(bit + 1)
        if len(S2) > 0:
            splits.append((frozenset(S1), frozenset(S2)))
    return splits


def sample_split(child_masses) -> Tuple[FrozenSet[int], FrozenSet[int]]:
    """Sample a binary split of children according to the 
    additive coalescent splitting distribution."""
    k = len(child_masses)
    if k == 2:
        return frozenset({0}), frozenset({1})
    
    splits = enumerate_splits(k)
    log_weights = [log_split_weight(child_masses, S1) for S1, S2 in splits]
    max_lw = max(log_weights)
    weights = [math.exp(lw - max_lw) for lw in log_weights]
    total = sum(weights)
    probs = [w / total for w in weights]
    idx = random.choices(range(len(splits)), weights=probs, k=1)[0]
    return splits[idx]


# ── Vertex selection ─────────────────────────────────────────

def longest_edge_cdf(x, M, k):
    """CDF of longest internal edge in coalescent on k particles, 
    total mass M.
    
    Internal edges: E_j ~ Exp(j*M), j=1,...,k-1
    P(max <= x) = prod_{j=1}^{k-1} (1 - exp(-j*M*x))
    """
    if x <= 0:
        return 0.0
    cdf = 1.0
    for j in range(1, k):
        cdf *= (1 - math.exp(-j * M * x))
    return cdf


def longest_edge_pdf(x, M, k):
    """PDF of longest internal edge.
    d/dx prod_{j=1}^{k-1} (1 - exp(-j*M*x))
    """
    if x <= 0:
        return 0.0
    factors = [(1 - math.exp(-j * M * x)) for j in range(1, k)]
    pdf = 0.0
    for idx in range(k - 1):
        j = idx + 1
        deriv = j * M * math.exp(-j * M * x)
        product = 1.0
        for other_idx in range(k - 1):
            if other_idx != idx:
                product *= factors[other_idx]
        pdf += deriv * product
    return pdf


def vertex_selection_probs(nodes_info, epsilon):
    """Exact vertex selection probabilities.
    
    P(v first) = int_0^eps f_v(x) prod_{u!=v} F_u(x) dx / Z
    where Z = prod_u F_u(eps)
    
    Args:
        nodes_info: list of (node, M, k) for each non-binary node
        epsilon: current resolution
    Returns:
        list of probabilities, one per node
    """
    n = len(nodes_info)
    if n == 0:
        return []
    if n == 1:
        return [1.0]
    
    params = [(M, k) for _, M, k in nodes_info]
    
    denom = 1.0
    for M, k in params:
        denom *= longest_edge_cdf(epsilon, M, k)
    
    if denom < 1e-300:
        return [1.0 / n] * n
    
    probs = []
    for v_idx in range(n):
        M_v, k_v = params[v_idx]
        
        def integrand(x, vi=v_idx):
            val = longest_edge_pdf(x, params[vi][0], params[vi][1])
            for u_idx in range(n):
                if u_idx != vi:
                    M_u, k_u = params[u_idx]
                    val *= longest_edge_cdf(x, M_u, k_u)
            return val
        
        result, _ = integrate.quad(integrand, 0, epsilon)
        probs.append(result / denom)
    
    # Normalize to fix numerical drift
    total = sum(probs)
    probs = [p / total for p in probs]
    return probs


# ── Edge length sampling ─────────────────────────────────────

def sample_longest_edge(M, k, epsilon):
    """Sample the longest internal edge of a coalescent on k particles,
    total mass M, conditioned on all edges < epsilon.
    
    This is max(E_1, ..., E_{k-1}) where E_j ~ Exp(j*M),
    conditioned on max < epsilon.
    
    Method: sample each E_j from Exp(j*M) truncated to [0, epsilon],
    then return the max. But this isn't quite right because we condition 
    on ALL being < epsilon, not each individually...
    
    Actually, since E_j are independent, conditioning max < epsilon 
    IS the same as conditioning each E_j < epsilon independently.
    So: sample each from truncated Exp(j*M) on [0, epsilon], return max.
    """
    edges = []
    for j in range(1, k):
        rate = j * M
        # Truncated exponential: F(x) = (1-e^{-rate*x})/(1-e^{-rate*eps})
        # Inverse: x = -log(1 - u*(1-e^{-rate*eps})) / rate
        u = random.random()
        x = -math.log(1 - u * (1 - math.exp(-rate * epsilon))) / rate
        edges.append(x)
    return max(edges), edges


# ── Full refinement step ─────────────────────────────────────

def get_nonbinary_nodes(root):
    """Collect all non-binary internal nodes with their M and k."""
    nodes = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.degree > 2:
            nodes.append((node, node.mass, node.degree))
        for c in node.children:
            if not c.is_leaf:
                stack.append(c)
    return nodes


def refine_step(root, epsilon):
    """One step of the inverse refinement process.
    
    1. Find all non-binary nodes
    2. Select one using exact order-statistic probabilities
    3. Sample the split: P({S1,S2}) ∝ w_S1^{|S1|-1} * w_S2^{|S2|-1}
    4. Sample edge length from conditioned max-of-exponentials
    5. Insert new internal node
    
    Returns (selected_node, S1, S2, edge_length) or None if fully binary.
    """
    candidates = get_nonbinary_nodes(root)
    if not candidates:
        return None
    
    # Step 1: Vertex selection
    probs = vertex_selection_probs(candidates, epsilon)
    v_idx = random.choices(range(len(candidates)), weights=probs, k=1)[0]
    v, M_v, k_v = candidates[v_idx]
    
    # Step 2: Sample split
    child_masses = [c.mass for c in v.children]
    S1, S2 = sample_split(child_masses)
    
    # Step 3: Sample edge length (longest internal edge, conditioned < eps)
    edge_len, _ = sample_longest_edge(M_v, k_v, epsilon)
    
    # Step 4: Insert new internal node
    # One side gets grouped under a new internal node.
    # Never wrap a singleton — always wrap the multi-element side.
    children_S1 = [v.children[i] for i in sorted(S1)]
    children_S2 = [v.children[i] for i in sorted(S2)]
    w1 = sum(c.mass for c in children_S1)
    w2 = sum(c.mass for c in children_S2)
    
    if len(children_S1) == 1 and len(children_S2) == 1:
        # Both singletons: this split reduces degree by 1.
        # Group S2 under new node (arbitrary).
        new_node = Node(mass=w2, children=children_S2,
                       parent=v, edge_length=edge_len,
                       label=f"int_{w2:.0f}")
        for c in children_S2:
            c.parent = new_node
        v.children = children_S1 + [new_node]
    elif len(children_S2) == 1:
        # S2 is singleton: wrap S1 (the multi-element side)
        new_node = Node(mass=w1, children=children_S1,
                       parent=v, edge_length=edge_len,
                       label=f"int_{w1:.0f}")
        for c in children_S1:
            c.parent = new_node
        v.children = [new_node] + children_S2
    else:
        # S1 is singleton or both have multiple: wrap S2
        new_node = Node(mass=w2, children=children_S2,
                       parent=v, edge_length=edge_len,
                       label=f"int_{w2:.0f}")
        for c in children_S2:
            c.parent = new_node
        v.children = children_S1 + [new_node]
    
    return v, S1, S2, edge_len


def refine_fully(root, epsilon):
    """Refine until all nodes are binary."""
    step = 0
    while True:
        result = refine_step(root, epsilon)
        if result is None:
            break
        step += 1
        v, S1, S2, elen = result
        child_masses = lambda S: [root.mass for _ in S]  # placeholder
        print(f"Step {step}: Split node (M={v.mass:.0f}, deg={v.degree+1}→"
              f"{v.degree}) into groups of size {len(S1)} and {len(S2)}, "
              f"edge_len={elen:.4f}")
    return step


# ── Monte Carlo validation ───────────────────────────────────

def run_additive_coalescent(masses):
    """Run the full additive coalescent on given masses.
    Returns: list of (merge_time, S1_members, S2_members, edge_length)
    in chronological order.
    """
    k = len(masses)
    particles = list(range(k))
    particle_masses = {i: masses[i] for i in range(k)}
    particle_members = {i: frozenset({i}) for i in range(k)}
    particle_last_merge = {i: 0.0 for i in range(k)}
    
    mergers = []
    t = 0
    next_id = k
    
    while len(particles) > 1:
        j = len(particles)
        total_mass = sum(particle_masses[p] for p in particles)
        
        # All pair rates
        rates = []
        for a_idx in range(j):
            for b_idx in range(a_idx + 1, j):
                a, b = particles[a_idx], particles[b_idx]
                rate = particle_masses[a] + particle_masses[b]
                rates.append((rate, a, b))
        
        total_rate = sum(r for r, _, _ in rates)
        dt = random.expovariate(total_rate)
        t += dt
        
        # Pick pair
        u = random.random() * total_rate
        cum = 0
        chosen = None
        for rate, a, b in rates:
            cum += rate
            if cum >= u:
                chosen = (a, b)
                break
        
        a, b = chosen
        new_mass = particle_masses[a] + particle_masses[b]
        new_members = particle_members[a] | particle_members[b]
        
        # Edge length = time since this node was last involved in a merge
        # For root split: this is the time between the last two merges
        edge_a = t - particle_last_merge[a]
        edge_b = t - particle_last_merge[b]
        
        mergers.append({
            'time': t,
            'dt': dt,
            'S1': particle_members[a],
            'S2': particle_members[b],
            'mass': new_mass,
            'edge_length_a': edge_a,
            'edge_length_b': edge_b,
        })
        
        particle_masses[next_id] = new_mass
        particle_members[next_id] = new_members
        particle_last_merge[next_id] = t
        
        particles = [p for p in particles if p not in (a, b)] + [next_id]
        next_id += 1
    
    return mergers


def validate_splitting(masses, n_samples=100000):
    """Validate splitting formula against full coalescent MC."""
    k = len(masses)
    
    print(f"\nValidation: masses={masses}")
    print(f"Running {n_samples} coalescent samples...\n")
    
    # Count root splits from full coalescent (keyed by mass tuples)
    root_split_counts = Counter()
    for _ in range(n_samples):
        mergers = run_additive_coalescent(masses)
        last = mergers[-1]
        S1, S2 = last['S1'], last['S2']
        m1 = tuple(sorted(masses[i] for i in S1))
        m2 = tuple(sorted(masses[i] for i in S2))
        key = (min(m1, m2), max(m1, m2))
        root_split_counts[key] += 1
    
    # Compute exact probabilities (same key format)
    splits = enumerate_splits(k)
    exact_probs = {}
    log_weights = [log_split_weight(masses, S1) for S1, S2 in splits]
    max_lw = max(log_weights)
    ws = [math.exp(lw - max_lw) for lw in log_weights]
    total_w = sum(ws)
    
    for (S1, S2), w in zip(splits, ws):
        m1 = tuple(sorted(masses[i] for i in S1))
        m2 = tuple(sorted(masses[i] for i in S2))
        key = (min(m1, m2), max(m1, m2))
        exact_probs[key] = exact_probs.get(key, 0) + w / total_w
    
    print(f"{'S1 masses':>25} {'S2 masses':>25} {'MC':>8} {'Exact':>8} {'Err':>8}")
    print("-" * 80)
    
    for key in sorted(root_split_counts.keys(),
                       key=lambda k: root_split_counts[k], reverse=True)[:10]:
        mc = root_split_counts[key] / n_samples
        exact = exact_probs.get(key, 0)
        err = abs(mc - exact)
        print(f"{str(key[0]):>25} {str(key[1]):>25} {mc:>8.4f} {exact:>8.4f} {err:>8.4f}")


# ── Demos ─────────────────────────────────────────────────────

def demo_full_refinement():
    """Demo: start from a star tree, refine to binary."""
    masses = [50, 30, 10, 7, 3]
    epsilon = 0.05
    
    print("=" * 65)
    print("FULL REFINEMENT: Star → Binary")
    print(f"Masses: {masses}, epsilon={epsilon}")
    print("=" * 65)
    
    root = make_star_tree(masses)
    print("\nInitial tree:")
    print_tree(root)
    
    step = 0
    while True:
        result = refine_step(root, epsilon)
        if result is None:
            print("\nTree is fully binary.")
            break
        step += 1
        v, S1, S2, elen = result
        s1_masses = sorted([root.children[0].mass if False else 0])  # placeholder
        print(f"\n--- Step {step}: Resolved node M={v.mass:.0f} ---")
        print(f"  Split: sizes ({len(S1)}, {len(S2)}), edge_len={elen:.4f}")
        print_tree(root)


def demo_vertex_selection():
    """Demo: show which node gets selected at different epsilon."""
    print("\n" + "=" * 65)
    print("VERTEX SELECTION PROBABILITIES")
    print("=" * 65)
    
    # Build a tree with two non-binary nodes
    # Root has 3 children, one of which has 4 children
    inner_leaves = [Node(mass=m, label=f"L({m})") for m in [20, 15, 8, 7]]
    inner = Node(mass=50, children=inner_leaves, label="inner(50)")
    for c in inner_leaves:
        c.parent = inner
    
    outer_leaves = [Node(mass=m, label=f"L({m})") for m in [30, 20]]
    root = Node(mass=100, children=[inner] + outer_leaves, label="root(100)")
    inner.parent = root
    for c in outer_leaves:
        c.parent = root
    
    print("\nTree structure:")
    print_tree(root)
    
    candidates = get_nonbinary_nodes(root)
    print(f"\nNon-binary nodes: {[(n.label, n.mass, n.degree) for n, m, k in candidates]}")
    
    for eps in [0.1, 0.05, 0.02, 0.01, 0.005]:
        probs = vertex_selection_probs(candidates, eps)
        prob_str = ", ".join(f"{n.label}={p:.3f}" for (n, _, _), p in zip(candidates, probs))
        print(f"  epsilon={eps:.3f}: {prob_str}")


def demo_validation():
    """Validate against full coalescent."""
    print("\n" + "=" * 65)
    print("VALIDATION: Splitting formula vs full coalescent MC")
    print("=" * 65)
    
    validate_splitting([50, 30, 10, 7, 3])
    validate_splitting([100, 1, 1, 1, 1])


def demo_multiple_runs():
    """Run refinement many times and show distribution of first splits."""
    masses = [50, 30, 10, 7, 3]
    epsilon = 0.05
    n_runs = 50000
    
    print("\n" + "=" * 65)
    print(f"DISTRIBUTION OF FIRST SPLIT ({n_runs} runs)")
    print(f"Masses: {masses}, epsilon={epsilon}")
    print("=" * 65)
    
    first_splits = Counter()
    
    for _ in range(n_runs):
        root = make_star_tree(masses)
        result = refine_step(root, epsilon)
        if result:
            v, S1, S2, elen = result
            m1 = tuple(sorted(masses[i] for i in S1))
            m2 = tuple(sorted(masses[i] for i in S2))
            key = (min(m1, m2), max(m1, m2))
            first_splits[key] += 1
    
    # Compare with the direct splitting formula
    splits = enumerate_splits(len(masses))
    exact = {}
    log_ws = [log_split_weight(masses, S1) for S1, S2 in splits]
    max_lw = max(log_ws)
    ws = [math.exp(lw - max_lw) for lw in log_ws]
    total_w = sum(ws)
    for (S1, S2), w in zip(splits, ws):
        m1 = tuple(sorted(masses[i] for i in S1))
        m2 = tuple(sorted(masses[i] for i in S2))
        key = (min(m1, m2), max(m1, m2))
        exact[key] = exact.get(key, 0) + w / total_w
    
    print(f"\n{'S1':>25} {'S2':>25} {'Observed':>10} {'Exact':>10}")
    print("-" * 75)
    for key in sorted(first_splits.keys(),
                       key=lambda k: first_splits[k], reverse=True)[:10]:
        obs = first_splits[key] / n_runs
        ex = exact.get(key, 0)
        print(f"{str(key[0]):>25} {str(key[1]):>25} {obs:>10.4f} {ex:>10.4f}")
    
    print("\n(These should match since there's only one non-binary node.)")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    
    demo_full_refinement()
    demo_vertex_selection()
    demo_validation()
    demo_multiple_runs()
