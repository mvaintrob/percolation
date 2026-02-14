# Coalescent Tree Library — Design Spec

## Overview

Three modules implementing the additive coalescent on weighted leaves,
plus forward (coarsening) and inverse (expansion) operations on the
resulting binary trees. Outputs in torch format.

## Modules

### 1. `coalescent.py` — Core coalescent sampler

Runs the forward additive coalescent on k weighted particles.
Produces a binary tree with edge lengths and merge ordering.

**Input:**
- `k` (int, optional): number of leaves. Default = len(masses).
- `masses` (Tensor, optional): leaf masses. Default = uniform (1/k each).
- Must provide at least one of k or masses.

**Algorithm:**
- Gillespie exact simulation: O(k^2).
- Each step: sample waiting time Exp((j-1)*M), pick merging pair
  with prob (m_a + m_b) / ((j-1)*M) via size-biased sampling.

**Output dict (torch tensors):**
- `merge_pairs`: (k-1, 2) int — which clusters merged at each step.
  Rows are in merge order (earliest merge first).
- `merge_times`: (k-1,) float — absolute time of each merge (= height
  of the resulting internal node).
- `edge_lengths`: (2k-2,) float — length of each edge. Convention TBD
  but canonical: edges indexed by their child node (k leaf edges
  followed by k-2 internal edges, or interleaved with merge order).
- `masses`: (k,) float — the input leaf masses.
- `heights`: (2k-1,) float — height of each node. Leaves have height 0,
  internal node i has height = merge_times[i]. Root = max height.

Edge indexing convention: node j (0-indexed) has edge to its parent.
Leaves are 0..k-1. Internal nodes are k..2k-2 in merge order.
So internal node k is the first merge, 2k-2 is the root.

### 2. `contract.py` — Forward coarsening

Collapses short edges to produce a non-binary tree.

**Input:**
- A binary tree (output of coalescent.py).
- `epsilon` (float): collapse all internal edges with length < epsilon.

**Output dict (torch tensors):**
- `children`: ragged or padded (n_internal, max_degree) int — children
  of each surviving internal node.
- `node_masses`: (n_nodes,) float — descendant mass of each node.
- `node_heights`: (n_nodes,) float — height of each surviving node.
- `edge_lengths`: lengths of surviving edges.
- `merge_order`: the subsequence of original merge_pairs for surviving edges.

The coarsened tree has fewer internal nodes. Each surviving internal
node may have degree > 2 (it absorbed a cluster of short edges).

### 3. `expand.py` — Inverse expansion

Resolves non-binary nodes into full binary trees by running local
forward coalescents.

**Input:**
- A (possibly non-binary) tree, e.g. output of contract.py, or a
  star tree on k leaves.
- `masses` for the leaves.

**Algorithm:**
- For each non-binary node v (degree > 2): run the forward additive
  coalescent on v's children (using their descendant masses).
  This gives the full binary sub-tree inside v, including edge
  lengths and merge ordering.
- Graft each local binary sub-tree in place of v's children.
- Concatenate local merge orderings into a global expansion order.

The expansion of a k-star IS the coalescent on k particles.
The expansion of an already-binary tree is a no-op.

**Output:** same format as coalescent.py.

### 4. `test_trees.py` — Tests and validation

Separate test file. Suggested test modes (selectable via CLI / main):

- `test_coalescent`: Run coalescent, verify merge_pairs and merge_times
  are consistent (times increasing, pairs valid).
- `test_roundtrip`: coalescent -> contract(eps) -> expand, verify
  output tree has correct distribution (compare root split statistics
  against exact formula for small k).
- `test_symmetry`: For masses [2,2,1,1,1], verify tree distribution
  modulo S_2 x S_3 matches between coalescent and expand(star).
- `test_metric`: Verify tree metric d(i,j) = h_i + h_j - 2*h_lca(i,j)
  is consistent between original and roundtripped trees (up to
  sub-epsilon error from contraction).

## Data format details

All outputs are dicts of torch tensors. The canonical binary tree is:

```python
{
    "merge_pairs": torch.LongTensor,   # (k-1, 2)
    "merge_times": torch.FloatTensor,  # (k-1,)
    "edge_lengths": torch.FloatTensor, # (2k-2,)
    "heights": torch.FloatTensor,      # (2k-1,)
    "masses": torch.FloatTensor,       # (k,) leaf masses
}
```

Node indexing: 0..k-1 are leaves, k..2k-2 are internal nodes in
merge order. Node k+i is the internal node created by merge i.
The root is node 2k-2.

Edge indexing: edge j is the edge from node j to its parent.
The root (2k-2) has no edge. So there are 2k-2 edges total.
edge_lengths[j] = heights[parent(j)] - heights[j].

## Notes

- The coalescent and expand are both exact (no epsilon, no
  discretization, no MCMC). They use continuous-time Gillespie
  simulation.
- Contract is deterministic given the tree and epsilon.
- Expand is stochastic: it samples a random binary resolution
  consistent with the additive coalescent. Different calls to
  expand on the same contracted tree give different binary trees,
  drawn from the correct posterior.
- For k < 100, everything runs in milliseconds. The O(k^2)
  coalescent is the bottleneck. For larger k, approximate methods
  (excursion sampling, heavy/light splitting) would be needed but
  are not implemented here.
