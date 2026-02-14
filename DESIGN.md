# Coalescent Tree Library — Design Spec

## Overview

Library for sampling, embedding, coarsening, and visualizing additive
coalescent trees on weighted leaves. All tree data stored as dicts of
torch tensors.

## Modules

### `coalescent.py` — Core coalescent sampler

Gillespie exact simulation of the additive coalescent. O(k²).

**Input:** `k` (int) and/or `masses` (Tensor of leaf masses, default uniform 1/k).

**Algorithm:** At each step with j active particles of total mass M:
1. Sample waiting time ~ Exp((j-1)*M).
2. Pick particle a with prob ∝ m_a/M (size-biased), then b uniformly
   from the rest. This gives pair (a,b) with rate (m_a+m_b)/((j-1)*M).
3. Merge a and b into a new particle with mass m_a + m_b.

**Output dict:**
- `merge_pairs`: (k-1, 2) long — merged node ids per step
- `merge_times`: (k-1,) float64 — absolute merge times
- `edge_lengths`: (2k-2,) float64 — edge j = height[parent(j)] - height[j]
- `heights`: (2k-1,) float64 — node heights (leaves=0, root=max)
- `masses`: (k,) float64 — input leaf masses
- `parent`: (2k-1,) long — parent of each node (-1 for root)

Node indexing: 0..k-1 leaves, k..2k-2 internal in merge order, 2k-2 = root.

### `contract.py` — Forward coarsening

Deterministic edge collapse: all internal edges shorter than epsilon
are collapsed, producing a non-binary tree.

**Output dict:**
- `surviving_nodes`, `children` (list of lists), `node_masses`,
  `node_heights`, `surviving_merges`, `epsilon`, `original_tree`.

### `expand.py` — Inverse expansion

Resolves non-binary nodes via a constrained forward coalescent.
Particles within each group (defined by the contracted tree) can only
merge with particles in the same group. When a group is fully resolved
(single particle), it promotes to the parent group.

This is exact: the constrained coalescent produces the correct
conditional distribution given the contracted tree structure.

**Output:** same format as coalescent.py.

### `embed.py` — Spatial embedding

`embed_tree(tree, d)`: Embed tree in R^d via independent Gaussian
displacements along edges. Displacement on edge e ~ N(0, ℓ_e · I_d).

`project_2d(positions, k)`: For d > 2, PCA on leaf positions, project
all nodes. For d ≤ 2, pass-through. The top PCs align with the
deepest tree splits.

`TreeConfig`: Dataclass holding experiment parameters (k, d, masses,
seed, embed).

### `plot.py` — Single tree visualization

Edges colored by child's descendant mass (blue=light, red=heavy,
log scale with actual mass values on colorbar). Internal nodes sized
∝ sqrt(descendant mass), uniform dark color. Leaves as small hollow
circles.

### `plot_sequence.py` — Contraction sequence

Generates PNG frames + PDF slideshow + optional GIF showing progressive
tree expansion.

**Edge ordering:**
- `"merge"` (default): reverse coalescent order = true inverse.
  Reveals root split first, works down to leaves.
- `"length"`: metric coarsening, longest edges first.
- Custom: pass a list of internal node ids.

**Centroid modes** for contracted cluster positions:
- `"path"` (default): length-weighted average of edge midpoints
  within the collapsed cluster. = centroid under uniform measure on
  the tree.
- `"leaf"`: unweighted average of leaf descendant positions.
- `"root"`: position of the collapsed cluster's root node.

### `io.py` — Experiment persistence

`save_experiment(tree, config, base_dir, name)`: Creates a folder with
`config.json` + `tree.pt`.

`load_experiment(exp_dir)`: Returns (tree, config) tuple.

### `test_trees.py` — Test suite

- `smoke`: shapes, types, edge cases
- `consistency`: merge times increasing, edge lengths positive,
  binary structure
- `roundtrip`: coalescent → contract → expand produces valid binary tree
- `symmetry`: tree distribution on [2,2,1,1,1] modulo S₂×S₃
- `metric`: ultrametric property of tree distances

## Theory notes

**Splitting formula.** For root split {S₁,S₂} of a coalescent on k
particles with masses m₁,...,mₖ:
  P({S₁,S₂}) ∝ w_{S₁}^{|S₁|-1} · w_{S₂}^{|S₂|-1}
where w_S = Σ_{i∈S} mᵢ. Validated against forward coalescent.

**Embedding quality.** Euclidean distance² / d → tree distance as
d → ∞. Relative error ~ 1/√d. PCA recovers tree hierarchy because
the leaf covariance matrix encodes LCA heights.

**Contraction inverse.** The `"merge"` ordering (reverse height order)
is the exact inverse of forward coalescent construction. The `"length"`
ordering (metric coarsening) gives a different contraction path —
natural for ε-threshold coarsening but not the coalescent inverse.
