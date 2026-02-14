# CRT — Coalescent Random Tree Library

Sample, embed, coarsen, and visualize additive coalescent trees.

## Setup

```bash
pip install torch matplotlib Pillow
```

## Quick start

```bash
python main.py demo                          # run the forward coalescent on 5 weighted particles
python main.py plot -k 30 -d 50              # embed in R^50, PCA-project to 2D, plot
python main.py sequence -k 20 -d 50 --gif    # contraction sequence with PDF slideshow + GIF
python main.py sequence --config configs/medium_highd.json  # run from config file
python main.py test                          # run all tests
```

## Project structure

```
main.py                  # CLI entry point
crt/
    __init__.py
    coalescent.py        # Forward additive coalescent (Gillespie, O(k²))
    contract.py          # Collapse short edges → non-binary tree
    expand.py            # Resolve non-binary nodes via constrained coalescent
    embed.py             # Embed tree in R^d, PCA projection
    plot.py              # Single tree plot
    plot_sequence.py     # Contraction sequence → PNG frames / PDF / GIF
    io.py                # Save/load experiments
    test_trees.py        # Test suite
    DESIGN.md            # Technical design spec
configs/                 # Sample config files
    small_2d.json        # k=10, d=2
    medium_highd.json    # k=50, d=50
    metric_coarsening.json   # k=30, d=11, edge_order=length
    large_tree.json      # k=500, d=100
experiments/             # Auto-created, one folder per run
    <timestamp>/
        config.json      # Parameters
        tree.pt          # Tree tensors
        slides.pdf       # Clickable slideshow
        tree.gif         # Animation (if --gif)
        frames/          # Individual PNG frames
```

## Config files

JSON files with any subset of these fields:

```json
{
  "k": 50,
  "d": 50,
  "seed": 123,
  "embed": true,
  "edge_order": "merge",
  "centroid_mode": "path",
  "gif": true
}
```

CLI flags override config values. Config values override defaults.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` | 5 | Number of leaves |
| `d` | 2 | Embedding dimension |
| `seed` | 42 | Random seed |
| `edge_order` | `"merge"` | Contraction order: `"merge"` (coalescent inverse) or `"length"` (metric) |
| `centroid_mode` | `"path"` | Contracted node position: `"path"`, `"leaf"`, or `"root"` |
| `gif` | false | Generate animated GIF |

## Key concepts

**Additive coalescent.** Particles with masses $m_1, \ldots, m_k$ merge
pairwise at rate $m_a + m_b$. This produces a random binary tree whose
structure is governed by the Poisson-Dirichlet PD(1/2) partition. The
continuum limit is the Continuum Random Tree (CRT).

**Embedding.** Each tree edge gets an independent Gaussian displacement
$\Delta x \sim N(0, \ell_e \cdot I_d)$. For $d > 2$, PCA on leaf
positions recovers the dominant tree splits. Euclidean distances
approximate tree distances with relative error $\sim 1/\sqrt{d}$.

**Contraction.** Two orderings are supported:
- `"merge"` (default): reverse coalescent order. This is the exact
  inverse of the forward coalescent — each step collapses one merge.
- `"length"`: metric coarsening. Collapse shortest edges first,
  equivalent to raising an $\varepsilon$-threshold.

**Centroid modes** for positioning contracted clusters:
- `"path"`: length-weighted edge midpoint average (centroid of uniform
  measure on collapsed sub-tree). Default.
- `"leaf"`: unweighted average of leaf descendant positions (Bayesian
  optimal estimate of cluster root position in ultrametric case).
- `"root"`: position of the root node of the collapsed sub-tree.

## Tests

```bash
python main.py test smoke        # imports, shapes, types
python main.py test consistency  # internal consistency of tree tensors
python main.py test roundtrip    # coalescent → contract → expand
python main.py test symmetry     # distribution test on [2,2,1,1,1]
python main.py test metric       # ultrametric property
```

## References

- Aldous, "The Continuum Random Tree III" (1993)
- Aldous & Pitman, "The standard additive coalescent" (1998)
- Fitzner & van der Hofstad, on cluster geometry in high-dimensional percolation
