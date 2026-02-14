"""
Run with:
    python main.py                              # demo + all tests
    python main.py demo                         # small demo
    python main.py demo --embed -d 2            # embed in R^2
    python main.py plot -k 20 -d 50             # plot with PCA
    python main.py sequence -k 20 -d 50 --gif   # contraction sequence
    python main.py sequence --config configs/medium_highd.json
    python main.py test                         # all tests
    python main.py test smoke                   # specific test

Config files (JSON) override defaults. CLI flags override config.
"""

import sys
import os
import json
import random
import torch

from crt.coalescent import coalescent
from crt.embed import embed_tree, TreeConfig
from crt.plot import plot_tree
from crt.io import save_experiment


def load_config(path):
    """Load a JSON config file, return as dict."""
    with open(path, 'r') as f:
        return json.load(f)


def parse_args(args):
    opts = {
        "command": "all",
        "test_name": "all",
        "embed": False,
        "save": None,
        "d": 2,
        "k": 5,
        "seed": 42,
        "gif": False,
        "config": None,
        "edge_order": "merge",
        "centroid_mode": "path",
    }

    # First pass: find --config and load it
    i = 0
    while i < len(args):
        if args[i] == "--config" and i + 1 < len(args):
            opts["config"] = args[i + 1]
            cfg = load_config(args[i + 1])
            # Config values become defaults (CLI can still override)
            for key in cfg:
                if key in opts:
                    opts[key] = cfg[key]
            break
        i += 1

    # Second pass: parse all CLI flags (override config)
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("demo", "test", "plot", "sequence"):
            opts["command"] = a
            if a == "test" and i + 1 < len(args) and not args[i+1].startswith("-"):
                opts["test_name"] = args[i + 1]
                i += 1
        elif a == "--embed":
            opts["embed"] = True
        elif a == "--gif":
            opts["gif"] = True
        elif a == "--config":
            i += 1  # already handled
        elif a == "--save":
            if i + 1 < len(args) and not args[i+1].startswith("-"):
                opts["save"] = args[i + 1]
                i += 1
            else:
                opts["save"] = True
        elif a == "-d" and i + 1 < len(args):
            opts["d"] = int(args[i + 1])
            i += 1
        elif a == "-k" and i + 1 < len(args):
            opts["k"] = int(args[i + 1])
            i += 1
        elif a == "--seed" and i + 1 < len(args):
            opts["seed"] = int(args[i + 1])
            i += 1
        elif a == "--edge-order" and i + 1 < len(args):
            opts["edge_order"] = args[i + 1]
            i += 1
        elif a == "--centroid" and i + 1 < len(args):
            opts["centroid_mode"] = args[i + 1]
            i += 1
        i += 1
    return opts


def demo(opts):
    from crt.contract import contract
    from crt.expand import expand

    masses = torch.tensor([50., 30., 10., 7., 3.])
    k = len(masses)
    d = opts["d"]

    config = TreeConfig(k=k, masses=masses, d=d,
                        embed=opts["embed"], seed=opts["seed"])

    print("=== Forward coalescent ===")
    t = coalescent(masses=masses)
    print(f"Masses: {t['masses']}")
    print(f"Merge pairs:\n{t['merge_pairs']}")
    print(f"Merge times: {t['merge_times']}")
    print(f"Heights: {t['heights']}")
    print(f"Edge lengths: {t['edge_lengths']}")
    print(f"Parent: {t['parent']}")

    if opts["embed"]:
        print(f"\n=== Embedding in R^{d} ===")
        positions = embed_tree(t, d)
        t["positions"] = positions
        print(f"Leaf positions (first 3 dims):")
        for i in range(k):
            pos = positions[i, :min(3, d)]
            print(f"  Leaf {i} (mass={masses[i]:.0f}): {pos.tolist()}")

    if opts["save"]:
        path = save_experiment(t, config)
        print(f"\n=== Saved to {path} ===")

    eps = t["edge_lengths"].median().item()
    print(f"\n=== Contract (epsilon={eps:.4f}) ===")
    ct = contract(t, eps)
    print(f"Surviving nodes: {ct['surviving_nodes']}")
    print(f"Children: {ct['children']}")
    print(f"Node masses: {ct['node_masses']}")

    print(f"\n=== Expand ===")
    t2 = expand(ct)
    print(f"Merge pairs:\n{t2['merge_pairs']}")
    print(f"Merge times: {t2['merge_times']}")


def do_plot(opts):
    k = opts["k"]
    d = opts["d"]

    print(f"Generating tree: k={k}, d={d}")
    t = coalescent(k=k)
    positions = embed_tree(t, d)

    save_path = opts["save"] if isinstance(opts["save"], str) else None
    plot_tree(t, positions=positions, save=save_path)


def do_sequence(opts):
    from crt.plot_sequence import plot_contraction_sequence
    from crt.io import save_experiment
    from crt.embed import TreeConfig
    import time, hashlib

    k = opts["k"]
    d = opts["d"]

    config = TreeConfig(k=k, d=d, embed=True, seed=opts["seed"])

    print(f"Generating tree: k={k}, d={d}")
    t = coalescent(k=k)
    positions = embed_tree(t, d)
    t["positions"] = positions

    # Create experiment folder
    ts = time.strftime("%Y%m%d_%H%M%S")
    h = hashlib.md5(str(time.time_ns()).encode()).hexdigest()[:6]
    exp_name = f"{ts}_{h}"
    if isinstance(opts["save"], str):
        exp_dir = opts["save"]
    else:
        exp_dir = os.path.join("experiments", exp_name)
    frames_dir = os.path.join(exp_dir, "frames")

    # Save tree data + copy config if used
    save_experiment(t, config, base_dir=os.path.dirname(exp_dir),
                    name=os.path.basename(exp_dir))

    if opts["config"]:
        import shutil
        shutil.copy(opts["config"],
                    os.path.join(exp_dir, "run_config.json"))

    gif_path = os.path.join(exp_dir, "tree.gif") if opts["gif"] else None
    pdf_path = os.path.join(exp_dir, "slides.pdf")

    plot_contraction_sequence(
        t, positions,
        output_dir=frames_dir,
        edge_order=opts["edge_order"],
        centroid_mode=opts["centroid_mode"],
        make_gif=opts["gif"],
        gif_path=gif_path or "tree.gif",
        make_pdf=True,
        pdf_path=pdf_path,
    )

    print(f"Experiment saved to {exp_dir}/")


def run_tests(which="all"):
    from crt.test_trees import TESTS
    if which == "all":
        for name, fn in TESTS.items():
            fn()
    elif which in TESTS:
        TESTS[which]()
    else:
        print(f"Unknown test: {which}")
        print(f"Available: {', '.join(TESTS.keys())}, all")
        sys.exit(1)
    print("\nAll tests done.")


if __name__ == "__main__":
    args = sys.argv[1:]
    opts = parse_args(args)

    random.seed(opts["seed"])
    torch.manual_seed(opts["seed"])

    if opts["command"] == "all" and not args:
        demo(opts)
        print("\n" + "=" * 60 + "\n")
        run_tests()
    elif opts["command"] == "demo":
        demo(opts)
    elif opts["command"] == "plot":
        do_plot(opts)
    elif opts["command"] == "sequence":
        do_sequence(opts)
    elif opts["command"] == "test":
        run_tests(opts["test_name"])
    else:
        print(__doc__)
