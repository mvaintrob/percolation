"""
io.py — Save and load experiment results.

Creates a folder per experiment with:
    config.json  — the TreeConfig
    tree.pt      — the tree tensors
"""

import json
import os
import time
import hashlib
import torch
from dataclasses import asdict

from .embed import TreeConfig


def save_experiment(tree, config, base_dir="experiments", name=None):
    """Save tree and config to an experiment folder.

    Args:
        tree: dict of tensors from coalescent.py (+ optional positions)
        config: TreeConfig instance
        base_dir: parent directory for experiments
        name: folder name (default: timestamp + short hash)

    Returns:
        path to the created experiment folder
    """
    if name is None:
        t = time.strftime("%Y%m%d_%H%M%S")
        h = hashlib.md5(str(time.time_ns()).encode()).hexdigest()[:6]
        name = f"{t}_{h}"

    exp_dir = os.path.join(base_dir, name)
    os.makedirs(exp_dir, exist_ok=True)

    # Save config as JSON
    cfg_dict = asdict(config)
    # Convert tensors to lists for JSON
    if cfg_dict["masses"] is not None:
        cfg_dict["masses"] = cfg_dict["masses"].tolist()
    cfg_path = os.path.join(exp_dir, "config.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg_dict, f, indent=2)

    # Save tree as .pt
    tree_path = os.path.join(exp_dir, "tree.pt")
    torch.save(tree, tree_path)

    return exp_dir


def load_experiment(exp_dir):
    """Load tree and config from an experiment folder.

    Returns:
        (tree, config) tuple
    """
    cfg_path = os.path.join(exp_dir, "config.json")
    with open(cfg_path, "r") as f:
        cfg_dict = json.load(f)

    # Reconstruct config
    if cfg_dict["masses"] is not None:
        cfg_dict["masses"] = torch.tensor(cfg_dict["masses"],
                                           dtype=torch.float64)
    config = TreeConfig(**cfg_dict)

    tree_path = os.path.join(exp_dir, "tree.pt")
    tree = torch.load(tree_path, weights_only=False)

    return tree, config
