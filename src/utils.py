"""
Utils génériques.

Fonctions attendues (signatures imposées) :
- set_seed(seed: int) -> None
- get_device(prefer: str | None = "auto") -> str
- count_parameters(model) -> int
- save_config_snapshot(config: dict, out_dir: str) -> None
"""

import os
import random
import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    """Initialise les seeds (numpy/torch/python) pour reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[utils] Seeds initialized: {seed}")


def get_device(prefer: str | None = "auto") -> str:
    """
    Retourne 'cpu' ou 'cuda'.

    Args:
        prefer: "cpu", "cuda", ou "auto" (choisir GPU si dispo)

    Returns:
        device string: "cpu" ou "cuda"
    """
    if prefer == "cpu":
        device = "cpu"
    elif prefer == "cuda":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:  # "auto" ou autre
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[utils] Using device: {device}")
    return device


def count_parameters(model) -> int:
    """Retourne le nombre de paramètres entraînables du modèle."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[utils] Model has {total:,} trainable parameters")
    return total


def save_config_snapshot(config: dict, out_dir: str) -> None:
    """
    Sauvegarde une copie de la config (format YAML) dans out_dir.

    Utile pour traçabilité : on peut récupérer exactement quelle config
    a généré un run spécifique.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "config_snapshot.yaml")

    with open(out_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"[utils] Config snapshot saved to {out_path}")