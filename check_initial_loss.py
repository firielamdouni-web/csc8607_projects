import yaml
import torch
import torch.nn as nn
import numpy as np
from src.data_loading import get_dataloaders
from src.model import build_model
from src.utils import set_seed


def main():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    
    set_seed(config["train"]["seed"])
    train_loader, _, _, meta = get_dataloaders(config)
    model = build_model(config, meta)
    
    inputs, labels = next(iter(train_loader))
    
    print(f"\n{'='*70}")
    print("VÉRIFICATION LOSS INITIALE (M2)")
    print(f"{'='*70}")
    
    criterion = nn.CrossEntropyLoss()
    model.train()
    logits = model(inputs)
    loss = criterion(logits, labels)
    
    expected_loss = -np.log(1.0 / meta["num_classes"])
    
    print(f"\nLoss observée : {loss.item():.4f}")
    print(f"Loss attendue : {expected_loss:.4f} (-log(1/{meta['num_classes']}))")
    print(f"Écart relatif : {100*abs(loss.item()-expected_loss)/expected_loss:.2f}%")
    
    model.zero_grad()
    loss.backward()
    
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"Norme gradients : {grad_norm:.4f}")
    print(f"\n Gradients non nuls : backward OK")


if __name__ == "__main__":
    main()