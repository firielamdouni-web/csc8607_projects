"""
Mini grid search — à implémenter.

Doit exposer un main() exécutable via :
    python -m src.grid_search --config configs/config.yaml

Exigences minimales :
- lire la section 'hparams' de la config
- lancer plusieurs runs en variant les hyperparamètres
- journaliser les hparams et résultats de chaque run (ex: TensorBoard HParams ou équivalent)
"""
"""
Mini grid search : test rapide de combinaisons d'hyperparamètres.

Usage:
    python -m src.grid_search --config configs/config.yaml
    
======================================================================
RÉCAPITULATIF
======================================================================

1. Val Acc: 0.8721 - {'lr': 0.002, 'weight_decay': 0.0, 'hidden_size': 192, 'embedding_dim': 
200}
2. Val Acc: 0.8711 - {'lr': 0.002, 'weight_decay': 1e-05, 'hidden_size': 192, 'embedding_dim': 200}
3. Val Acc: 0.8690 - {'lr': 0.002, 'weight_decay': 0.0, 'hidden_size': 192, 'embedding_dim': 
150}
4. Val Acc: 0.8668 - {'lr': 0.002, 'weight_decay': 0.0, 'hidden_size': 128, 'embedding_dim': 
150}
5. Val Acc: 0.8663 - {'lr': 0.002, 'weight_decay': 1e-05, 'hidden_size': 192, 'embedding_dim': 150}

======================================================================
Meilleure configuration : {'lr': 0.002, 'weight_decay': 0.0, 'hidden_size': 192, 'embedding_dim': 200}
Val Accuracy : 0.8721
======================================================================


"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import product
from torch.utils.tensorboard import SummaryWriter
from src.data_loading import get_dataloaders
from src.model import build_model
from src.utils import set_seed, get_device, count_parameters


def train_one_config(config, hparams, run_name, epochs=3):
    """Entraîne un modèle avec une configuration d'hyperparamètres."""
    
    set_seed(config["train"]["seed"])
    device = get_device(config["train"]["device"])
    
    # Mise à jour de la config avec les hyperparamètres
    config["train"]["optimizer"]["lr"] = hparams["lr"]
    config["train"]["optimizer"]["weight_decay"] = hparams["weight_decay"]
    config["model"]["rnn"]["hidden_size"] = hparams["hidden_size"]
    config["model"]["rnn"]["embedding_dim"] = hparams["embedding_dim"]
    
    train_loader, val_loader, _, meta = get_dataloaders(config)
    model = build_model(config, meta).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=hparams["lr"],
        weight_decay=hparams["weight_decay"]
    )
    
    log_dir = os.path.join(config["paths"]["runs_dir"], "grid_search", run_name)
    writer = SummaryWriter(log_dir)
    
    # Log des hyperparamètres
    writer.add_hparams(hparams, {})
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model(inputs)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        val_loss /= len(val_loader)
        val_acc = correct / total
        best_val_acc = max(best_val_acc, val_acc)
        
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/accuracy", val_acc, epoch)
        
        print(f"  Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
    
    writer.close()
    return best_val_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Grille d'hyperparamètres
    hparam_grid = {
        "lr": [0.0005, 0.001, 0.002],
        "weight_decay": [0.0, 1e-5],
        "hidden_size": [128, 192],
        "embedding_dim": [150, 200]
    }
    
    print(f"\n{'='*70}")
    print(f"GRID SEARCH : {len(list(product(*hparam_grid.values())))} combinaisons")
    print(f"{'='*70}\n")
    
    results = []
    
    for lr, wd, hidden, embed in product(*hparam_grid.values()):
        hparams = {
            "lr": lr,
            "weight_decay": wd,
            "hidden_size": hidden,
            "embedding_dim": embed
        }
        
        run_name = f"lr={lr}_wd={wd}_h={hidden}_e={embed}"
        print(f"\n{run_name}")
        print("-" * 70)
        
        best_acc = train_one_config(config, hparams, run_name, args.epochs)
        results.append((hparams, best_acc))
        
        print(f"  Meilleure Val Accuracy : {best_acc:.4f}")
    
    # Affichage du récapitulatif
    print(f"\n{'='*70}")
    print("RÉCAPITULATIF")
    print(f"{'='*70}\n")
    
    results.sort(key=lambda x: x[1], reverse=True)
    
    for i, (hparams, acc) in enumerate(results[:5], 1):
        print(f"{i}. Val Acc: {acc:.4f} - {hparams}")
    
    print(f"\n{'='*70}")
    print(f"Meilleure configuration : {results[0][0]}")
    print(f"Val Accuracy : {results[0][1]:.4f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()