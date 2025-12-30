"""
Recherche de taux d'apprentissage (LR finder) — à implémenter.

Doit exposer un main() exécutable via :
    python -m src.lr_finder --config configs/config.yaml

Exigences minimales :
- produire un log/trace permettant de visualiser (lr, loss) dans TensorBoard ou équivalent.
"""
"""
LR Finder : balayage logarithmique du learning rate.

Usage:
    python -m src.lr_finder --config configs/config.yaml
    
(.venv) PS C:\Users\Amdouni\OneDrive - IMTBS-TSP\Documents\PFA-deeplea\csc8607_projects-main> python -m src.lr_finder --config configs/config.yaml
[utils] Seeds initialized: 42

======================================================================
MODÈLE CONSTRUIT : BiGRU_Attention
======================================================================
  - vocab_size      : 50002
  - embedding_dim   : 200 (HYPERPARAMÈTRE 2)
  - hidden_size     : 192 (HYPERPARAMÈTRE 1)
  - num_layers      : 1
  - num_classes     : 20
  - dropout_embed   : 0.1
  - dropout_rnn     : 0.1
  - dropout_fc      : 0.5
  - padding_idx     : 0
======================================================================


======================================================================
LR FINDER : 1.0e-06 → 1.0e+00 sur 100 itérations
===================================================================

C:\Users\Amdouni\OneDrive - IMTBS-TSP\Documents\PFA-deeplea\.venv\Lib\site-packages\torch\utils\data\dataloader.py:668: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
  warnings.warn(warn_msg)
Iter 20/100 - LR: 1.42e-05 - Loss: 2.9897
Iter 40/100 - LR: 2.31e-04 - Loss: 3.0084
Iter 60/100 - LR: 3.76e-03 - Loss: 2.9745
Iter 80/100 - LR: 6.14e-02 - Loss: 4.0934

Divergence détectée à LR=1.87e-01, arrêt anticipé.

======================================================================
LR suggéré (loss minimale lissée) : 1.52e-02
Plage recommandée : [1.52e-03, 7.60e-02]
======================================================================

(.venv) PS C:\Users\Amdouni\OneDrive - IMTBS-TSP\Documents\PFA-deeplea\csc8607_projects-main> tensorboard --logdir runs/lr_finder     
C:\Users\Amdouni\OneDrive - IMTBS-TSP\Documents\PFA-deeplea\.venv\Lib\site-packages\tensorboard\default.py:30: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin 
to Setuptools<81.
  import pkg_resources
TensorFlow installation not found - running with reduced feature set.
Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.20.0 at http://localhost:6006/ (Press CTRL+C to quit)   
 
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from src.data_loading import get_dataloaders
from src.model import build_model
from src.utils import set_seed, get_device


def find_lr(config, min_lr=1e-6, max_lr=1.0, num_iter=100):
    """Balayage logarithmique du LR."""
    
    set_seed(config["train"]["seed"])
    device = get_device(config["train"]["device"])
    
    train_loader, _, _, meta = get_dataloaders(config)
    model = build_model(config, meta).to(device)
    criterion = nn.CrossEntropyLoss()
    
    lrs = np.logspace(np.log10(min_lr), np.log10(max_lr), num_iter)
    losses = []
    
    log_dir = os.path.join(config["paths"]["runs_dir"], "lr_finder")
    writer = SummaryWriter(log_dir)
    
    print(f"\n{'='*70}")
    print(f"LR FINDER : {min_lr:.1e} → {max_lr:.1e} sur {num_iter} itérations")
    print(f"{'='*70}\n")
    
    optimizer = optim.Adam(model.parameters(), lr=min_lr)
    data_iter = iter(train_loader)
    
    for i, lr in enumerate(lrs):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        try:
            inputs, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            inputs, labels = next(data_iter)
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        loss_val = loss.item()
        losses.append(loss_val)
        
        writer.add_scalar("lr_finder/lr", lr, i)
        writer.add_scalar("lr_finder/loss", loss_val, i)
        
        if (i + 1) % 20 == 0:
            print(f"Iter {i+1}/{num_iter} - LR: {lr:.2e} - Loss: {loss_val:.4f}")
        
        if loss_val > 10.0 or np.isnan(loss_val):
            print(f"\nDivergence détectée à LR={lr:.2e}, arrêt anticipé.")
            break
    
    writer.close()
    
    # Analyse simple
    smooth_losses = np.convolve(losses, np.ones(5)/5, mode='valid')
    min_idx = np.argmin(smooth_losses)
    optimal_lr = lrs[min_idx]
    
    print(f"\n{'='*70}")
    print(f"LR suggéré (loss minimale lissée) : {optimal_lr:.2e}")
    print(f"Plage recommandée : [{optimal_lr/10:.2e}, {optimal_lr*5:.2e}]")
    print(f"{'='*70}\n")
    
    return optimal_lr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--max_lr", type=float, default=1.0)
    parser.add_argument("--num_iter", type=int, default=100)
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    find_lr(config, args.min_lr, args.max_lr, args.num_iter)


if __name__ == "__main__":
    main()