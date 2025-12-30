"""
Entraînement principal (à implémenter par l'étudiant·e).

Doit exposer un main() exécutable via :
    python -m src.train --config configs/config.yaml [--seed 42]

Exigences minimales :
- lire la config YAML
- respecter les chemins 'runs/' et 'artifacts/' définis dans la config
- journaliser les scalars 'train/loss' et 'val/loss' (et au moins une métrique de classification si applicable)
- supporter le flag --overfit_small (si True, sur-apprendre sur un très petit échantillon)

"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import os

from src.data_loading import get_dataloaders
from src.model import BiGRU_Attention


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--max_epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Charger config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Seed
    torch.manual_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Charger données (meilleure config: batch_size=64)
    train_loader, val_loader, test_loader, meta = get_dataloaders(config)
    vocab_size = meta['vocab_size']
    
    # Modèle (meilleure config: hidden_size=192, embedding_dim=200)
    model = BiGRU_Attention(
        vocab_size=vocab_size,
        embedding_dim=200,
        hidden_size=192,
        num_classes=20,
        dropout_embed=0.1,
        dropout_fc=0.5
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Optimiseur (meilleure config: LR=0.002, weight_decay=0.0)
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    
    # TensorBoard
    log_dir = Path(config['paths']['runs_dir']) / 'final_training'
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard logs: {log_dir}")
    
    # Checkpoint dir
    ckpt_dir = Path(config['paths']['artifacts_dir'])
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    best_ckpt_path = str(ckpt_dir / 'best.ckpt') 
    
    # Entraînement
    best_val_acc = 0.0
    
    for epoch in range(1, args.max_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.max_epochs}")
        print(f"{'='*60}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Log TensorBoard (tags exacts requis)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/accuracy', val_acc, epoch)
        
        # Sauvegarder meilleur checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config': {
                    'lr': 0.002,
                    'weight_decay': 0.0,
                    'hidden_size': 192,
                    'embedding_dim': 200,
                    'batch_size': 64,
                    'max_epochs': args.max_epochs
                }
            }, best_ckpt_path)
            print(f" Best checkpoint saved: {best_ckpt_path} (val_acc={val_acc:.2f}%)")
    
    writer.close()
    print(f"\n{'='*60}")
    print(f"Training complete! Best val accuracy: {best_val_acc:.2f}%")
    print(f"Checkpoint: {best_ckpt_path}")
    print(f"TensorBoard: tensorboard --logdir={log_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
    
"""
(deeplearning) famdouni@arcadia-slurm-node-1:~/csc8607_projects-main$ python -m src.train --config configs/config.yaml --max_epochs 15 --seed 42
Using device: cuda
Total parameters: 10,462,373
TensorBoard logs: runs/final_training

============================================================
Epoch 1/15
============================================================
Train Loss: 1.7455 | Train Acc: 48.35%                                                                                                      
Val Loss:   0.7910 | Val Acc:   77.45%
 Best checkpoint saved: artifacts/best.ckpt (val_acc=77.45%)

============================================================
Epoch 2/15
============================================================
Train Loss: 0.6027 | Train Acc: 82.93%                                                                                                      
Val Loss:   0.5256 | Val Acc:   84.30%
 Best checkpoint saved: artifacts/best.ckpt (val_acc=84.30%)

============================================================
Epoch 3/15
============================================================
Train Loss: 0.2920 | Train Acc: 91.66%                                                                                                      
Val Loss:   0.4301 | Val Acc:   87.21%
 Best checkpoint saved: artifacts/best.ckpt (val_acc=87.21%)

============================================================
Epoch 4/15
============================================================
Train Loss: 0.1592 | Train Acc: 95.60%                                                                                                      
Val Loss:   0.4178 | Val Acc:   88.22%
 Best checkpoint saved: artifacts/best.ckpt (val_acc=88.22%)

============================================================
Epoch 5/15
============================================================
Train Loss: 0.0911 | Train Acc: 97.49%                                                                                                      
Val Loss:   0.4344 | Val Acc:   88.38%
 Best checkpoint saved: artifacts/best.ckpt (val_acc=88.38%)

============================================================
Epoch 6/15
============================================================
Train Loss: 0.0529 | Train Acc: 98.65%                                                                                                      
Val Loss:   0.4348 | Val Acc:   89.18%
 Best checkpoint saved: artifacts/best.ckpt (val_acc=89.18%)

============================================================
Epoch 7/15
============================================================
Train Loss: 0.0698 | Train Acc: 98.06%                                                                                                      
Val Loss:   0.4483 | Val Acc:   89.34%
 Best checkpoint saved: artifacts/best.ckpt (val_acc=89.34%)

============================================================
Epoch 8/15
============================================================
Train Loss: 0.0405 | Train Acc: 98.97%                                                                                                      
Val Loss:   0.4361 | Val Acc:   90.08%
 Best checkpoint saved: artifacts/best.ckpt (val_acc=90.08%)

============================================================
Epoch 9/15
============================================================
Train Loss: 0.0288 | Train Acc: 99.20%                                                                                                      
Val Loss:   0.4715 | Val Acc:   89.76%

============================================================
Epoch 10/15
============================================================
Train Loss: 0.0215 | Train Acc: 99.50%                                                                                                      
Val Loss:   0.4814 | Val Acc:   89.92%

============================================================
Epoch 11/15
============================================================
Train Loss: 0.0220 | Train Acc: 99.50%                                                                                                      
Val Loss:   0.4987 | Val Acc:   89.81%

============================================================
Epoch 12/15
============================================================
Train Loss: 0.0199 | Train Acc: 99.40%                                                                                                      
Val Loss:   0.5133 | Val Acc:   89.87%

============================================================
Epoch 13/15
============================================================
Train Loss: 0.0158 | Train Acc: 99.63%                                                                                                      
Val Loss:   0.5185 | Val Acc:   89.44%

============================================================
Epoch 14/15
============================================================
Train Loss: 0.0154 | Train Acc: 99.61%                                                                                                      
Val Loss:   0.5434 | Val Acc:   89.97%

============================================================
Epoch 15/15
============================================================
Train Loss: 0.0181 | Train Acc: 99.56%                                                                                                      
Val Loss:   0.5247 | Val Acc:   90.45%
 Best checkpoint saved: artifacts/best.ckpt (val_acc=90.45%)

============================================================
Training complete! Best val accuracy: 90.45%
Checkpoint: artifacts/best.ckpt
TensorBoard: tensorboard --logdir=runs/final_training
============================================================
"""