"""
Itération supplémentaire - Grid Search Resserrée (M8)

Stratégie:
- Resserrage LR autour de 0.002 : {0.0015, 0.002, 0.0025}
- Test num_layers : {1, 2} (exploration de BiGRU plus profond)
- Hyperparamètres fixes : hidden_size=192, embedding_dim=200, weight_decay=0.0
- Durée : 5 époques par run (validation rapide)
- Sauvegarde : runs/iteration_supplementaire/, snapshots configs dans artifacts/
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import yaml
from datetime import datetime
import json

from src.data_loading import get_dataloaders
from src.model import BiGRU_Attention
from src.utils import set_seed, save_config_snapshot


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Entraîne le modèle sur une époque."""
    model.train()
    total_loss = 0.0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    """Évalue le modèle sur validation."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            logits = model(inputs)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


def run_single_config(config, train_loader, val_loader, device):
    """Lance un entraînement pour une configuration donnée."""
    # Créer nom du run
    lr = config['lr']
    num_layers = config['num_layers']
    run_name = f"lr={lr}_layers={num_layers}_h=192_e=200"
    
    # Créer writer TensorBoard
    log_dir = Path("runs/iteration_supplementaire") / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    
    # Sauvegarder snapshot config dans un sous-dossier par run
    snapshot_dir = Path("artifacts") / "configs" / run_name
    save_config_snapshot(config, str(snapshot_dir))
    
    # Initialiser modèle
    model = BiGRU_Attention(
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        num_classes=config['num_classes'],
        dropout_embed=0.1,
        dropout_fc=0.5
    ).to(device)
    
    # Optimiseur et loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n{'='*60}")
    print(f"Run: {run_name}")
    print(f"LR={lr}, num_layers={num_layers}, epochs={config['num_epochs']}")
    print(f"{'='*60}")
    
    # Boucle d'entraînement
    best_val_acc = 0.0
    
    for epoch in range(config['num_epochs']):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Log TensorBoard
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/accuracy', val_acc, epoch)
        
        # Log HParams
        if epoch == 0:
            writer.add_hparams(
                {'lr': lr, 'num_layers': num_layers, 'hidden_size': 192, 'embedding_dim': 200},
                {'hparam/val_accuracy': 0.0, 'hparam/val_loss': 999.0}
            )
        
        print(f"Epoch {epoch+1}/{config['num_epochs']} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc*100:.2f}%")
        
        # Sauvegarder meilleure accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    # Log final HParams avec vraies valeurs
    writer.add_hparams(
        {'lr': lr, 'num_layers': num_layers, 'hidden_size': 192, 'embedding_dim': 200},
        {'hparam/val_accuracy': best_val_acc, 'hparam/val_loss': val_loss}
    )
    
    writer.close()
    
    print(f" Best Val Accuracy: {best_val_acc*100:.2f}%\n")
    
    return {
        'run_name': run_name,
        'lr': lr,
        'num_layers': num_layers,
        'best_val_acc': best_val_acc,
        'final_val_loss': val_loss
    }


def main():
    """Exécute la grid search d'itération supplémentaire."""
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Charger les données
    print("Chargement des données...")
    config_data = {
        'dataset': {
            'name': '20newsgroups',
            'root': './data',
            'split': {'train': 0.8, 'val': 0.1, 'test': 0.1},
            'num_workers': 0,
            'shuffle': True
        },
        'train': {
            'seed': 42,
            'batch_size': 64,
            'overfit_small': False
        },
        'preprocess': {
            'max_seq_len': 400,
            'min_freq': 2,
            'max_vocab_size': 50000
        }
    }
    train_loader, val_loader, test_loader, meta = get_dataloaders(config_data)
        
    # Grilles d'hyperparamètres
    lr_values = [0.0015, 0.002, 0.0025]
    num_layers_values = [1, 2]
    
    # Configuration de base
    base_config = {
        'vocab_size': meta['vocab_size'],
        'embedding_dim': 200,
        'hidden_size': 192,
        'num_classes': meta['num_classes'],
        'weight_decay': 0.0,  # Meilleure valeur précédente
        'num_epochs': 5  # 5 époques pour validation rapide
    }
    
    print(f"\n{'='*60}")
    print("Itération Supplémentaire - Grid Search Resserrée")
    print(f"{'='*60}")
    print(f"LR values: {lr_values}")
    print(f"num_layers values: {num_layers_values}")
    print(f"Fixed: hidden_size=192, embedding_dim=200, weight_decay=0.0")
    print(f"Epochs per run: {base_config['num_epochs']}")
    print(f"Total runs: {len(lr_values) * len(num_layers_values)}")
    print(f"{'='*60}\n")
    
    # Lancer tous les runs
    results = []
    
    for lr in lr_values:
        for num_layers in num_layers_values:
            config = base_config.copy()
            config['lr'] = lr
            config['num_layers'] = num_layers
            
            result = run_single_config(config, train_loader, val_loader, device)
            results.append(result)
    
    # Afficher résumé
    print(f"\n{'='*60}")
    print("RÉSUMÉ DES RÉSULTATS")
    print(f"{'='*60}")
    
    # Trier par accuracy décroissante
    results_sorted = sorted(results, key=lambda x: x['best_val_acc'], reverse=True)
    
    print(f"\n{'Run':<35} | {'LR':>6} | {'Layers':>6} | {'Val Acc':>8} | {'Val Loss':>8}")
    print("-" * 80)
    
    for r in results_sorted:
        print(f"{r['run_name']:<35} | {r['lr']:>6.4f} | {r['num_layers']:>6} | "
              f"{r['best_val_acc']*100:>7.2f}% | {r['final_val_loss']:>8.4f}")
    
    # Sauvegarder résultats JSON
    results_file = Path("artifacts/iteration_supplementaire_results.json")
    with open(results_file, 'w') as f:
        json.dump(results_sorted, f, indent=2)
    
    print(f"\n Résultats sauvegardés dans {results_file}")
    print(f" Logs TensorBoard dans runs/iteration_supplementaire/")
    
    # Meilleure config
    best = results_sorted[0]
    print(f"\n{'='*60}")
    print("MEILLEURE CONFIGURATION")
    print(f"{'='*60}")
    print(f"Run: {best['run_name']}")
    print(f"LR: {best['lr']}")
    print(f"num_layers: {best['num_layers']}")
    print(f"Val Accuracy: {best['best_val_acc']*100:.2f}%")
    print(f"Val Loss: {best['final_val_loss']:.4f}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
    
"""
(deeplearning) famdouni@arcadia-slurm-node-2:~/csc8607_projects-main$ python iteration_supplementaire.py
[utils] Seeds initialized: 42
Device: cuda
Chargement des données...

============================================================
Itération Supplémentaire - Grid Search Resserrée
============================================================
LR values: [0.0015, 0.002, 0.0025]
num_layers values: [1, 2]
Fixed: hidden_size=192, embedding_dim=200, weight_decay=0.0
Epochs per run: 5
Total runs: 6
============================================================


============================================================
Run: lr=0.0015_layers=1_h=192_e=200
LR=0.0015, num_layers=1, epochs=5
============================================================
Epoch 1/5 | Train Loss: 1.9231 | Val Loss: 0.9455 | Val Acc: 74.06%
Epoch 2/5 | Train Loss: 0.7394 | Val Loss: 0.5727 | Val Acc: 82.97%
Epoch 3/5 | Train Loss: 0.3848 | Val Loss: 0.4643 | Val Acc: 86.53%
Epoch 4/5 | Train Loss: 0.2268 | Val Loss: 0.4421 | Val Acc: 87.64%
Epoch 5/5 | Train Loss: 0.1331 | Val Loss: 0.4291 | Val Acc: 88.75%
 Best Val Accuracy: 88.75%


============================================================
Run: lr=0.0015_layers=2_h=192_e=200
LR=0.0015, num_layers=2, epochs=5
============================================================
Epoch 1/5 | Train Loss: 1.8423 | Val Loss: 0.8272 | Val Acc: 74.75%
Epoch 2/5 | Train Loss: 0.5629 | Val Loss: 0.4820 | Val Acc: 85.57%
Epoch 3/5 | Train Loss: 0.2325 | Val Loss: 0.4559 | Val Acc: 87.00%
Epoch 4/5 | Train Loss: 0.1106 | Val Loss: 0.4685 | Val Acc: 87.80%
Epoch 5/5 | Train Loss: 0.0693 | Val Loss: 0.4739 | Val Acc: 88.22%
 Best Val Accuracy: 88.22%


============================================================
Run: lr=0.002_layers=1_h=192_e=200
LR=0.002, num_layers=1, epochs=5
============================================================
Epoch 1/5 | Train Loss: 1.7737 | Val Loss: 0.8202 | Val Acc: 75.60%
Epoch 2/5 | Train Loss: 0.5991 | Val Loss: 0.5092 | Val Acc: 84.83%
Epoch 3/5 | Train Loss: 0.2818 | Val Loss: 0.4277 | Val Acc: 88.01%
Epoch 4/5 | Train Loss: 0.1445 | Val Loss: 0.4453 | Val Acc: 88.75%
Epoch 5/5 | Train Loss: 0.0812 | Val Loss: 0.4301 | Val Acc: 89.87%
 Best Val Accuracy: 89.87%


============================================================
Run: lr=0.002_layers=2_h=192_e=200
LR=0.002, num_layers=2, epochs=5
============================================================
Epoch 1/5 | Train Loss: 1.6661 | Val Loss: 0.6792 | Val Acc: 78.78%
Epoch 2/5 | Train Loss: 0.4771 | Val Loss: 0.4559 | Val Acc: 87.59%
Epoch 3/5 | Train Loss: 0.1931 | Val Loss: 0.4588 | Val Acc: 87.96%
Epoch 4/5 | Train Loss: 0.0903 | Val Loss: 0.4105 | Val Acc: 90.34%
Epoch 5/5 | Train Loss: 0.0551 | Val Loss: 0.4470 | Val Acc: 90.03%
 Best Val Accuracy: 90.34%


============================================================
Run: lr=0.0025_layers=1_h=192_e=200
LR=0.0025, num_layers=1, epochs=5
============================================================
Epoch 1/5 | Train Loss: 1.6187 | Val Loss: 0.7200 | Val Acc: 79.05%
Epoch 2/5 | Train Loss: 0.5255 | Val Loss: 0.4806 | Val Acc: 85.89%
Epoch 3/5 | Train Loss: 0.2485 | Val Loss: 0.4019 | Val Acc: 88.75%
Epoch 4/5 | Train Loss: 0.1266 | Val Loss: 0.4079 | Val Acc: 89.76%
Epoch 5/5 | Train Loss: 0.0774 | Val Loss: 0.4161 | Val Acc: 89.92%
 Best Val Accuracy: 89.92%


============================================================
Run: lr=0.0025_layers=2_h=192_e=200
LR=0.0025, num_layers=2, epochs=5
============================================================
Epoch 1/5 | Train Loss: 1.5613 | Val Loss: 0.6311 | Val Acc: 80.42%
Epoch 2/5 | Train Loss: 0.4358 | Val Loss: 0.4003 | Val Acc: 88.38%
Epoch 3/5 | Train Loss: 0.1622 | Val Loss: 0.4101 | Val Acc: 89.76%
Epoch 4/5 | Train Loss: 0.0837 | Val Loss: 0.4485 | Val Acc: 88.49%
Epoch 5/5 | Train Loss: 0.0566 | Val Loss: 0.4431 | Val Acc: 90.03%
 Best Val Accuracy: 90.03%


============================================================
RÉSUMÉ DES RÉSULTATS
============================================================

Run                                 |     LR | Layers |  Val Acc | Val Loss
--------------------------------------------------------------------------------
lr=0.002_layers=2_h=192_e=200       | 0.0020 |      2 |   90.34% |   0.4470
lr=0.0025_layers=2_h=192_e=200      | 0.0025 |      2 |   90.03% |   0.4431
lr=0.0025_layers=1_h=192_e=200      | 0.0025 |      1 |   89.92% |   0.4161
lr=0.002_layers=1_h=192_e=200       | 0.0020 |      1 |   89.87% |   0.4301
lr=0.0015_layers=1_h=192_e=200      | 0.0015 |      1 |   88.75% |   0.4291
lr=0.0015_layers=2_h=192_e=200      | 0.0015 |      2 |   88.22% |   0.4739

 Résultats sauvegardés dans artifacts/iteration_supplementaire_results.json
 Logs TensorBoard dans runs/iteration_supplementaire/

============================================================
MEILLEURE CONFIGURATION
============================================================
Run: lr=0.002_layers=2_h=192_e=200
LR: 0.002
num_layers: 2
Val Accuracy: 90.34%
Val Loss: 0.4470
============================================================
"""