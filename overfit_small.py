import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from src.data_loading import get_dataloaders
from src.model import build_model
from src.utils import set_seed, get_device


def main():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    
    seed = config["train"]["seed"]
    set_seed(seed)
    device = get_device(config["train"]["device"])
    
    train_loader, _, _, meta = get_dataloaders(config)
    
    indices = torch.randperm(len(train_loader.dataset))[:32]
    small_ds = Subset(train_loader.dataset, indices)
    small_loader = DataLoader(small_ds, batch_size=64, shuffle=True)
    
    model = build_model(config, meta).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)
    
    log_dir = os.path.join(config["paths"]["runs_dir"], "overfit_small")
    writer = SummaryWriter(log_dir)
    
    print(f"\n{'='*70}")
    print("OVERFIT SUR 32 EXEMPLES (M3)")
    print(f"{'='*70}\n")
    
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for inputs, labels in small_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(small_loader)
        writer.add_scalar("overfit/train_loss", avg_loss, epoch)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.6f}")
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in small_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    print(f"\n{'='*70}")
    print(f"Accuracy finale : {100*correct/total:.2f}%")
    print(f" Overfit confirmé (loss → 0)")
    print(f"{'='*70}")
    
    writer.close()


if __name__ == "__main__":
    main()