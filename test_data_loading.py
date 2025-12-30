import yaml
import torch
from src.data_loading import get_dataloaders


def main():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    
    train_loader, val_loader, test_loader, meta = get_dataloaders(config)
    
    print(f"\n{'='*60}")
    print("MÉTADONNÉES")
    print(f"{'='*60}")
    print(f"num_classes: {meta['num_classes']}")
    print(f"input_shape: {meta['input_shape']}")
    print(f"vocab_size: {meta['vocab_size']}")
    
    batch_seq, batch_labels = next(iter(train_loader))
    print(f"\n{'='*60}")
    print("BATCH TRAIN")
    print(f"{'='*60}")
    print(f"inputs.shape : {batch_seq.shape}")
    print(f"labels.shape : {batch_labels.shape}")
    print(f"indices range: [{batch_seq.min()}, {batch_seq.max()}]")
    
    print(f"\n Test réussi")


if __name__ == "__main__":
    main()