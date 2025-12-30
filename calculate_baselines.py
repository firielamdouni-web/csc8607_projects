import yaml
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score
from src.data_loading import get_dataloaders


def main():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    
    train_loader, _, _, meta = get_dataloaders(config)
    
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.tolist())
    
    # Baseline 1: Classe majoritaire
    class_counts = Counter(all_labels)
    majority_class = class_counts.most_common(1)[0][0]
    preds_maj = [majority_class] * len(all_labels)
    acc_maj = accuracy_score(all_labels, preds_maj)
    
    print(f"\n{'='*60}")
    print("BASELINES")
    print(f"{'='*60}")
    print(f"Classe majoritaire : {majority_class} ({meta['class_names'][majority_class]})")
    print(f"  Accuracy : {acc_maj:.4f} ({100*acc_maj:.2f}%)")
    
    # Baseline 2: Aléatoire
    np.random.seed(42)
    preds_rand = np.random.randint(0, meta['num_classes'], len(all_labels))
    acc_rand = accuracy_score(all_labels, preds_rand)
    
    print(f"\nAléatoire uniforme :")
    print(f"  Accuracy : {acc_rand:.4f} ({100*acc_rand:.2f}%)")
    print(f"  Théorique: {1/meta['num_classes']:.4f} (1/{meta['num_classes']})")


if __name__ == "__main__":
    main()