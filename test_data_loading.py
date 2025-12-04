"""
Script de test : vérifie que get_dataloaders fonctionne correctement.

Exécution : python test_data_loading.py
"""

import yaml
import statistics
from collections import Counter
from src.data_loading import get_dataloaders


def main():
    """Fonction principale (exécutée uniquement si lancée directement)."""
    
    # Charger la configuration
    print("[test] Chargement de la configuration...")
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Charger les données
    print("[test] Appel de get_dataloaders()...\n")
    train_loader, val_loader, test_loader, meta = get_dataloaders(config)

    # === MÉTADONNÉES ===
    print("\n" + "=" * 60)
    print("MÉTADONNÉES (meta)")
    print("=" * 60)
    print(f"num_classes: {meta['num_classes']}")
    print(f"input_shape: {meta['input_shape']}")
    print(f"vocab_size: {meta['vocab_size']}")
    print(f"\nPremières classes:")
    for i, name in enumerate(meta['class_names'][:5]):
        print(f"  {i}: {name}")
    print("  ...")

    # === VÉRIFICATION D'UN BATCH ===
    print("\n" + "=" * 60)
    print("VÉRIFICATION D'UN BATCH (TRAIN)")
    print("=" * 60)
    batch_seq, batch_labels = next(iter(train_loader))
    print(f"Shape des séquences: {batch_seq.shape}")
    print(f"  → Batch size: {batch_seq.shape[0]}")
    print(f"  → Longueur séquences: {batch_seq.shape[1]}")
    print(f"Shape des labels: {batch_labels.shape}")
    print(f"Exemples de labels: {batch_labels[:8].tolist()}")
    print(f"Valeurs min/max des indices: [{batch_seq.min()}, {batch_seq.max()}]")

    # === DISTRIBUTION DES CLASSES (TRAIN) ===
    print("\n" + "=" * 60)
    print("DISTRIBUTION DES CLASSES (TRAIN)")
    print("=" * 60)
    all_labels_train = []
    for _, labels in train_loader:
        all_labels_train.extend(labels.tolist())

    class_counts = Counter(all_labels_train)
    print(f"Total exemples train: {len(all_labels_train)}\n")
    for cls_id in sorted(class_counts.keys()):
        count = class_counts[cls_id]
        pct = 100 * count / len(all_labels_train)
        print(f"  Classe {cls_id:2d}: {count:4d} exemples ({pct:5.2f}%)")

    # === DISTRIBUTION DES CLASSES (VAL ET TEST) ===
    print("\n" + "=" * 60)
    print("DISTRIBUTION DES CLASSES (VAL et TEST)")
    print("=" * 60)

    all_labels_val = []
    for _, labels in val_loader:
        all_labels_val.extend(labels.tolist())

    all_labels_test = []
    for _, labels in test_loader:
        all_labels_test.extend(labels.tolist())

    print(f"\nVal set: {len(all_labels_val)} exemples")
    print(f"Test set: {len(all_labels_test)} exemples")

    # === STATISTIQUES DE TOKENS ===
    print("\n" + "=" * 60)
    print("STATISTIQUES DE TOKENS")
    print("=" * 60)
    all_seq_lengths = []
    for seq, _ in train_loader:
        # Compte les tokens non-padding (non-zero)
        lengths = (seq != 0).sum(dim=1)
        all_seq_lengths.extend(lengths.tolist())

    print(f"Longueur moyenne (tokens): {statistics.mean(all_seq_lengths):.1f}")
    print(f"Longueur médiane (tokens): {statistics.median(all_seq_lengths):.1f}")
    print(f"Longueur min: {min(all_seq_lengths)}")
    print(f"Longueur max: {max(all_seq_lengths)}")

    print("\n" + "=" * 60)
    print("✓ TEST RÉUSSI !")
    print("=" * 60)


if __name__ == "__main__":
    main()