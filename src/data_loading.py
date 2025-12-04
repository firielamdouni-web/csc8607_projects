"""
Chargement des données pour 20 Newsgroups (Classification de texte).

Signature imposée :
get_dataloaders(config: dict) -> (train_loader, val_loader, test_loader, meta: dict)

Le dictionnaire meta doit contenir au minimum :
- "num_classes": int
- "input_shape": tuple
"""

import os
import re
import pickle
import codecs
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
from collections import Counter


class NewsGroupDataset(Dataset):
    """Dataset wrapper pour 20 Newsgroups (texte encodé en indices)."""

    def __init__(self, texts, labels, word2idx, max_seq_len):
        """
        Args:
            texts: liste de textes bruts
            labels: liste d'indices de classes (0-19)
            word2idx: dictionnaire {mot: indice}
            max_seq_len: longueur fixe des séquences (400)
        """
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.max_seq_len = max_seq_len
        self.unk_idx = word2idx.get("<unk>", 1)

    def tokenize_and_encode(self, text):
        """Tokenize et encode un texte en indices."""
        # Lowercase + suppression ponctuation
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        tokens = text.split()

        # Encode : mots connus → indice, sinon <unk>
        encoded = [self.word2idx.get(token, self.unk_idx) for token in tokens]
        return encoded

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Encode le texte
        encoded = self.tokenize_and_encode(text)

        # Padding/Truncation à max_seq_len
        if len(encoded) < self.max_seq_len:
            encoded = encoded + [0] * (self.max_seq_len - len(encoded))
        else:
            encoded = encoded[:self.max_seq_len]

        # Retourner en tensors
        seq_tensor = torch.tensor(encoded, dtype=torch.long)  # (max_seq_len,)
        label_tensor = torch.tensor(label, dtype=torch.long)  # ()

        return seq_tensor, label_tensor


def build_vocabulary(texts, min_freq=2, max_vocab_size=50000):
    """
    Construit vocabulaire à partir des textes d'entraînement.
    
    Tokens spéciaux :
    - <pad> (index 0) : padding
    - <unk> (index 1) : mots inconnus
    
    Returns:
        tuple: (word2idx, idx2word)
    """
    # Tokenize tous les textes
    all_tokens = []
    for text in texts:
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        tokens = text.split()
        all_tokens.extend(tokens)

    # Comptage des fréquences
    token_counts = Counter(all_tokens)

    # Filtrer par fréquence minimale
    filtered_tokens = [
        token for token, count in token_counts.items() if count >= min_freq
    ]

    # Limiter la taille du vocabulaire
    filtered_tokens = filtered_tokens[:max_vocab_size]

    # Ajouter tokens spéciaux
    special_tokens = ["<pad>", "<unk>"]
    vocab = special_tokens + filtered_tokens

    # Créer mappings
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}

    return word2idx, idx2word


def _load_cached_newsgroups(cache_path):
    """
    Charge 20 Newsgroups depuis PKL/PKZ.
    
    Détecte automatiquement :
    - PKZ (compressé) → décompresse zlib
    - PKL (brut) → charge directement
    """
    print(f"[data_loading] Chargement depuis cache: {cache_path}")

    with open(cache_path, "rb") as f:
        content = f.read()

    # Détection format
    if content[:2] == b'x\x9c':  # PKZ compressé
        print(f"  [Format détecté] PKZ (compressé avec zlib)")
        uncompressed_content = codecs.decode(content, "zlib_codec")
        cache = pickle.loads(uncompressed_content)
    else:  # PKL brut
        print(f"  [Format détecté] PKL (brut)")
        cache = pickle.loads(content)

    # Extraction train/test
    train_data = cache["train"]
    test_data = cache["test"]

    texts_train = train_data.data
    labels_train = train_data.target
    target_names = train_data.target_names
    num_classes = len(target_names)

    texts_test = test_data.data
    labels_test = test_data.target

    print(f"  ✓ Chargé: Train={len(texts_train)}, Test={len(texts_test)}, Classes={num_classes}")

    return texts_train, labels_train, texts_test, labels_test, target_names, num_classes


def get_dataloaders(config: dict):
    """
    Crée et retourne les DataLoaders (train/val/test) et métadonnées.
    
    Pipeline :
    1. Charger dataset 20 Newsgroups depuis PKL/PKZ
    2. Combiner train+test et refaire les splits stratifiés
    3. Construire vocabulaire à partir du train set
    4. Créer Datasets avec tokenization/encoding
    5. Appliquer mode overfit_small si activé
    6. Créer DataLoaders
    7. Retourner loaders + métadonnées
    """

    # Configuration
    dataset_cfg = config["dataset"]
    train_cfg = config["train"]
    preprocess_cfg = config.get("preprocess", {})

    seed = train_cfg["seed"]
    batch_size = train_cfg["batch_size"]
    num_workers = dataset_cfg["num_workers"]
    shuffle = dataset_cfg["shuffle"]

    train_ratio = dataset_cfg["split"]["train"]
    val_ratio = dataset_cfg["split"]["val"]
    test_ratio = dataset_cfg["split"]["test"]

    max_seq_len = preprocess_cfg.get("max_seq_len", 400)
    min_freq = preprocess_cfg.get("min_freq", 2)
    max_vocab_size = preprocess_cfg.get("max_vocab_size", 50000)

    print(f"[data_loading] Chargement du dataset 20 Newsgroups...")
    print(f"  - max_seq_len: {max_seq_len}")
    print(f"  - min_freq: {min_freq}")
    print(f"  - max_vocab_size: {max_vocab_size}")
    print(f"  - seed: {seed}")

    # ===== 1. Charger depuis PKL/PKZ =====
    possible_paths = [
        os.path.join(dataset_cfg["root"], "20news-bydate.pkl"),
        os.path.join(dataset_cfg["root"], "20news-bydate_py3.pkz"),
        os.path.join(dataset_cfg["root"], "20news-bydate.pkz"),
    ]

    cache_path = None
    for path in possible_paths:
        if os.path.exists(path):
            cache_path = path
            break

    if cache_path is None:
        raise FileNotFoundError(
            f"[data_loading] ❌ Aucun fichier PKL/PKZ dans {dataset_cfg['root']}\n"
            f"Fichiers attendus:\n"
            f"  - 20news-bydate.pkl\n"
            f"  - 20news-bydate_py3.pkz\n"
            f"  - 20news-bydate.pkz"
        )

    texts_train, labels_train, texts_test, labels_test, target_names, num_classes = _load_cached_newsgroups(cache_path)

    # ===== 2. Split stratifié (train+val vs test) =====
    all_texts = texts_train + texts_test
    all_labels = np.concatenate([labels_train, labels_test])

    print(f"[data_loading] Pool total: {len(all_texts)} exemples")

    test_size_abs = test_ratio / (train_ratio + val_ratio + test_ratio)
    texts_trainval, texts_test_new, labels_trainval, labels_test_new = train_test_split(
        all_texts,
        all_labels,
        test_size=test_size_abs,
        random_state=seed,
        stratify=all_labels,
    )

    # ===== 3. Split stratifié (train vs val) =====
    val_size_rel = val_ratio / (train_ratio + val_ratio)
    texts_train_new, texts_val, labels_train_new, labels_val = train_test_split(
        texts_trainval,
        labels_trainval,
        test_size=val_size_rel,
        random_state=seed,
        stratify=labels_trainval,
    )

    print(f"  ✓ Split final: Train={len(texts_train_new)}, Val={len(texts_val)}, Test={len(texts_test_new)}")

    # ===== 4. Construire vocabulaire =====
    print(f"[data_loading] Construction du vocabulaire...")
    word2idx, idx2word = build_vocabulary(
        texts_train_new, min_freq=min_freq, max_vocab_size=max_vocab_size
    )
    vocab_size = len(word2idx)
    print(f"  ✓ Taille vocabulaire: {vocab_size}")

    # ===== 5. Créer Datasets =====
    train_dataset = NewsGroupDataset(texts_train_new, labels_train_new, word2idx, max_seq_len)
    val_dataset = NewsGroupDataset(texts_val, labels_val, word2idx, max_seq_len)
    test_dataset = NewsGroupDataset(texts_test_new, labels_test_new, word2idx, max_seq_len)

    # ===== 6. Mode overfit_small (comme ton ami) =====
    if train_cfg.get("overfit_small", False):
        n_samples = 32
        indices = torch.randperm(len(train_dataset))[:n_samples]
        train_dataset = Subset(train_dataset, indices)
        print(f"⚠️ Mode overfit_small: train réduit à {n_samples} exemples")

    # ===== 7. DataLoaders =====
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # ===== 8. Métadonnées =====
    meta = {
        "num_classes": num_classes,
        "input_shape": (max_seq_len,),  # Forme d'une séquence : (400,)
        "vocab_size": vocab_size,
        "word2idx": word2idx,
        "idx2word": idx2word,
        "class_names": list(target_names),
    }

    print(f"[data_loading] ✓ Complet.")
    print(f"  - num_classes: {num_classes}")
    print(f"  - input_shape: {meta['input_shape']}")
    print(f"  - vocab_size: {vocab_size}")

    return train_loader, val_loader, test_loader, meta