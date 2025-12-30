"""
Chargement des données.

Signature imposée :
get_dataloaders(config: dict) -> (train_loader, val_loader, test_loader, meta: dict)

Le dictionnaire meta doit contenir au minimum :
- "num_classes": int
- "input_shape": tuple (ex: (3, 32, 32) pour des images)
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
    def __init__(self, texts, labels, word2idx, max_seq_len):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.max_seq_len = max_seq_len
        self.unk_idx = word2idx.get("<unk>", 1)

    def tokenize(self, text):
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        tokens = text.split()
        encoded = [self.word2idx.get(t, self.unk_idx) for t in tokens]
        
        if len(encoded) < self.max_seq_len:
            encoded += [0] * (self.max_seq_len - len(encoded))
        else:
            encoded = encoded[:self.max_seq_len]
        
        return torch.tensor(encoded, dtype=torch.long)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        seq = self.tokenize(self.texts[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return seq, label


def build_vocab(texts, min_freq=2, max_size=50000):
    all_tokens = []
    for text in texts:
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        all_tokens.extend(text.split())
    
    counts = Counter(all_tokens)
    vocab = [w for w, c in counts.items() if c >= min_freq][:max_size]
    
    word2idx = {"<pad>": 0, "<unk>": 1}
    word2idx.update({w: i+2 for i, w in enumerate(vocab)})
    idx2word = {i: w for w, i in word2idx.items()}
    
    return word2idx, idx2word


def load_cached(path):
    with open(path, "rb") as f:
        content = f.read()
    
    if content[:2] == b'x\x9c':
        content = codecs.decode(content, "zlib_codec")
    
    cache = pickle.loads(content)
    train_data = cache["train"]
    test_data = cache["test"]
    
    return (train_data.data, train_data.target,
            test_data.data, test_data.target,
            train_data.target_names)


def get_dataloaders(config: dict):
    cfg = config["dataset"]
    train_cfg = config["train"]
    preproc = config.get("preprocess", {})
    
    seed = train_cfg["seed"]
    batch_size = train_cfg["batch_size"]
    max_len = preproc.get("max_seq_len", 400)
    min_freq = preproc.get("min_freq", 2)
    max_vocab = preproc.get("max_vocab_size", 50000)
    
    cache_paths = [
        os.path.join(cfg["root"], "20news-bydate.pkl"),
        os.path.join(cfg["root"], "20news-bydate_py3.pkz"),
        os.path.join(cfg["root"], "20news-bydate.pkz"),
    ]
    
    cache_path = next((p for p in cache_paths if os.path.exists(p)), None)
    if not cache_path:
        raise FileNotFoundError(f"Dataset not found in {cfg['root']}")
    
    texts_train, labels_train, texts_test, labels_test, classes = load_cached(cache_path)
    
    all_texts = texts_train + texts_test
    all_labels = np.concatenate([labels_train, labels_test])
    
    test_ratio = cfg["split"]["test"] / sum(cfg["split"].values())
    texts_tv, texts_test, labels_tv, labels_test = train_test_split(
        all_texts, all_labels, test_size=test_ratio, 
        stratify=all_labels, random_state=seed
    )
    
    val_ratio = cfg["split"]["val"] / (cfg["split"]["train"] + cfg["split"]["val"])
    texts_train, texts_val, labels_train, labels_val = train_test_split(
        texts_tv, labels_tv, test_size=val_ratio,
        stratify=labels_tv, random_state=seed
    )
    
    word2idx, idx2word = build_vocab(texts_train, min_freq, max_vocab)
    
    train_ds = NewsGroupDataset(texts_train, labels_train, word2idx, max_len)
    val_ds = NewsGroupDataset(texts_val, labels_val, word2idx, max_len)
    test_ds = NewsGroupDataset(texts_test, labels_test, word2idx, max_len)
    
    if train_cfg.get("overfit_small", False):
        indices = torch.randperm(len(train_ds))[:32]
        train_ds = Subset(train_ds, indices)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=cfg["shuffle"],
                             num_workers=cfg["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                           num_workers=cfg["num_workers"], pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                            num_workers=cfg["num_workers"], pin_memory=True)
    
    meta = {
        "num_classes": len(classes),
        "input_shape": (max_len,),
        "vocab_size": len(word2idx),
        "word2idx": word2idx,
        "idx2word": idx2word,
        "class_names": list(classes),
    }
    
    return train_loader, val_loader, test_loader, meta