"""
Construction du modèle (à implémenter par l'étudiant·e).

Signature imposée :
build_model(config: dict) -> torch.nn.Module

"""
import torch
import torch.nn as nn


class BiGRU_Attention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes,
                 num_layers=1, dropout_embed=0.1, dropout_fc=0.5, pad_idx=0):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.dropout_embed = nn.Dropout(dropout_embed)
        
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers,
                         batch_first=True, bidirectional=True)
        
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.dropout_fc = nn.Dropout(dropout_fc)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        embedded = self.dropout_embed(self.embedding(x))
        hidden_states, _ = self.gru(embedded)
        
        attn_scores = self.attention(hidden_states)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.sum(hidden_states * attn_weights, dim=1)
        
        context = self.dropout_fc(context)
        logits = self.fc(context)
        return logits


def build_model(config: dict, meta: dict = None) -> nn.Module:
    model_cfg = config["model"]
    rnn_cfg = model_cfg.get("rnn", {})
    
    vocab_size = meta["vocab_size"] if meta else rnn_cfg.get("vocab_size", 50002)
    num_classes = meta["num_classes"] if meta else model_cfg.get("num_classes", 20)
    
    model = BiGRU_Attention(
        vocab_size=vocab_size,
        embedding_dim=rnn_cfg.get("embedding_dim", 200),
        hidden_size=rnn_cfg.get("hidden_size", 192),
        num_classes=num_classes,
        num_layers=rnn_cfg.get("num_layers", 1),
        dropout_embed=model_cfg.get("dropout", 0.1),
        dropout_fc=0.5,
        pad_idx=rnn_cfg.get("padding_idx", 0)
    )
    
    print(f"\n{'='*70}")
    print("MODÈLE CONSTRUIT : BiGRU_Attention")
    print(f"{'='*70}")
    print(f"  - vocab_size      : {vocab_size}")
    print(f"  - embedding_dim   : {rnn_cfg.get('embedding_dim', 200)} (HYPERPARAMÈTRE 2)")
    print(f"  - hidden_size     : {rnn_cfg.get('hidden_size', 192)} (HYPERPARAMÈTRE 1)")
    print(f"  - num_layers      : {rnn_cfg.get('num_layers', 1)}")
    print(f"  - num_classes     : {num_classes}")
    print(f"  - dropout_embed   : {model_cfg.get('dropout', 0.1)}")
    print(f"  - dropout_rnn     : {model_cfg.get('dropout', 0.1)}")
    print(f"  - dropout_fc      : 0.5")
    print(f"  - padding_idx     : {rnn_cfg.get('padding_idx', 0)}")
    print(f"{'='*70}\n")
    
    return model