"""
Évaluation — à implémenter.

Doit exposer un main() exécutable via :
    python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best.ckpt

Exigences minimales :
- charger le modèle et le checkpoint
- calculer et afficher/consigner les métriques de test
"""

import argparse
import yaml
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from src.data_loading import get_dataloaders
from src.model import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    
    # Charger la configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Préparer le device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Charger les données (on ne garde que test_loader)
    print("Chargement des données...")
    _, _, test_loader, meta = get_dataloaders(config)
    print(f"Test set: {len(test_loader.dataset)} exemples\n")
    
    # Construire le modèle
    print("Construction du modèle...")
    model = build_model(config, meta)
    model = model.to(device)
    
    # Charger le checkpoint
    print(f"Chargement du checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Checkpoint epoch: {checkpoint.get('epoch', '?')}")
    best_val = checkpoint.get('best_val_acc')
    if best_val is not None:
        print(f"Best val acc: {best_val:.4f}\n")
    else:
        print("Best val acc: N/A\n")
    
    # Évaluation sur le test set
    print("="*70)
    print("ÉVALUATION SUR TEST SET")
    print("="*70)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            logits = model(inputs)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculer les métriques
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
    print(f"\nRÉSULTATS:")
    print(f"  Test Accuracy:  {accuracy*100:.2f}%")
    print(f"  F1 Macro:       {f1_macro:.4f}")
    print(f"  F1 Weighted:    {f1_weighted:.4f}")
    print("="*70)
    
    # Rapport détaillé par classe
    if meta.get('class_names'):
        print("\nRAPPORT DE CLASSIFICATION:\n")
        report = classification_report(all_labels, all_preds, 
                                       target_names=meta['class_names'],
                                       digits=4)
        print(report)
    
    print("\n Évaluation terminée!")


if __name__ == "__main__":
    main()

"""
(.venv) PS C:\Users\Amdouni\OneDrive - IMTBS-TSP\Documents\PFA-deeplea\csc8607_projects-main> python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best.ckpt

Device: cpu
Chargement des données...
Test set: 1885 exemples

Construction du modèle...

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

Chargement du checkpoint: artifacts/best.ckpt
Checkpoint epoch: 15
Best val acc: N/A

======================================================================
ÉVALUATION SUR TEST SET
======================================================================
C:\Users\Amdouni\OneDrive - IMTBS-TSP\Documents\PFA-deeplea\.venv\Lib\site-packages\torch\utils\data\dataloader.py:668: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
  warnings.warn(warn_msg)

RÉSULTATS:
  Test Accuracy:  91.14%
  F1 Macro:       0.9106
  F1 Weighted:    0.9117
======================================================================

RAPPORT DE CLASSIFICATION:

                          precision    recall  f1-score   support     

             alt.atheism     0.9114    0.9000    0.9057        80     
           comp.graphics     0.8696    0.8247    0.8466        97     
 comp.os.ms-windows.misc     0.7965    0.9091    0.8491        99     
comp.sys.ibm.pc.hardware     0.8316    0.8061    0.8187        98     
   comp.sys.mac.hardware     0.9310    0.8438    0.8852        96     
          comp.windows.x     0.9200    0.9293    0.9246        99     
            misc.forsale     0.8900    0.9082    0.8990        98     
               rec.autos     0.8980    0.8889    0.8934        99     
         rec.motorcycles     0.9333    0.9800    0.9561       100
      rec.sport.baseball     0.9588    0.9394    0.9490        99
        rec.sport.hockey     0.9894    0.9300    0.9588       100
               sci.crypt     0.9892    0.9293    0.9583        99
         sci.electronics     0.8532    0.9490    0.8986        98
                 sci.med     0.9048    0.9596    0.9314        99
               sci.space     1.0000    0.9596    0.9794        99
  soc.religion.christian     0.8750    0.9100    0.8922       100
      talk.politics.guns     0.9149    0.9451    0.9297        91
   talk.politics.mideast     0.9889    0.9468    0.9674        94
      talk.politics.misc     0.9577    0.8831    0.9189        77
      talk.religion.misc     0.8438    0.8571    0.8504        63

                accuracy                         0.9114      1885
               macro avg     0.9128    0.9100    0.9106      1885
            weighted avg     0.9136    0.9114    0.9117      1885


 Évaluation terminée!
 """