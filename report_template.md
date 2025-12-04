# Rapport de projet — CSC8607 : Introduction au Deep Learning

> **Consignes générales**
> - Tenez-vous au **format** et à l’**ordre** des sections ci-dessous.
> - Intégrez des **captures d’écran TensorBoard** lisibles (loss, métriques, LR finder, comparaisons).
> - Les chemins et noms de fichiers **doivent** correspondre à la structure du dépôt modèle (ex. `runs/`, `artifacts/best.ckpt`, `configs/config.yaml`).
> - Répondez aux questions **numérotées** (D1–D11, M0–M9, etc.) directement dans les sections prévues.

---

## 0) Informations générales

- **Étudiant·e** : AMDOUNI, Firiel
- **Projet** : 20 Newsgroups (20 catégories de textes) avec BiGRU et attention moyenne pondérée
- **Dépôt Git** : _URL publique_
- **Environnement** : `python == 3.13.3`, `torch == 2.9.0+cpu`, `cuda == None`  
- **Commandes utilisées** :
  - Entraînement : `python -m src.train --config configs/config.yaml`
  - LR finder : `python -m src.lr_finder --config configs/config.yaml`
  - Grid search : `python -m src.grid_search --config configs/config.yaml`
  - Évaluation : `python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best.ckpt`

---

## 1) Données

### 1.1 Description du dataset
- **Source** (lien) :
  - Scikit-learn : https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html
  - Dataset officiel : http://qwone.com/~jason/20Newsgroups/

- **Type d’entrée** (image / texte / audio / séries) : Texte (articles de discussion)
- **Tâche** (multiclasses, multi-label, régression) : Classification multiclasse (20 catégories)
- **Dimensions d’entrée attendues** (`meta["input_shape"]`) : `(400,)` 
  - 400 = longueur maximale des séquences (en nombre de tokens)

- **Nombre de classes** (`meta["num_classes"]`) : 20

- **Format détaillé** :
  - Chaque exemple = texte brut d'un article (variable length)
  - Après preprocessing : séquence d'indices d'indices (0-50001)
  - Après padding/truncation : tenseur de shape `(400,)` contenant des indices entiers

**D1.** Quel dataset utilisez-vous ? D’où provient-il et quel est son format (dimensions, type d’entrée) ?

Le dataset utilisé est le 20 Newsgroups provenant de scikit-learn. Il contient 18 846 articles de discussion issus de 20 catégories de forums (alt.atheism, comp.graphics, sci.med, talk.politics.guns, etc.). Chaque article est un texte brut de longueur variable. Le format des données après preprocessing est : (N, T) = (nombre d'exemples, longueur séquence fixe) = (18846, 400), où chaque position contient un indice de vocabulaire (0–50001). Les tokens spéciaux sont <pad> (index 0) et <unk> (index 1) pour les mots inconnus.


### 1.2 Splits et statistiques

| Split | #Exemples  | Particularités (déséquilibre, longueur moyenne, etc.) |
|------:|----------: |-------------------------------------------------------|
| Train | 15 076     |  Distribution équilibrée (3.33% à 5.30% par classe)   |
| Val   | 1 885      |  Split stratifié, même distribution que train           |
| Test  | 1 885      |  Split stratifié, même distribution que train
            |

**D2.** Donnez la taille de chaque split et le nombre de classes.  
  Train : 15 076 exemples (80%)
  Val : 1 885 exemples (10%)
  Test : 1 885 exemples (10%)
  Nombre de classes : 20
  Meta :
  
meta = {
    "num_classes": 20,
    "input_shape": (400,),
    "vocab_size": 50002
}

**D3.** Si vous avez créé un split (ex. validation), expliquez **comment** (stratification, ratio, seed).

Un split stratifié a été effectué en deux étapes :

Étape 1 : Pool total = train original (11 314) + test original (7 532) = 18 846 exemples

Split 80/10/10 stratifié :

Pool train+val : 80% × 18 846 = 15 076 exemples
Pool test : 10% × 18 846 = 1 885 exemples
Stratification sur tous les labels pour préserver la distribution des 20 classes
Étape 2 : Sub-split du pool train+val (90/10 pour train/val) :

Train final : 90% × 15 076 = 15 076 exemples
Val final : 10% × 15 076 = 1 885 exemples
Stratification appliquée à nouveau pour conserver l'équilibre des classes
Seed utilisée : 42 pour garantir la reproductibilité
Méthode : train_test_split() de scikit-learn avec stratify=all_labels

La stratification préserve les proportions de chaque catégorie dans chaque split, ce qui est crucial pour éviter un biais d'entraînement vers les classes majoritaires et pour fiabiliser l'évaluation.


**D4.** Donnez la **distribution des classes** (graphique ou tableau) et commentez en 2–3 lignes l’impact potentiel sur l’entraînement.  

Distribution des classes sur le train set (15 076 exemples) :

| Classe | Catégorie                      | # Exemples | % du train |
|------:|--------------------------------|------------|-----------|
| 0     | alt.atheism                    | 639        | 4.24%     |
| 1     | comp.graphics                  | 779        | 5.17%     |
| 2     | comp.os.ms-windows.misc        | 788        | 5.23%     |
| 3     | comp.sys.ibm.pc.hardware       | 786        | 5.21%     |
| 4     | comp.sys.mac.hardware          | 771        | 5.11%     |
| 5     | comp.windows.x                 | 790        | 5.24%     |
| 6     | misc.forsale                   | 780        | 5.17%     |
| 7     | rec.autos                      | 792        | 5.25%     |
| 8     | rec.motorcycles                | 796        | 5.28%     |
| 9     | rec.sport.baseball             | 795        | 5.27%     |
| 10    | rec.sport.hockey               | 799        | 5.30%     |
| 11    | sci.crypt                      | 793        | 5.26%     |
| 12    | sci.electronics                | 788        | 5.23%     |
| 13    | sci.med                        | 792        | 5.25%     |
| 14    | sci.space                      | 789        | 5.23%     |
| 15    | soc.religion.christian         | 797        | 5.29%     |
| 16    | talk.politics.guns             | 728        | 4.83%     |
| 17    | talk.politics.mideast          | 752        | 4.99%     |
| 18    | talk.politics.misc             | 620        | 4.11%     |
| 19    | talk.religion.misc             | 502        | 3.33%     |

Analyse : La distribution est très équilibrée, variant de 3.33% à 5.30% (ratio majorité/minorité = 1.59:1). Cela signifie que le dataset ne souffre pas d'un déséquilibre drastique. L'impact sur l'entraînement est minimal : le modèle n'aura pas de biais significatif vers les classes majoritaires, et la métrique d'accuracy seule est suffisante pour évaluer la performance globale. Aucune pondération de classe ou sursampling n'est nécessaire.

**D5.** Mentionnez toute particularité détectée (tailles variées, longueurs variables, multi-labels, etc.).
Le dataset 20 Newsgroups présente les caractéristiques suivantes après analyse complète :

Longueurs de texte (en tokens) :

Minimum : 18 tokens
Moyenne : 214.8 tokens
Médiane : 189.0 tokens
Maximum (observé) : 400 tokens
Justification de max_seq_len=400 : Couvre 100% des textes (aucune perte par truncation)
Distribution des longueurs :

~25% des textes < 145 tokens (beaucoup de padding)
~50% des textes < 189 tokens (médiane)
~75% des textes < 255 tokens
~100% des textes ≤ 400 tokens (aucun truncation)
Implication computationnelle : ~40% des positions seront du padding (tokens index 0), ce qui est normal et géré par le modèle (embedding layer ignore les positions padées via attention masking dans le BiGRU)
Vocabulaire :

Taille : 50 002 tokens
Tokens spéciaux : <pad> (index 0), <unk> (index 1)
Mots rares éliminés : min_freq=2 (mots apparaissant ≥2 fois conservés)
Mots hors vocabulaire → mappés à <unk> (index 1)
Impact : Vocabulaire compact (~50k vs. potentiellement 200k+), réduit le bruit lexical et améliore la généralisation
Format après preprocessing :

Chaque texte = liste d'indices (0 à 50 001)
Shape d'une séquence : (400,) → tenseur torch.long
Shape d'un batch (train) : (64, 400) (64 séquences de 400 tokens)
Labels : indices 0–19 (classe de l'article)
Classification simple (pas de multi-label) :

Chaque article appartient à exactement 1 catégorie (mutually exclusive)
Pas d'étiquettes manquantes ou ambiguës
Loss function appropriée : CrossEntropyLoss

Cohérence check :

 Min/max indices observés [0, 49 939] → dans range attendu [0, 50 001]
 Distribution classes équilibrée sur train/val/test (stratification respectée)
 Aucune séquence avec NaN ou valeur aberrante
 Batch shape (64, 400) cohérent avec meta["input_shape"]=(400,)
 Labels valides [0-19] cohérents avec meta["num_classes"]=20

### 1.3 Prétraitements (preprocessing) — _appliqués à train/val/test_

Listez précisément les opérations et paramètres (valeurs **fixes**) :
**Pipeline de preprocessing (NLP) :**

1. **Tokenization** :
   - **Opération** : Lowercase + suppression ponctuation + split whitespace
   - **Paramètres** :
     - Lowercase : `text.lower()` (tous les caractères → minuscules)
     - Suppression ponctuation : `re.sub(r"[^\w\s]", " ", text)` (garde uniquement alphanumériques + whitespace)
     - Split : `text.split()` (délimiteur whitespace)
   - **Justification** : Réduit vocabulaire (ex. "The" et "the" → même token), élimine bruit ponctuation. Standard NLP.

2. **Encoding (Vocabulaire)** :
   - **Opération** : Transformer tokens en indices numériques
   - **Paramètres** :
     - Vocabulaire : 50 002 tokens (50 000 + 2 tokens spéciaux)
     - Mots connus → indice dans `word2idx` (0 à 50 001)
     - Mots inconnus → `<unk>` (index 1)
     - Token padding → `<pad>` (index 0)
   - **Justification** : Entrée obligatoire pour embedding layer (exige indices entiers). Vocabulaire limité réduit bruit.

3. **Padding / Truncation** :
   - **Opération** : Fixer longueur séquence à `max_seq_len = 400`
   - **Paramètres** :
     - Si `len(tokens) < 400` : ajouter padding (index 0) à droite
     - Si `len(tokens) ≥ 400` : garder premiers 400 tokens (truncation)
   - **Justification** : BiGRU exige batch de shapes uniformes. Choix de 400 basé sur analyse distribution (couvre 100% textes).

4. **Construction Vocabulaire** :
   - **Opération** : Construire `word2idx` et `idx2word` depuis train set uniquement
   - **Paramètres** :
     - `min_freq = 2` : garder mots avec fréquence ≥ 2
     - `max_vocab_size = 50 000` : limiter vocabulaire (performance mémoire)
     - Tokens spéciaux : `["<pad>", "<unk>"]` ajoutés en premier
   - **Justification** : Réduire bruit (mots hapax) et compute (GPU memory). Standard NLP.

5. **Conversion Tenseur** :
   - **Opération** : Transformer liste d'indices en tenseur PyTorch
   - **Paramètres** :
     - Type : `torch.long` (requis pour `nn.Embedding`)
     - Shape : `(max_seq_len,)` = `(400,)`
   - **Justification** : Format obligatoire pour PyTorch DataLoader et modèle.

- Vision : resize = __, center-crop = __, normalize = (mean=__, std=__)…
- Audio : resample = __ Hz, mel-spectrogram (n_mels=__, n_fft=__, hop_length=__), AmplitudeToDB…
- NLP : tokenizer = __, vocab = __, max_length = __, padding/truncation = __…
- Séries : normalisation par canal, fenêtrage = __…

**D6.** Quels **prétraitements** avez-vous appliqués (opérations + **paramètres exacts**) et **pourquoi** ?  *

Les prétraitements appliqués sont standards pour la classification NLP :

1. **Tokenization** (lowercase + suppression ponctuation + split) : Réduit vocabulaire de ~30% sans perte sémantique, standard NLP.
Paramètres : text.lower() + re.sub(r"[^\w\s]", " ", text) + text.split()
2. **Encoding** (word2idx) : Obligatoire pour embedding layer (exige indices entiers).
Paramètres : min_freq=2 (élimine hapax = bruit), max_vocab_size=50k (limite mémoire), <pad>=0 et <unk>=1 (tokens spéciaux).
3. **Padding/Truncation** (max_seq_len=400) : Nécessaire pour batch uniforme dans BiGRU. Choix de 400 basé sur analyse distribution (couvre 100% textes, médiane=189).
Paramètres : Padding à droite (compatible BiGRU forward pass), truncation garde premiers 400 tokens.
4. **Vocabulaire** (min_freq=2, max_vocab_size=50k) : Élimine bruit (mots hapax), réduit mémoire, améliore généralisation.
Paramètres : 50 002 tokens (50k + 2 spéciaux), construit avec min_freq=2.
5. **Tokens spéciaux** (<pad>=0, <unk>=1) : Gestion uniforme padding et mots inconnus, nécessaires pour robustesse.
Paramètres : Shape (400,) par séquence, batch shape (64, 400).

**Justification :** Ces choix sont fixes (non recherchés en grid search) et adaptés au NLP classique. Ils optimisent le compromis entre taille vocabulaire, mémoire GPU, et performance de généralisation.


**D7.** Les prétraitements diffèrent-ils entre train/val/test (ils ne devraient pas, sauf recadrage non aléatoire en val/test) ?
Les prétraitements sont strictement identiques pour train/val/test. Aucune différence d'opération ou de paramètres.

Détail par étape :
 Étape | Train                      | Val | Test | Identique?
|------:|--------------------------------|------------|-----------|
| Tokenization	    | Lowercase + ponctuation + split	 | Idem	     | Idem    | Oui 
| Encoding | Via word2idx (construit sur train)   | Via même word2idx     | Via même word2idx   | Oui 
| Padding/Truncation	    | max_seq_len=400     | Idem        | Idem     | Oui
| Vocabulaire	   | Construit sur train        | Réutilisé tel quel        | Réutilisé tel quel   | Oui
| Augmentation	  | Aucune (pas d'opération aléatoire)    | Aucune        | Aucune  | Oui


Points critiques validés :
 Vocabulaire fixe : Le word2idx est construit uniquement sur train, puis réutilisé tel quel pour val/test. Les mots nouveaux (absents du train) sont mappés à <unk> (index 1).

 Pas d'augmentation en val/test : Aucune transformation aléatoire (ex. random masking, paraphrasing, word dropout) appliquée en val/test. Seul le preprocessing déterministe (tokenization, encoding, padding) est effectué.

 Cohérence garantie :

Évite fuite d'information (vocabulaire train-only)
Évite biais d'évaluation (pas d'opération aléatoire en val/test)
Respecte les bonnes pratiques NLP

**Vérification**
Code python : 
# Même Dataset class avec même vocabulaire pour tous les splits
dataset_train = NewsGroupDataset(texts_train, labels_train, word2idx, max_seq_len=400)
dataset_val   = NewsGroupDataset(texts_val,   labels_val,   word2idx, max_seq_len=400)  # ← Même word2idx
dataset_test  = NewsGroupDataset(texts_test,  labels_test,  word2idx, max_seq_len=400)  # ← Même word2idx


### 1.4 Augmentation de données — _train uniquement_

- Liste des **augmentations** (opérations + **paramètres** et **probabilités**) :
  - ex. Flip horizontal p=0.5, RandomResizedCrop scale=__, ratio=__ …
  - Audio : time/freq masking (taille, nb masques) …
  - Séries : jitter amplitude=__, scaling=__ …

**D8.** Quelles **augmentations** avez-vous appliquées (paramètres précis) et **pourquoi** ?  
**D9.** Les augmentations **conservent-elles les labels** ? Justifiez pour chaque transformation retenue.

### 1.5 Sanity-checks

- **Exemples** après preprocessing/augmentation (insérer 2–3 images/spectrogrammes) :

> _Insérer ici 2–3 captures illustrant les données après transformation._

**D10.** Montrez 2–3 exemples et commentez brièvement.  
**D11.** Donnez la **forme exacte** d’un batch train (ex. `(batch, C, H, W)` ou `(batch, seq_len)`), et vérifiez la cohérence avec `meta["input_shape"]`.

---

## 2) Modèle

### 2.1 Baselines

**M0.**
- **Classe majoritaire** — Métrique : `_____` → score = `_____`
- **Prédiction aléatoire uniforme** — Métrique : `_____` → score = `_____`  
_Commentez en 2 lignes ce que ces chiffres impliquent._

### 2.2 Architecture implémentée

- **Description couche par couche** (ordre exact, tailles, activations, normalisations, poolings, résiduels, etc.) :
  - Input → …
  - Stage 1 (répéter N₁ fois) : …
  - Stage 2 (répéter N₂ fois) : …
  - Stage 3 (répéter N₃ fois) : …
  - Tête (GAP / linéaire) → logits (dimension = nb classes)

- **Loss function** :
  - Multi-classe : CrossEntropyLoss
  - Multi-label : BCEWithLogitsLoss
  - (autre, si votre tâche l’impose)

- **Sortie du modèle** : forme = __(batch_size, num_classes)__ (ou __(batch_size, num_attributes)__)

- **Nombre total de paramètres** : `_____`

**M1.** Décrivez l’**architecture** complète et donnez le **nombre total de paramètres**.  
Expliquez le rôle des **2 hyperparamètres spécifiques au modèle** (ceux imposés par votre sujet).


### 2.3 Perte initiale & premier batch

- **Loss initiale attendue** (multi-classe) ≈ `-log(1/num_classes)` ; exemple 100 classes → ~4.61
- **Observée sur un batch** : `_____`
- **Vérification** : backward OK, gradients ≠ 0

**M2.** Donnez la **loss initiale** observée et dites si elle est cohérente. Indiquez la forme du batch et la forme de sortie du modèle.

---

## 3) Overfit « petit échantillon »

- **Sous-ensemble train** : `N = ____` exemples
- **Hyperparamètres modèle utilisés** (les 2 à régler) : `_____`, `_____`
- **Optimisation** : LR = `_____`, weight decay = `_____` (0 ou très faible recommandé)
- **Nombre d’époques** : `_____`

> _Insérer capture TensorBoard : `train/loss` montrant la descente vers ~0._

**M3.** Donnez la **taille du sous-ensemble**, les **hyperparamètres** du modèle utilisés, et la **courbe train/loss** (capture). Expliquez ce qui prouve l’overfit.

---

## 4) LR finder

- **Méthode** : balayage LR (log-scale), quelques itérations, log `(lr, loss)`
- **Fenêtre stable retenue** : `_____ → _____`
- **Choix pour la suite** :
  - **LR** = `_____`
  - **Weight decay** = `_____` (valeurs classiques : 1e-5, 1e-4)

> _Insérer capture TensorBoard : courbe LR → loss._

**M4.** Justifiez en 2–3 phrases le choix du **LR** et du **weight decay**.

---

## 5) Mini grid search (rapide)

- **Grilles** :
  - LR : `{_____ , _____ , _____}`
  - Weight decay : `{1e-5, 1e-4}`
  - Hyperparamètre modèle A : `{_____, _____}`
  - Hyperparamètre modèle B : `{_____, _____}`

- **Durée des runs** : `_____` époques par run (1–5 selon dataset), même seed

| Run (nom explicite) | LR    | WD     | Hyp-A | Hyp-B | Val metric (nom=_____) | Val loss | Notes |
|---------------------|-------|--------|-------|-------|-------------------------|----------|-------|
|                     |       |        |       |       |                         |          |       |
|                     |       |        |       |       |                         |          |       |

> _Insérer capture TensorBoard (onglet HParams/Scalars) ou tableau récapitulatif._

**M5.** Présentez la **meilleure combinaison** (selon validation) et commentez l’effet des **2 hyperparamètres de modèle** sur les courbes (stabilité, vitesse, overfit).

---

## 6) Entraînement complet (10–20 époques, sans scheduler)

- **Configuration finale** :
  - LR = `_____`
  - Weight decay = `_____`
  - Hyperparamètre modèle A = `_____`
  - Hyperparamètre modèle B = `_____`
  - Batch size = `_____`
  - Époques = `_____` (10–20)
- **Checkpoint** : `artifacts/best.ckpt` (selon meilleure métrique val)

> _Insérer captures TensorBoard :_
> - `train/loss`, `val/loss`
> - `val/accuracy` **ou** `val/f1` (classification)

**M6.** Montrez les **courbes train/val** (loss + métrique). Interprétez : sous-apprentissage / sur-apprentissage / stabilité d’entraînement.

---

## 7) Comparaisons de courbes (analyse)

> _Superposez plusieurs runs dans TensorBoard et insérez 2–3 captures :_

- **Variation du LR** (impact au début d’entraînement)
- **Variation du weight decay** (écart train/val, régularisation)
- **Variation des 2 hyperparamètres de modèle** (convergence, plateau, surcapacité)

**M7.** Trois **comparaisons** commentées (une phrase chacune) : LR, weight decay, hyperparamètres modèle — ce que vous attendiez vs. ce que vous observez.

---

## 8) Itération supplémentaire (si temps)

- **Changement(s)** : `_____` (resserrage de grille, nouvelle valeur d’un hyperparamètre, etc.)
- **Résultat** : `_____` (val metric, tendances des courbes)

**M8.** Décrivez cette itération, la motivation et le résultat.

---

## 9) Évaluation finale (test)

- **Checkpoint évalué** : `artifacts/best.ckpt`
- **Métriques test** :
  - Metric principale (nom = `_____`) : `_____`
  - Metric(s) secondaire(s) : `_____`

**M9.** Donnez les **résultats test** et comparez-les à la validation (écart raisonnable ? surapprentissage probable ?).

---

## 10) Limites, erreurs & bug diary (court)

- **Limites connues** (données, compute, modèle) :
- **Erreurs rencontrées** (shape mismatch, divergence, NaN…) et **solutions** :
- **Idées « si plus de temps/compute »** (une phrase) :

---

## 11) Reproductibilité

- **Seed** : `_____`
- **Config utilisée** : joindre un extrait de `configs/config.yaml` (sections pertinentes)
- **Commandes exactes** :

```bash
# Exemple (remplacer par vos commandes effectives)
python -m src.train --config configs/config.yaml --max_epochs 15
python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best.ckpt
````

* **Artifacts requis présents** :

  * [ ] `runs/` (runs utiles uniquement)
  * [ ] `artifacts/best.ckpt`
  * [ ] `configs/config.yaml` aligné avec la meilleure config

---

## 12) Références (courtes)

* PyTorch docs des modules utilisés (Conv2d, BatchNorm, ReLU, LSTM/GRU, transforms, etc.).
* Lien dataset officiel (et/ou HuggingFace/torchvision/torchaudio).
* Toute ressource externe substantielle (une ligne par référence).


