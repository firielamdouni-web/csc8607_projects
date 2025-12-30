"""
Pré-traitements.

Signature imposée :
get_preprocess_transforms(config: dict) -> objet/transform callable
"""

def get_preprocess_transforms(config: dict):
    """Retourne les transformations de pré-traitement. À implémenter."""
    """Le preprocessing est déjà intégré dans data_loading.py (voir la méthode NewsGroupDataset.tokenize())"""
    return None