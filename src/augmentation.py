"""
Data augmentation.

Pour 20 Newsgroups (classification de texte) :
Pas d'augmentations pertinentes. Dropout dans le modèle suffit.
"""

def get_augmentation_transforms(config: dict):
    """Retourne les transformations d'augmentation."""
    return None