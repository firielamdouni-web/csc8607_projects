"""
Data augmentation

Signature imposée :
get_augmentation_transforms(config: dict) -> objet/transform callable (ou None)
"""

def get_augmentation_transforms(config: dict):
        """Retourne les transformations d'augmentation. À implémenter."""
        """L'augmentation NLP classique est via dropout (déjà dans le modèle)"""
    return None