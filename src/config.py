# Configuration multimodale centralisée
# Fichier: src/config.py

class MultimodalConfig:
    """Configuration pour le traitement multimodal (texte + images)"""
    
    # Modèle CLIP pour texte et images
    CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
    EMBEDDING_DIMENSIONS = 512


    @classmethod
    def get_model_info(cls):
        return {
            'name': cls.CLIP_MODEL_NAME,
            'dimensions': cls.EMBEDDING_DIMENSIONS,
            'type': 'multimodal'
        }