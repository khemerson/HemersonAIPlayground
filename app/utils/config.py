# app/utils/config.py
"""
Configuration principale pour HemersonAIBuild avec Chainlit
Tous les paramètres sont centralisés ici pour faciliter la maintenance
"""

import os
from typing import Dict, Any, List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Configuration principale
    
    Toutes les valeurs peuvent être modifiées via les variables d'environnement
    ou directement dans ce fichier pour un débutant
    """
    
    # === ENDPOINTS DES SERVICES IA ===
    # Configuration exacte de votre infrastructure HemersonAIBuild
    
    # Mistral Small sur RTX 4090
    MISTRAL_ENDPOINT: str = os.getenv("MISTRAL_ENDPOINT", "http://ollamartx40900:11434")
    
    # Magistral pour analyses approfondies sur RTX 3090
    MAGISTRAL_ENDPOINT: str = os.getenv("MAGISTRAL_ENDPOINT", "http://hybridworker30901:11434")
    
    # Pixtral pour analyse d'images sur RTX 4090
    PIXTRAL_ENDPOINT: str = os.getenv("PIXTRAL_ENDPOINT", "http://hybridworker30903:11434")
    
    # Whisper Large V3 sur RTX 3090 (port 8000)
    WHISPER_ENDPOINT: str = os.getenv("WHISPER_ENDPOINT", "http://hybridworker30901:8000")
    
    # TTS5 sur RTX 3090 (port 8001)
    TTS_ENDPOINT: str = os.getenv("TTS_ENDPOINT", "http://hybridworker30901:8001")
    
    # ComfyUI sur RTX 4090
    COMFYUI_ENDPOINT: str = os.getenv("COMFYUI_ENDPOINT", "http://comfyuirtx40902:8188")
    
    # === CONFIGURATION REDIS ===
    # Redis pour les sessions et le cache
    REDIS_HOST: str = os.getenv("REDIS_HOST", "redis")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    
    # === LIMITATIONS DE SÉCURITÉ ===
    # Protection contre les abus - ajustez selon vos besoins
    MAX_REQUESTS_PER_MINUTE: int = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "15"))
    MAX_REQUESTS_PER_HOUR: int = int(os.getenv("MAX_REQUESTS_PER_HOUR", "100"))
    MAX_TOKENS_PER_REQUEST: int = int(os.getenv("MAX_TOKENS_PER_REQUEST", "3000"))
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "15"))
    
    # Durée de vie des sessions
    SESSION_TIMEOUT_MINUTES: int = int(os.getenv("SESSION_TIMEOUT_MINUTES", "45"))
    
    # === MESSAGES PERSONNALISÉS ===
    # Messages affichés dans votre interface Chainlit
    WELCOME_MESSAGE: str = """
    🚀 **Bienvenue dans HemersonAIBuild Playground**
    
    **Votre Infrastructure IA Souveraine est Active !**
    
    🎯 **Capacités Disponibles :**
    - 💬 **Chat Intelligent** : Mistral Small (RTX 4090)
    - 🧠 **Analyse Cogito** : Magistral (RTX 3090)
    - 🖼️ **Vision IA** : Pixtral (RTX 4090)
    - 🎙️ **Audio Intelligent** : Whisper V3 (RTX 3090)
    - 🔊 **Synthèse Vocale** : TTS5 (RTX 3090)
    - 🎨 **Génération d'Images** : ComfyUI (RTX 4090)
    - 📄 **RAG Temporaire** : Recherche dans vos documents
    
    **Commencez par dire bonjour ou uploadez un fichier !**
    """
    
    # === CONFIGURATION CHAINLIT ===
    # Paramètres spécifiques à Chainlit
    CHAINLIT_HOST: str = os.getenv("CHAINLIT_HOST", "0.0.0.0")
    CHAINLIT_PORT: int = int(os.getenv("CHAINLIT_PORT", "8000"))
    
    # === CONFIGURATION DÉVELOPPEMENT ===
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Instance globale des paramètres
settings = Settings()

# === CONFIGURATION DES MODÈLES IA ===
# Informations détaillées sur chaque modèle de votre infrastructure
AI_MODELS = {
    "mistral": {
        "name": "Mistral Small",
        "endpoint": settings.MISTRAL_ENDPOINT,
        "model_name": "mistral",
        "description": "Modèle polyvalent rapide et efficace",
        "icon": "💬",
        "gpu": "RTX 4090",
        "max_tokens": 2000,
        "temperature_default": 0.7,
        "use_case": "Conversation générale, questions-réponses"
    },
    "magistral": {
        "name": "Magistral",
        "endpoint": settings.MAGISTRAL_ENDPOINT,
        "model_name": "magistral",
        "description": "Analyse approfondie et réflexion complexe",
        "icon": "🧠",
        "gpu": "RTX 3090",
        "max_tokens": 4000,
        "temperature_default": 0.3,
        "use_case": "Analyse cognitive, réflexion approfondie"
    },
    "pixtral": {
        "name": "Pixtral",
        "endpoint": settings.PIXTRAL_ENDPOINT,
        "model_name": "pixtral",
        "description": "Analyse et compréhension d'images",
        "icon": "🖼️",
        "gpu": "RTX 4090",
        "max_tokens": 1500,
        "temperature_default": 0.2,
        "use_case": "Analyse d'images, vision par ordinateur"
    }
}

# === CONFIGURATION SERVICES AUDIO ===
# Configuration des services audio de votre infrastructure
AUDIO_SERVICES = {
    "whisper": {
        "name": "Whisper Large V3",
        "endpoint": settings.WHISPER_ENDPOINT,
        "description": "Transcription audio haute précision",
        "icon": "🎙️",
        "gpu": "RTX 3090",
        "supported_formats": ["wav", "mp3", "m4a", "ogg", "flac"],
        "max_duration_seconds": 300,
        "languages": ["fr", "en", "es", "de", "it", "pt"]
    },
    "tts": {
        "name": "TTS5",
        "endpoint": settings.TTS_ENDPOINT,
        "description": "Synthèse vocale naturelle",
        "icon": "🔊",
        "gpu": "RTX 3090",
        "voices": ["male", "female", "neutral"],
        "max_text_length": 1000,
        "supported_formats": ["wav", "mp3"]
    }
}

# === CONFIGURATION GÉNÉRATION D'IMAGES ===
# Configuration ComfyUI
IMAGE_GENERATION = {
    "comfyui": {
        "name": "ComfyUI",
        "endpoint": settings.COMFYUI_ENDPOINT,
        "description": "Génération d'images avec Stable Diffusion",
        "icon": "🎨",
        "gpu": "RTX 4090",
        "max_resolution": "1024x1024",
        "supported_formats": ["png", "jpg"],
        "default_steps": 20,
        "default_cfg_scale": 7.0
    }
}

print("✅ Configuration HemersonAIBuild chargée avec succès!")
