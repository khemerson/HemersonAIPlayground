# app/core/settings.py  (Puis contenu complet (reprendre le contenu détaillé de la Phase 9A))
"""Configuration centralisée PlaygroundV1"""

from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    """Configuration principale avec validation"""
    
    # === INFORMATIONS PROJET ===
    PROJECT_NAME: str = "HemersonAIBuild PlaygroundV1"
    
    # === ENDPOINTS IA ===
    MISTRAL_ENDPOINT: str
    MAGISTRAL_ENDPOINT: str
    # ... autres endpoints
    
    # === REDIS ===
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    
    class Config:
        env_file = ".env"

settings = Settings()
