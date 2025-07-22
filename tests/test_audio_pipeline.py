# tests/test_audio_pipeline.py
"""
Tests du pipeline audio complet
"""

import pytest
import asyncio
from pathlib import Path
import time

from app.services.audio.audio_pipeline import AudioPipelineService
from app.services.rate_limiter.advanced_limiter import CacheManager
from app.services.ai.unified_ai_service import UnifiedAIService

class TestAudioPipeline:
    """Suite de tests pour le pipeline audio"""
    
    @pytest.fixture
    async def audio_pipeline(self):
        """Fixture du pipeline audio"""
        # Mock des services pour tests
        cache_manager = CacheManager(None)  # Mode test
        ai_service = UnifiedAIService(cache_manager)
        
        pipeline = AudioPipelineService(cache_manager, ai_service)
        yield pipeline
        await pipeline.close()
    
    @pytest.fixture
    def sample_audio_wav(self):
        """Génère un échantillon WAV minimal pour tests"""
        # Header WAV minimal (44 bytes) + données audio factices
        wav_header = b'RIFF' + (1000).to_bytes(4, 'little') + b'WAVE'
        wav_header += b'fmt ' + (16).to_bytes(4, 'little') + (1).to_bytes(2, 'little')
        wav_header += (1).to_bytes(2, 'little') + (44100).to_bytes(4, 'little')
        wav_header += (88200).to_bytes(4, 'little') + (2).to_bytes(2, 'little')
        wav_header += (16).to_bytes(2, 'little') + b'data' + (964).to_bytes(4, 'little')
        
        # Ajouter des données audio factices
        audio_data = b'\x00' * 964
        
        return wav_header + audio_data
    
    @pytest.mark.asyncio
    async def test_audio_format_validation(self, audio_pipeline, sample_audio_wav):
        """Test de validation des formats audio"""
        
        # Test format WAV valide
        wav_analysis = audio_pipeline._analyze_audio_format(sample_audio_wav, "test.wav")
        assert wav_analysis["is_valid"] == True
        assert wav_analysis["format"] == "wav"
        assert wav_analysis["mime_type"] == "audio/wav"
        
        # Test format invalide
        invalid_data = b"This is not audio data"
        invalid_analysis = audio_pipeline._analyze_audio_format(invalid_data, "test.txt")
        assert invalid_analysis["is_valid"] == False
    
    @pytest.mark.asyncio
    async def test_text_preparation_for_synthesis(self, audio_pipeline):
        """Test de préparation de texte pour TTS"""
        
        # Texte normal
        normal_text = "Bonjour, comment allez-vous aujourd'hui ?"
        cleaned = audio_pipeline._prepare_text_for_synthesis(normal_text)
        assert cleaned == normal_text
        
        # Texte avec caractères spéciaux
        special_text = "Résultat : A > B & C < D"
        cleaned = audio_pipeline._prepare_text_for_synthesis(special_text)
        assert "&" not in cleaned
        assert "<" not in cleaned
        assert ">" not in cleaned
        
        # Texte trop long
        long_text = "Un texte très long. " * 100  # > 1000 chars
        cleaned = audio_pipeline._prepare_text_for_synthesis(long_text)
        assert len(cleaned) <= 1000
        assert cleaned.endswith('.') or cleaned.endswith('...')
    
    @pytest.mark.asyncio
    async def test_transcription_quality_analysis(self, audio_pipeline):
        """Test d'analyse de qualité de transcription"""
        
        # Réponse Whisper simulée avec bonne qualité
        good_response = {
            "text": "Bonjour, j'espère que vous allez bien aujourd'hui.",
            "segments": [
                {"avg_logprob": -0.2, "text": "Bonjour,"},
                {"avg_logprob": -0.1, "text": " j'espère que vous allez bien aujourd'hui."}
            ]
        }
        
        quality = audio_pipeline._analyze_transcription_quality(good_response)
        assert quality["quality"] in ["good", "excellent"]
        assert quality["overall_confidence"] > 0.6
        
        # Réponse avec mauvaise qualité
        poor_response = {
            "text": "euh hmm",
            "segments": [
                {"avg_logprob": -2.5, "text": "euh hmm"}
            ]
        }
        
        quality = audio_pipeline._analyze_transcription_quality(poor_response)
        assert quality["overall_confidence"] < 0.5
    
    @pytest.mark.asyncio  
    async def test_pipeline_efficiency_calculation(self, audio_pipeline):
        """Test du calcul d'efficacité du pipeline"""
        
        # Pipeline rapide avec fichier moyen
        efficiency = audio_pipeline._calculate_pipeline_efficiency(
            total_time=5.0,
            input_size=2 * 1024 * 1024  # 2MB
        )
        assert efficiency in ["good", "excellent"]
        
        # Pipeline lent
        efficiency = audio_pipeline._calculate_pipeline_efficiency(
            total_time=30.0,
            input_size=1 * 1024 * 1024  # 1MB
        )
        assert efficiency in ["slow", "fair"]
    
    @pytest.mark.asyncio
    async def test_metrics_tracking(self, audio_pipeline):
        """Test du tracking des métriques"""
        
        initial_metrics = await audio_pipeline.get_audio_metrics()
        
        # Les métriques initiales doivent être à zéro
        assert initial_metrics["transcriptions_total"] == 0
        assert initial_metrics["synthesis_total"] == 0
        assert initial_metrics["pipeline_complete_total"] == 0
        
        # Simuler quelques opérations pour tester l'incrémentation
        audio_pipeline.audio_metrics["transcriptions_total"] = 5
        audio_pipeline.audio_metrics["transcriptions_successful"] = 4
        
        updated_metrics = await audio_pipeline.get_audio_metrics()
        assert updated_metrics["success_rates"]["transcription"] == 80.0

# Tests d'intégration avec services réels (à exécuter manuellement)
@pytest.mark.integration
class TestAudioIntegration:
    """Tests d'intégration avec les services réels"""
    
    @pytest.mark.asyncio
    async def test_real_transcription(self):
        """Test de transcription réelle (nécessite Whisper actif)"""
        # Ce test nécessite un service Whisper actif
        # À exécuter manuellement lors des tests d'intégration
        pass
    
    @pytest.mark.asyncio
    async def test_real_synthesis(self):
        """Test de synthèse réelle (nécessite TTS actif)"""  
        # Ce test nécessite un service TTS actif
        # À exécuter manuellement lors des tests d'intégration
        pass
    
    @pytest.mark.asyncio
    async def test_complete_pipeline_integration(self):
        """Test complet du pipeline (nécessite tous les services)"""
        # Test d'intégration complète
        # À exécuter lors des tests de validation finale
        pass
