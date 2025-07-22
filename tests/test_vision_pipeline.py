# tests/test_vision_pipeline.py
"""
Tests du pipeline de vision avec Pixtral
"""

import pytest
import asyncio
from pathlib import Path
import time
from PIL import Image
import io

from app.services.vision.pixtral_service import PixtralVisionService
from app.services.rate_limiter.advanced_limiter import CacheManager

class TestVisionPipeline:
    """Suite de tests pour le pipeline de vision"""
    
    @pytest.fixture
    async def vision_service(self):
        """Fixture du service vision"""
        cache_manager = CacheManager(None)  # Mode test
        service = PixtralVisionService(cache_manager)
        yield service
        await service.close()
    
    @pytest.fixture
    def sample_image_jpeg(self):
        """Génère une image JPEG de test"""
        # Créer une image simple
        img = Image.new('RGB', (800, 600), color='red')
        
        # Ajouter du contenu
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        draw.rectangle([100, 100, 700, 500], fill='blue', outline='white', width=5)
        draw.text((200, 250), "TEST IMAGE", fill='white')
        
        # Convertir en bytes
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        return buffer.getvalue()
    
    def test_image_validation(self, vision_service, sample_image_jpeg):
        """Test de validation d'images"""
        
        # Test image valide
        validation = vision_service._validate_image(sample_image_jpeg, "test.jpg")
        assert validation["valid"] == True
        assert validation["format"] == "JPEG"
        assert validation["size"] == (800, 600)
        
        # Test données invalides
        invalid_validation = vision_service._validate_image(b"invalid data", "test.txt")
        assert invalid_validation["valid"] == False
        assert "corrompue" in invalid_validation["error"]
    
    @pytest.mark.asyncio
    async def test_image_preprocessing(self, vision_service, sample_image_jpeg):
        """Test du préprocessing d'images"""
        
        processed_data, metadata = await vision_service._preprocess_image(
            sample_image_jpeg,
            "test.jpg",
            "balanced"
        )
        
        # Vérifier les métadonnées
        assert metadata["original_format"] == "JPEG"
        assert metadata["original_size"] == (800, 600)
        assert metadata["processed_format"] == "JPEG"
        assert metadata["optimization_level"] == "balanced"
        assert "compression_ratio" in metadata
        
        # Vérifier que l'image traitée est valide
        with Image.open(io.BytesIO(processed_data)) as img:
            assert img.format == "JPEG"
            assert img.size[0] <= 800  # Peut être redimensionnée
            assert img.size[1] <= 600
    
    def test_cache_key_generation(self, vision_service, sample_image_jpeg):
        """Test de génération des clés de cache"""
        
        prompt = "Analyse cette image"
        analysis_type = "comprehensive"
        params = {"temperature": 0.2}
        
        key1 = vision_service._generate_image_cache_key(
            sample_image_jpeg, prompt, analysis_type, params
        )
        
        key2 = vision_service._generate_image_cache_key(
            sample_image_jpeg, prompt, analysis_type, params
        )
        
        # Les mêmes paramètres donnent la même clé
        assert key1 == key2
        assert key1.startswith("pixtral_vision:")
        
        # Des paramètres différents donnent des clés différentes
        key3 = vision_service._generate_image_cache_key(
            sample_image_jpeg, "Autre prompt", analysis_type, params
        )
        assert key1 != key3
    
    def test_prompt_specialization(self, vision_service):
        """Test de spécialisation des prompts"""
        
        base_prompt = "Analyse cette image"
        image_metadata = {"processed_size": (800, 600), "original_format": "JPEG"}
        
        # Test différents types d'analyse
        comprehensive_prompt = vision_service._build_specialized_prompt(
            base_prompt, "comprehensive", image_metadata
        )
        assert "COMPLÈTE et STRUCTURÉE" in comprehensive_prompt
        assert "🔍 **DESCRIPTION GÉNÉRALE**" in comprehensive_prompt
        
        technical_prompt = vision_service._build_specialized_prompt(
            base_prompt, "technical", image_metadata
        )
        assert "EXPERT TECHNIQUE" in technical_prompt
        assert "Aspects techniques" in technical_prompt
        
        creative_prompt = vision_service._build_specialized_prompt(
            base_prompt, "creative", image_metadata
        )
        assert "CRÉATIF" in creative_prompt
        assert "artistiques et esthétiques" in creative_prompt
    
    def test_analysis_quality_assessment(self, vision_service):
        """Test d'évaluation de la qualité d'analyse"""
        
        # Analyse de bonne qualité
        good_analysis = """
        🔍 **DESCRIPTION GÉNÉRALE**
        Cette image montre un paysage magnifique avec plusieurs éléments remarquables.
        La composition est équilibrée et les couleurs sont harmonieuses.
        
        🎨 **ANALYSE VISUELLE DÉTAILLÉE**  
        Les couleurs dominantes sont le bleu et le vert, créant une atmosphère paisible.
        Les textures sont variées et bien définies.
        """
        
        quality = vision_service._assess_analysis_quality(good_analysis)
        assert quality["quality_label"] in ["good", "excellent"]
        assert quality["overall_score"] > 0.5
        assert quality["word_count"] > 30
        
        # Analyse de faible qualité
        poor_analysis = "Image."
        
        quality = vision_service._assess_analysis_quality(poor_analysis)
        assert quality["quality_label"] == "poor"
        assert quality["overall_score"] < 0.3
    
    @pytest.mark.asyncio
    async def test_metrics_tracking(self, vision_service):
        """Test du tracking des métriques"""
        
        initial_metrics = await vision_service.get_vision_metrics()
        
        # Les métriques initiales doivent être à zéro
        assert initial_metrics["images_analyzed"] == 0
        assert initial_metrics["successful_analyses"] == 0
        
        # Simuler quelques analyses pour tester l'incrémentation
        vision_service.vision_metrics["images_analyzed"] = 10
        vision_service.vision_metrics["successful_analyses"] = 8
        vision_service.vision_metrics["cache_hits"] = 3
        
        updated_metrics = await vision_service.get_vision_metrics()
        assert updated_metrics["success_rate_percent"] == 80.0
        assert updated_metrics["cache_hit_rate_percent"] == 30.0
    
    def test_optimization_levels(self, vision_service):
        """Test des différents niveaux d'optimisation"""
        
        # Créer une grande image de test
        large_img = Image.new('RGB', (3000, 2000), color='blue')
        
        # Test optimisation "fast"
        fast_optimized = vision_service._apply_optimization(large_img, "fast")
        assert fast_optimized.size[0] <= 2048  # Devrait être redimensionnée
        assert fast_optimized.size[1] <= 2048
        
        # Test optimisation "quality"
        quality_optimized = vision_service._apply_optimization(large_img, "quality") 
        # Devrait préserver plus de qualité
        
        # Test optimisation "balanced"
        balanced_optimized = vision_service._apply_optimization(large_img, "balanced")
        assert balanced_optimized.size[0] <= 2048
        assert balanced_optimized.size[1] <= 2048

# Tests d'intégration avec services réels (à exécuter manuellement)
@pytest.mark.integration
class TestVisionIntegration:
    """Tests d'intégration avec Pixtral réel"""
    
    @pytest.mark.asyncio
    async def test_real_pixtral_analysis(self):
        """Test d'analyse réelle (nécessite Pixtral actif)"""
        # Ce test nécessite un service Pixtral actif
        # À exécuter manuellement lors des tests d'intégration
        pass
    
    @pytest.mark.asyncio
    async def test_multiple_image_formats(self):
        """Test avec différents formats d'images"""
        # Test avec PNG, WEBP, etc.
        # À exécuter lors des tests de validation finale
        pass
    
    @pytest.mark.asyncio
    async def test_large_image_handling(self):
        """Test avec des images de grande taille"""
        # Test de performance avec images haute résolution
        # À exécuter lors des tests de stress
        pass
