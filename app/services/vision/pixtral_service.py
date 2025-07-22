# app/services/vision/pixtral_service.py
"""
Service d'analyse d'images avec Pixtral - HemersonAIBuild PlaygroundV1
Support multimodal avancé avec optimisations pour Chainlit 2.5.5
"""

import asyncio
import time
import base64
import hashlib
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import tempfile
from PIL import Image, ImageOps
import io
import aiofiles
import httpx
from loguru import logger

from services.rate_limiter.advanced_limiter import CacheManager, RequestType
from config.settings import settings

class PixtralVisionService:
    """
    Service d'analyse d'images avec Pixtral - Optimisé pour production
    
    Fonctionnalités avancées :
    - Support des tailles d'images variables (native resolution)
    - Analyse multimodale texte + image
    - Cache intelligent des analyses
    - Préprocessing d'images optimisé
    - Métadonnées enrichies
    - Gestion d'erreurs robuste
    """
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.pixtral_endpoint = settings.PIXTRAL_ENDPOINT
        
        # Client HTTP optimisé pour les images
        self.vision_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=15.0,
                read=240.0,  # Timeout plus long pour l'analyse d'images
                write=180.0,
                pool=360.0
            ),
            limits=httpx.Limits(
                max_keepalive_connections=10,
                max_connections=30,
                keepalive_expiry=120.0
            )
        )
        
        # Configuration de traitement d'images
        self.image_config = {
            "max_size_mb": 25,
            "supported_formats": ["JPEG", "PNG", "WEBP", "GIF", "BMP"],
            "max_resolution": (2048, 2048),  # Pixtral peut gérer des résolutions variables
            "quality_optimization": True,
            "auto_orient": True
        }
        
        # Métriques spécifiques vision
        self.vision_metrics = {
            "images_analyzed": 0,
            "successful_analyses": 0,
            "preprocessing_time_avg": 0.0,
            "analysis_time_avg": 0.0,
            "cache_hits": 0,
            "format_distributions": {},
            "resolution_stats": {"min": None, "max": None, "avg": 0}
        }
        
        logger.info("🖼️ PixtralVisionService initialisé avec support multimodal")
    
    async def analyze_image(self,
                           image_data: bytes,
                           filename: str,
                           prompt: str = "Analyse cette image en détail",
                           analysis_type: str = "comprehensive",
                           use_cache: bool = True,
                           **kwargs) -> Dict[str, Any]:
        """
        Analyse d'image avec Pixtral - Version complète optimisée
        
        Args:
            image_data: Données image en bytes
            filename: Nom du fichier original
            prompt: Prompt d'analyse personnalisé
            analysis_type: Type d'analyse (comprehensive, technical, creative, document)
            use_cache: Utiliser le cache intelligent
            **kwargs: Paramètres additionnels
            
        Returns:
            Analyse structurée avec métadonnées complètes
        """
        analysis_start = time.time()
        self.vision_metrics["images_analyzed"] += 1
        
        try:
            logger.info(f"🖼️ Début analyse image: {filename} (type: {analysis_type})")
            
            # Étape 1: Validation et préprocessing
            preprocessing_start = time.time()
            
            validation_result = self._validate_image(image_data, filename)
            if not validation_result["valid"]:
                return {
                    "success": False,
                    "error": validation_result["error"],
                    "analysis": "",
                    "metadata": {"validation_failed": True}
                }
            
            # Préprocessing intelligent de l'image
            processed_image_data, image_metadata = await self._preprocess_image(
                image_data, 
                filename,
                optimization_level=kwargs.get("optimization_level", "balanced")
            )
            
            preprocessing_time = time.time() - preprocessing_start
            logger.info(f"✅ Préprocessing terminé en {preprocessing_time:.2f}s")
            
            # Étape 2: Vérification du cache
            if use_cache:
                cache_key = self._generate_image_cache_key(
                    processed_image_data, 
                    prompt, 
                    analysis_type,
                    kwargs
                )
                
                cached_result = await self.cache_manager.get_cached_response(cache_key)
                if cached_result:
                    self.vision_metrics["cache_hits"] += 1
                    logger.info(f"💾 Analyse depuis cache: {filename}")
                    # Ajouter les métadonnées de cette session
                    cached_result["metadata"]["cached"] = True
                    cached_result["metadata"]["cache_timestamp"] = time.time()
                    return cached_result
            
            # Étape 3: Analyse avec Pixtral
            analysis_start_time = time.time()
            
            # Construire le prompt spécialisé selon le type d'analyse
            specialized_prompt = self._build_specialized_prompt(
                prompt, 
                analysis_type, 
                image_metadata
            )
            
            # Encoder l'image pour Pixtral
            image_b64 = base64.b64encode(processed_image_data).decode('utf-8')
            
            # Préparer la requête Pixtral
            pixtral_payload = {
                "model": "pixtral",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": specialized_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": kwargs.get("max_tokens", 2000),
                "temperature": kwargs.get("temperature", 0.2),
                "stream": False
            }
            
            # Faire l'analyse avec retry
            result = await self._make_vision_request_with_retry(
                f"{self.pixtral_endpoint}/v1/chat/completions",
                pixtral_payload,
                max_retries=2,
                operation=f"pixtral_{analysis_type}"
            )
            
            analysis_time = time.time() - analysis_start_time
            
            if result["success"]:
                analysis_content = result["data"]["choices"][0]["message"]["content"]
                
                # Construire la réponse complète
                complete_result = {
                    "success": True,
                    "analysis": analysis_content,
                    "model": "Pixtral 12B",
                    "gpu": "RTX 4090 (192.168.1.212)",
                    "analysis_type": analysis_type.title(),
                    "original_prompt": prompt,
                    "specialized_prompt": specialized_prompt,
                    "processing_times": {
                        "preprocessing": preprocessing_time,
                        "analysis": analysis_time,
                        "total": time.time() - analysis_start
                    },
                    "metadata": {
                        "filename": filename,
                        "image_metadata": image_metadata,
                        "tokens_used": result["data"]["usage"]["total_tokens"],
                        "cached": False,
                        "analysis_quality": self._assess_analysis_quality(analysis_content),
                        "timestamp": time.time()
                    }
                }
                
                # Mettre en cache avec TTL adapté
                if use_cache:
                    cache_ttl = 7200  # 2 heures par défaut
                    if analysis_type == "document":
                        cache_ttl = 14400  # 4 heures pour documents (plus stable)
                    
                    await self.cache_manager.cache_response(
                        cache_key,
                        complete_result,
                        ttl=cache_ttl
                    )
                
                # Mettre à jour les métriques
                self._update_vision_metrics(True, preprocessing_time, analysis_time, image_metadata)
                self.vision_metrics["successful_analyses"] += 1
                
                logger.info(f"✅ Analyse Pixtral réussie: {filename} ({analysis_time:.2f}s)")
                
                return complete_result
            else:
                self._update_vision_metrics(False, preprocessing_time, 0, image_metadata)
                return {
                    "success": False,
                    "error": result["error"],
                    "analysis": "Erreur lors de l'analyse d'image avec Pixtral.",
                    "model": "Pixtral 12B",
                    "processing_times": {
                        "preprocessing": preprocessing_time,
                        "total": time.time() - analysis_start
                    }
                }
                
        except Exception as e:
            logger.error(f"❌ Erreur analyse Pixtral: {e}")
            return {
                "success": False,
                "error": str(e),
                "analysis": f"Erreur lors de l'analyse d'image: {str(e)}",
                "model": "Pixtral 12B",
                "processing_times": {
                    "total": time.time() - analysis_start
                }
            }
    
    def _validate_image(self, image_data: bytes, filename: str) -> Dict[str, Any]:
        """Validation complète d'une image"""
        try:
            # Vérifier la taille
            if len(image_data) > self.image_config["max_size_mb"] * 1024 * 1024:
                return {
                    "valid": False,
                    "error": f"Image trop volumineuse: {len(image_data) / (1024*1024):.1f}MB (max: {self.image_config['max_size_mb']}MB)"
                }
            
            # Vérifier si c'est une image valide
            try:
                with Image.open(io.BytesIO(image_data)) as img:
                    # Vérifier le format
                    if img.format not in self.image_config["supported_formats"]:
                        return {
                            "valid": False,
                            "error": f"Format non supporté: {img.format}. Formats acceptés: {', '.join(self.image_config['supported_formats'])}"
                        }
                    
                    # Vérifier les dimensions
                    width, height = img.size
                    max_w, max_h = self.image_config["max_resolution"]
                    
                    if width > max_w or height > max_h:
                        logger.warning(f"Image grande résolution détectée: {width}x{height}")
                    
                    # Vérifier l'intégrité
                    img.verify()
                    
                    return {
                        "valid": True,
                        "format": img.format,
                        "size": (width, height),
                        "mode": img.mode
                    }
                    
            except Exception as img_error:
                return {
                    "valid": False,
                    "error": f"Image corrompue ou invalide: {str(img_error)}"
                }
                
        except Exception as e:
            return {
                "valid": False,
                "error": f"Erreur de validation: {str(e)}"
            }
    
    async def _preprocess_image(self, 
                               image_data: bytes, 
                               filename: str,
                               optimization_level: str = "balanced") -> Tuple[bytes, Dict[str, Any]]:
        """
        Préprocessing intelligent d'image pour Pixtral
        
        Args:
            image_data: Données image originales
            filename: Nom du fichier
            optimization_level: Niveau d'optimisation (fast, balanced, quality)
            
        Returns:
            (processed_image_data, metadata)
        """
        try:
            with Image.open(io.BytesIO(image_data)) as img:
                # Métadonnées originales
                original_metadata = {
                    "original_format": img.format,
                    "original_size": img.size,
                    "original_mode": img.mode,
                    "original_bytes": len(image_data)
                }
                
                # Conversion RGBA -> RGB si nécessaire (pour JPEG)
                if img.mode in ('RGBA', 'LA', 'P'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                    img = background
                
                # Auto-orientation (EXIF)
                if self.image_config["auto_orient"]:
                    img = ImageOps.exif_transpose(img)
                
                # Optimisation selon le niveau
                processed_img = self._apply_optimization(img, optimization_level)
                
                # Sauvegarder en mémoire
                output_buffer = io.BytesIO()
                
                # Format et qualité optimaux pour Pixtral
                if optimization_level == "quality":
                    processed_img.save(output_buffer, format='JPEG', quality=95, optimize=True)
                elif optimization_level == "fast":
                    processed_img.save(output_buffer, format='JPEG', quality=75, optimize=True)
                else:  # balanced
                    processed_img.save(output_buffer, format='JPEG', quality=85, optimize=True)
                
                processed_data = output_buffer.getvalue()
                
                # Métadonnées complètes
                metadata = {
                    **original_metadata,
                    "processed_format": "JPEG",
                    "processed_size": processed_img.size,
                    "processed_bytes": len(processed_data),
                    "compression_ratio": len(image_data) / len(processed_data),
                    "optimization_level": optimization_level,
                    "aspect_ratio": processed_img.size[0] / processed_img.size[1],
                    "pixel_count": processed_img.size[0] * processed_img.size[1]
                }
                
                return processed_data, metadata
                
        except Exception as e:
            logger.error(f"❌ Erreur preprocessing: {e}")
            # En cas d'erreur, retourner l'image originale
            return image_data, {
                "preprocessing_failed": True,
                "error": str(e),
                "fallback_to_original": True
            }
    
    def _apply_optimization(self, img: Image.Image, level: str) -> Image.Image:
        """Applique les optimisations selon le niveau"""
        try:
            width, height = img.size
            max_w, max_h = self.image_config["max_resolution"]
            
            if level == "fast":
                # Redimensionnement agressif si nécessaire
                if width > max_w or height > max_h:
                    img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
                
            elif level == "quality":
                # Préserver la qualité autant que possible
                if width > max_w * 1.5 or height > max_h * 1.5:
                    img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
                
            else:  # balanced
                # Optimisation équilibrée
                if width > max_w or height > max_h:
                    img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
            
            return img
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur optimisation image: {e}")
            return img
    
    def _build_specialized_prompt(self, 
                                 base_prompt: str, 
                                 analysis_type: str,
                                 image_metadata: Dict[str, Any]) -> str:
        """Construit un prompt spécialisé selon le type d'analyse"""
        
        # Contexte sur l'image
        img_context = f"Image: {image_metadata.get('processed_size', 'Résolution inconnue')}, "
        img_context += f"format: {image_metadata.get('original_format', 'inconnu')}"
        
        prompts_templates = {
            "comprehensive": f"""Tu es Pixtral, expert en analyse d'images multimodale.

{img_context}

Effectue une analyse COMPLÈTE et STRUCTURÉE de cette image selon ce prompt : "{base_prompt}"

Structure ta réponse EXACTEMENT ainsi :

🔍 **DESCRIPTION GÉNÉRALE**
- Sujet principal et éléments secondaires
- Composition, cadrage et perspective
- Style artistique ou photographique

🎨 **ANALYSE VISUELLE DÉTAILLÉE**
- Couleurs dominantes et palette chromatique
- Textures, matières et surfaces
- Éclairage, ombres et atmosphère
- Qualité technique de l'image

📊 **ANALYSE TECHNIQUE**
- Technique de création (photo, dessin, rendu, etc.)
- Qualité et résolution observée
- Éléments techniques remarquables
- Post-traitement apparent

🧠 **INTERPRÉTATION CONTEXTUELLE**
- Contexte historique, culturel ou social
- Message ou signification possible
- Références artistiques ou culturelles
- Émotion ou sentiment transmis

💡 **OBSERVATIONS SPÉCIALISÉES**
- Détails fins ou éléments cachés
- Aspects inhabituels ou remarquables
- Analyse de la composition avancée
- Suggestions d'amélioration si pertinentes

Sois précis, détaillé et analytique. Utilise tes capacités de vision avancées.""",

            "technical": f"""Tu es Pixtral en mode EXPERT TECHNIQUE.

{img_context}

Analyse technique approfondie : "{base_prompt}"

FOCUS sur :
- Aspects techniques de l'image
- Qualité, résolution, compression
- Techniques de création utilisées
- Défauts ou problèmes techniques
- Optimisations possibles
- Métadonnées visuelles détectables

Sois technique et précis.""",

            "creative": f"""Tu es Pixtral en mode CRÉATIF.

{img_context}

Analyse créative : "{base_prompt}"

EXPLORE :
- Aspects artistiques et esthétiques
- Créativité et originalité
- Inspiration et influences
- Potentiel créatif
- Suggestions d'amélioration artistique
- Variations possibles

Sois créatif et inspirant.""",

            "document": f"""Tu es Pixtral, expert en analyse de DOCUMENTS.

{img_context}

Analyse de document : "{base_prompt}"

STRUCTURE pour documents :
- Type de document identifié
- Texte visible (transcription si possible)
- Structure et mise en page
- Qualité de lisibilité
- Informations extraites
- Conseils d'optimisation pour OCR

Sois méthodique et précis pour l'extraction d'informations."""
        }
        
        return prompts_templates.get(analysis_type, prompts_templates["comprehensive"])
    
    def _generate_image_cache_key(self, 
                                 image_data: bytes, 
                                 prompt: str,
                                 analysis_type: str, 
                                 params: Dict) -> str:
        """Génère une clé de cache unique pour l'image"""
        # Hash de l'image
        image_hash = hashlib.md5(image_data).hexdigest()[:16]
        
        # Hash du prompt et paramètres
        prompt_data = f"{prompt}_{analysis_type}_{sorted(params.items())}"
        prompt_hash = hashlib.md5(prompt_data.encode()).hexdigest()[:16]
        
        return f"pixtral_vision:{image_hash}:{prompt_hash}"
    
    def _assess_analysis_quality(self, analysis_content: str) -> Dict[str, Any]:
        """Évalue la qualité de l'analyse produite"""
        try:
            # Métriques de qualité
            content_length = len(analysis_content)
            word_count = len(analysis_content.split())
            paragraph_count = analysis_content.count('\n\n') + 1
            structured_sections = analysis_content.count('**')
            
            # Score de qualité basé sur plusieurs facteurs
            length_score = min(content_length / 1000, 1.0)  # Optimal vers 1000 chars
            structure_score = min(structured_sections / 10, 1.0)  # Sections structurées
            detail_score = min(word_count / 200, 1.0)  # Détail suffisant
            
            overall_quality = (length_score * 0.3 + structure_score * 0.4 + detail_score * 0.3)
            
            # Classification qualitative
            if overall_quality >= 0.8:
                quality_label = "excellent"
            elif overall_quality >= 0.6:
                quality_label = "good"
            elif overall_quality >= 0.4:
                quality_label = "fair"
            else:
                quality_label = "poor"
            
            return {
                "overall_score": overall_quality,
                "quality_label": quality_label,
                "content_length": content_length,
                "word_count": word_count,
                "structure_score": structure_score,
                "detail_level": detail_score
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur évaluation qualité: {e}")
            return {"overall_score": 0.5, "quality_label": "unknown"}
    
    async def _make_vision_request_with_retry(self, 
                                             endpoint: str,
                                             payload: Dict[str, Any],
                                             max_retries: int = 3,
                                             operation: str = "vision_request") -> Dict[str, Any]:
        """Requête HTTP spécialisée pour la vision avec retry"""
        
        for attempt in range(max_retries):
            try:
                response = await self.vision_client.post(
                    endpoint,
                    json=payload,
                    timeout=180  # 3 minutes pour analyse d'images
                )
                
                if response.status_code == 200:
                    return {"success": True, "data": response.json()}
                else:
                    logger.warning(f"⚠️ HTTP {response.status_code} pour {operation} (tentative {attempt + 1})")
                    
            except httpx.TimeoutException:
                logger.warning(f"⏰ Timeout {operation} (tentative {attempt + 1})")
            except Exception as e:
                logger.error(f"❌ Erreur {operation}: {e}")
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"⏳ Attente {wait_time}s avant retry {operation}...")
                await asyncio.sleep(wait_time)
        
        return {
            "success": False,
            "error": f"Échec {operation} après {max_retries} tentatives"
        }
    
    def _update_vision_metrics(self, 
                              success: bool, 
                              preprocessing_time: float,
                              analysis_time: float, 
                              image_metadata: Dict[str, Any]):
        """Met à jour les métriques de vision"""
        
        # Temps de préprocessing
        current_prep_avg = self.vision_metrics["preprocessing_time_avg"]
        total_images = self.vision_metrics["images_analyzed"]
        self.vision_metrics["preprocessing_time_avg"] = (
            (current_prep_avg * (total_images - 1) + preprocessing_time) / total_images
        )
        
        if success and analysis_time > 0:
            # Temps d'analyse
            current_analysis_avg = self.vision_metrics["analysis_time_avg"]
            successful_count = self.vision_metrics["successful_analyses"] + 1
            self.vision_metrics["analysis_time_avg"] = (
                (current_analysis_avg * (successful_count - 1) + analysis_time) / successful_count
            )
        
        # Distribution des formats
        original_format = image_metadata.get("original_format", "unknown")
        if original_format in self.vision_metrics["format_distributions"]:
            self.vision_metrics["format_distributions"][original_format] += 1
        else:
            self.vision_metrics["format_distributions"][original_format] = 1
        
        # Statistiques de résolution
        pixel_count = image_metadata.get("pixel_count", 0)
        if pixel_count > 0:
            if self.vision_metrics["resolution_stats"]["min"] is None:
                self.vision_metrics["resolution_stats"]["min"] = pixel_count
                self.vision_metrics["resolution_stats"]["max"] = pixel_count
            else:
                self.vision_metrics["resolution_stats"]["min"] = min(
                    self.vision_metrics["resolution_stats"]["min"], pixel_count
                )
                self.vision_metrics["resolution_stats"]["max"] = max(
                    self.vision_metrics["resolution_stats"]["max"], pixel_count
                )
            
            # Moyenne mobile
            current_avg = self.vision_metrics["resolution_stats"]["avg"]
            self.vision_metrics["resolution_stats"]["avg"] = (
                (current_avg * (total_images - 1) + pixel_count) / total_images
            )
    
    async def get_vision_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques de vision détaillées"""
        success_rate = 0
        if self.vision_metrics["images_analyzed"] > 0:
            success_rate = (
                self.vision_metrics["successful_analyses"] / 
                self.vision_metrics["images_analyzed"]
            ) * 100
        
        cache_hit_rate = 0
        if self.vision_metrics["images_analyzed"] > 0:
            cache_hit_rate = (
                self.vision_metrics["cache_hits"] / 
                self.vision_metrics["images_analyzed"]
            ) * 100
        
        return {
            **self.vision_metrics,
            "success_rate_percent": success_rate,
            "cache_hit_rate_percent": cache_hit_rate,
            "avg_image_resolution": self.vision_metrics["resolution_stats"]["avg"],
            "most_common_format": max(
                self.vision_metrics["format_distributions"], 
                key=self.vision_metrics["format_distributions"].get
            ) if self.vision_metrics["format_distributions"] else "unknown"
        }
    
    async def close(self):
        """Fermeture propre du service"""
        await self.vision_client.aclose()
        logger.info("🔌 PixtralVisionService fermé proprement")

# Instance globale
pixtral_service = PixtralVisionService(cache_manager)
