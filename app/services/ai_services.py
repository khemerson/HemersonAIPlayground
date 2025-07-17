# app/services/ai_services.py
"""
Services IA optimisés pour HemersonAIBuild
Connexions directes aux modèles avec gestion d'erreurs avancée
"""

import httpx
import json
import base64
import asyncio
import time
from typing import Dict, Any, Optional, List
from PIL import Image
import io
from loguru import logger
from ..utils.config import settings, AI_MODELS, AUDIO_SERVICES, IMAGE_GENERATION

class OptimizedAIServices:
    """
    Services IA optimisés pour votre infrastructure HemersonAIBuild
    
    Cette classe gère toutes les connexions directes vers vos modèles IA
    avec gestion d'erreurs avancée et optimisations de performance
    """
    
    def __init__(self):
        """Initialise les services IA avec client HTTP optimisé"""
        
        # Client HTTP asynchrone optimisé
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0),  # Timeout global
            limits=httpx.Limits(
                max_keepalive_connections=10,
                max_connections=20
            )
        )
        
        # Cache pour les réponses courtes
        self.response_cache = {}
        
        logger.info("✅ Services IA optimisés initialisés")
    
    async def __aenter__(self):
        """Support du context manager"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Fermeture propre du client HTTP"""
        await self.client.aclose()
    
    # === SERVICES DE CHAT ===
    
    async def chat_mistral(self, message: str, **kwargs) -> Dict[str, Any]:
        """
        Chat avec Mistral Small sur RTX 4090
        
        Args:
            message: Message de l'utilisateur
            **kwargs: Paramètres optionnels (temperature, max_tokens, etc.)
            
        Returns:
            Dict avec la réponse et les métadonnées
        """
        try:
            # Paramètres avec valeurs par défaut
            temperature = kwargs.get('temperature', AI_MODELS['mistral']['temperature_default'])
            max_tokens = kwargs.get('max_tokens', AI_MODELS['mistral']['max_tokens'])
            
            # Préparation de la requête
            payload = {
                "model": AI_MODELS['mistral']['model_name'],
                "prompt": message,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                },
                "stream": False
            }
            
            # Appel asynchrone à Mistral
            start_time = time.time()
            response = await self.client.post(
                f"{settings.MISTRAL_ENDPOINT}/api/generate",
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                processing_time = time.time() - start_time
                
                return {
                    "success": True,
                    "response": result.get("response", ""),
                    "model": AI_MODELS['mistral']['name'],
                    "gpu": AI_MODELS['mistral']['gpu'],
                    "tokens_used": result.get("prompt_eval_count", 0),
                    "processing_time": processing_time,
                    "temperature": temperature,
                    "endpoint": settings.MISTRAL_ENDPOINT
                }
            else:
                logger.error(f"Erreur Mistral HTTP {response.status_code}")
                return {
                    "success": False,
                    "error": f"Erreur HTTP {response.status_code}",
                    "response": "Désolé, Mistral n'est pas disponible actuellement."
                }
                
        except Exception as e:
            logger.error(f"Erreur Mistral: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": f"Erreur de connexion à Mistral: {str(e)}"
            }
    
    async def chat_magistral(self, message: str, **kwargs) -> Dict[str, Any]:
        """
        Chat avec Magistral sur RTX 3090 (analyses approfondies)
        
        Args:
            message: Message à analyser en profondeur
            **kwargs: Paramètres optionnels
            
        Returns:
            Dict avec l'analyse Cogito et les métadonnées
        """
        try:
            # Prompt spécialisé pour l'analyse approfondie
            analysis_prompt = f"""
            Tu es Magistral, un assistant IA spécialisé dans l'analyse approfondie.
            
            Effectue une analyse cognitive complète de cette requête :
            
            "{message}"
            
            Structure ton analyse ainsi :
            
            🎯 **Analyse Contextuelle**
            - Contexte et enjeux identifiés
            - Dimensions sous-jacentes
            
            🔍 **Perspectives Multiples**
            - Angles d'approche différents
            - Points de vue alternatifs
            
            💡 **Insights et Nuances**
            - Subtilités importantes
            - Implications profondes
            
            🚀 **Recommandations**
            - Actions concrètes
            - Pistes d'approfondissement
            
            Sois précis, structuré et apporte une valeur analytique réelle.
            """
            
            # Paramètres optimisés pour l'analyse
            payload = {
                "model": AI_MODELS['magistral']['model_name'],
                "prompt": analysis_prompt,
                "options": {
                    "temperature": AI_MODELS['magistral']['temperature_default'],
                    "num_predict": AI_MODELS['magistral']['max_tokens']
                },
                "stream": False
            }
            
            # Appel asynchrone à Magistral
            start_time = time.time()
            response = await self.client.post(
                f"{settings.MAGISTRAL_ENDPOINT}/api/generate",
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                processing_time = time.time() - start_time
                
                return {
                    "success": True,
                    "response": result.get("response", ""),
                    "model": AI_MODELS['magistral']['name'],
                    "gpu": AI_MODELS['magistral']['gpu'],
                    "analysis_type": "Cogito",
                    "tokens_used": result.get("prompt_eval_count", 0),
                    "processing_time": processing_time,
                    "original_message": message,
                    "endpoint": settings.MAGISTRAL_ENDPOINT
                }
            else:
                logger.error(f"Erreur Magistral HTTP {response.status_code}")
                return {
                    "success": False,
                    "error": f"Erreur HTTP {response.status_code}",
                    "response": "Désolé, Magistral n'est pas disponible actuellement."
                }
                
        except Exception as e:
            logger.error(f"Erreur Magistral: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": f"Erreur d'analyse Magistral: {str(e)}"
            }
    
    # === SERVICES VISION ===
    
    async def analyze_image_pixtral(self, image_data: bytes, prompt: str = "Décris cette image en détail") -> Dict[str, Any]:
        """
        Analyse d'image avec Pixtral sur RTX 4090
        
        Args:
            image_data: Données de l'image en bytes
            prompt: Prompt pour l'analyse
            
        Returns:
            Dict avec l'analyse et les métadonnées
        """
        try:
            # Encoder l'image en base64
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            # Prompt enrichi pour l'analyse
            enhanced_prompt = f"""
            Tu es Pixtral, un expert en analyse d'images.
            
            Analyse cette image selon ce prompt : "{prompt}"
            
            Structure ta réponse ainsi :
            
            🔍 **Description Générale**
            - Éléments principaux visibles
            - Composition et style
            
            📊 **Analyse Détaillée**
            - Couleurs et textures
            - Objets et personnes identifiés
            - Contexte et atmosphère
            
            💡 **Insights Spécialisés**
            - Aspects techniques remarquables
            - Interprétation contextuelle
            
            Sois précis et détaillé dans ton analyse.
            """
            
            # Payload pour Pixtral
            payload = {
                "model": AI_MODELS['pixtral']['model_name'],
                "prompt": enhanced_prompt,
                "images": [image_b64],
                "options": {
                    "temperature": AI_MODELS['pixtral']['temperature_default'],
                    "num_predict": AI_MODELS['pixtral']['max_tokens']
                },
                "stream": False
            }
            
            # Appel asynchrone à Pixtral
            start_time = time.time()
            response = await self.client.post(
                f"{settings.PIXTRAL_ENDPOINT}/api/generate",
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                processing_time = time.time() - start_time
                
                return {
                    "success": True,
                    "response": result.get("response", ""),
                    "model": AI_MODELS['pixtral']['name'],
                    "gpu": AI_MODELS['pixtral']['gpu'],
                    "analysis_type": "Vision",
                    "tokens_used": result.get("prompt_eval_count", 0),
                    "processing_time": processing_time,
                    "original_prompt": prompt,
                    "endpoint": settings.PIXTRAL_ENDPOINT
                }
            else:
                logger.error(f"Erreur Pixtral HTTP {response.status_code}")
                return {
                    "success": False,
                    "error": f"Erreur HTTP {response.status_code}",
                    "response": "Désolé, Pixtral n'est pas disponible actuellement."
                }
                
        except Exception as e:
            logger.error(f"Erreur Pixtral: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": f"Erreur d'analyse d'image: {str(e)}"
            }
    
    # === SERVICES AUDIO ===
    
    async def transcribe_whisper(self, audio_data: bytes, language: str = "fr") -> Dict[str, Any]:
        """
        Transcription audio avec Whisper Large V3 sur RTX 3090
        
        Args:
            audio_data: Données audio en bytes
            language: Langue de transcription
            
        Returns:
            Dict avec la transcription et les métadonnées
        """
        try:
            # Préparation des fichiers pour l'upload
            files = {
                'audio': ('audio.wav', audio_data, 'audio/wav')
            }
            
            data = {
                'model': 'whisper-large-v3',
                'language': language,
                'response_format': 'json'
            }
            
            # Appel asynchrone à Whisper
            start_time = time.time()
            response = await self.client.post(
                f"{settings.WHISPER_ENDPOINT}/v1/audio/transcriptions",
                files=files,
                data=data
            )
            
            if response.status_code == 200:
                result = response.json()
                processing_time = time.time() - start_time
                
                return {
                    "success": True,
                    "transcription": result.get("text", ""),
                    "model": AUDIO_SERVICES['whisper']['name'],
                    "gpu": AUDIO_SERVICES['whisper']['gpu'],
                    "language": language,
                    "confidence": result.get("confidence", 0.0),
                    "processing_time": processing_time,
                    "endpoint": settings.WHISPER_ENDPOINT
                }
            else:
                logger.error(f"Erreur Whisper HTTP {response.status_code}")
                return {
                    "success": False,
                    "error": f"Erreur HTTP {response.status_code}",
                    "transcription": "Erreur de transcription audio."
                }
                
        except Exception as e:
            logger.error(f"Erreur Whisper: {e}")
            return {
                "success": False,
                "error": str(e),
                "transcription": f"Erreur de transcription: {str(e)}"
            }
    
    async def generate_tts(self, text: str, voice: str = "female") -> Dict[str, Any]:
        """
        Génération vocale avec TTS5 sur RTX 3090
        
        Args:
            text: Texte à synthétiser
            voice: Voix à utiliser
            
        Returns:
            Dict avec les données audio et les métadonnées
        """
        try:
            # Préparation des paramètres
            payload = {
                'text': text,
                'voice': voice,
                'model': 'tts-5',
                'format': 'wav',
                'speed': 1.0,
                'pitch': 1.0
            }
            
            # Appel asynchrone à TTS5
            start_time = time.time()
            response = await self.client.post(
                f"{settings.TTS_ENDPOINT}/v1/audio/speech",
                json=payload
            )
            
            if response.status_code == 200:
                processing_time = time.time() - start_time
                
                return {
                    "success": True,
                    "audio_data": response.content,
                    "model": AUDIO_SERVICES['tts']['name'],
                    "gpu": AUDIO_SERVICES['tts']['gpu'],
                    "voice": voice,
                    "text_length": len(text),
                    "processing_time": processing_time,
                    "audio_format": "wav",
                    "endpoint": settings.TTS_ENDPOINT
                }
            else:
                logger.error(f"Erreur TTS HTTP {response.status_code}")
                return {
                    "success": False,
                    "error": f"Erreur HTTP {response.status_code}",
                    "audio_data": None
                }
                
        except Exception as e:
            logger.error(f"Erreur TTS: {e}")
            return {
                "success": False,
                "error": str(e),
                "audio_data": None
            }
    
    # === PIPELINE AUDIO COMPLET ===
    
    async def process_audio_pipeline(self, audio_data: bytes, language: str = "fr") -> Dict[str, Any]:
        """
        Pipeline complet optimisé : Audio → Whisper → Mistral → TTS
        
        Args:
            audio_data: Données audio en bytes
            language: Langue de traitement
            
        Returns:
            Dict avec toutes les étapes du pipeline
        """
        try:
            pipeline_start = time.time()
            
            # Étape 1: Transcription avec Whisper
            logger.info("🎙️ Début transcription Whisper")
            transcription_result = await self.transcribe_whisper(audio_data, language)
            
            if not transcription_result['success']:
                return {
                    "success": False,
                    "error": "Erreur de transcription",
                    "step": "whisper",
                    "details": transcription_result
                }
            
            transcription = transcription_result['transcription']
            logger.info(f"✅ Transcription terminée: {transcription[:50]}...")
            
            # Étape 2: Traitement avec Mistral
            logger.info("💬 Début traitement Mistral")
            chat_result = await self.chat_mistral(transcription)
            
            if not chat_result['success']:
                return {
                    "success": False,
                    "error": "Erreur de traitement",
                    "step": "mistral",
                    "transcription": transcription,
                    "details": chat_result
                }
            
            response_text = chat_result['response']
            logger.info(f"✅ Réponse générée: {response_text[:50]}...")
            
            # Étape 3: Synthèse vocale avec TTS
            logger.info("🔊 Début synthèse TTS")
            tts_result = await self.generate_tts(response_text)
            
            if not tts_result['success']:
                return {
                    "success": False,
                    "error": "Erreur de synthèse",
                    "step": "tts",
                    "transcription": transcription,
                    "response": response_text,
                    "details": tts_result
                }
            
            pipeline_time = time.time() - pipeline_start
            logger.info(f"✅ Pipeline complet terminé en {pipeline_time:.2f}s")
            
            # Résultat complet
            return {
                "success": True,
                "transcription": transcription,
                "response": response_text,
                "audio_response": tts_result['audio_data'],
                "pipeline_steps": {
                    "whisper": transcription_result,
                    "mistral": chat_result,
                    "tts": tts_result
                },
                "total_time": pipeline_time,
                "performance_breakdown": {
                    "transcription_time": transcription_result.get('processing_time', 0),
                    "generation_time": chat_result.get('processing_time', 0),
                    "synthesis_time": tts_result.get('processing_time', 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Erreur pipeline audio: {e}")
            return {
                "success": False,
                "error": str(e),
                "step": "pipeline"
            }
    
    # === GÉNÉRATION D'IMAGES ===
    
    async def generate_image_comfyui(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Génération d'image avec ComfyUI sur RTX 4090
        
        Args:
            prompt: Prompt pour la génération
            **kwargs: Paramètres supplémentaires
            
        Returns:
            Dict avec l'image générée et les métadonnées
        """
        try:
            # Paramètres avec valeurs par défaut
            width = kwargs.get('width', 512)
            height = kwargs.get('height', 512)
            steps = kwargs.get('steps', IMAGE_GENERATION['comfyui']['default_steps'])
            cfg_scale = kwargs.get('cfg_scale', IMAGE_GENERATION['comfyui']['default_cfg_scale'])
            
            payload = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "steps": steps,
                "sampler": "euler",
                "cfg_scale": cfg_scale,
                "seed": kwargs.get('seed', -1)
            }
            
            # Appel asynchrone à ComfyUI
            start_time = time.time()
            response = await self.client.post(
                f"{settings.COMFYUI_ENDPOINT}/api/generate",
                json=payload,
                timeout=180  # Plus de temps pour la génération
            )
            
            if response.status_code == 200:
                result = response.json()
                processing_time = time.time() - start_time
                
                return {
                    "success": True,
                    "image_url": result.get("image_url", ""),
                    "image_data": result.get("image_data", ""),
                    "model": IMAGE_GENERATION['comfyui']['name'],
                    "gpu": IMAGE_GENERATION['comfyui']['gpu'],
                    "prompt": prompt,
                    "processing_time": processing_time,
                    "settings": payload,
                    "endpoint": settings.COMFYUI_ENDPOINT
                }
            else:
                logger.error(f"Erreur ComfyUI HTTP {response.status_code}")
                return {
                    "success": False,
                    "error": f"Erreur HTTP {response.status_code}",
                    "image_url": None
                }
                
        except Exception as e:
            logger.error(f"Erreur ComfyUI: {e}")
            return {
                "success": False,
                "error": str(e),
                "image_url": None
            }
    
    # === VÉRIFICATION DE SANTÉ ===
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Vérification de l'état de tous les services
        
        Returns:
            Dict avec l'état de chaque service
        """
        services_status = {}
        
        # Test de tous les services en parallèle
        health_checks = [
            self._check_service("mistral", f"{settings.MISTRAL_ENDPOINT}/api/tags"),
            self._check_service("magistral", f"{settings.MAGISTRAL_ENDPOINT}/api/tags"),
            self._check_service("pixtral", f"{settings.PIXTRAL_ENDPOINT}/api/tags"),
            self._check_service("whisper", f"{settings.WHISPER_ENDPOINT}/health"),
            self._check_service("tts", f"{settings.TTS_ENDPOINT}/health"),
            self._check_service("comfyui", f"{settings.COMFYUI_ENDPOINT}/system_stats")
        ]
        
        # Exécution en parallèle
        results = await asyncio.gather(*health_checks, return_exceptions=True)
        
        # Traitement des résultats
        service_names = ["mistral", "magistral", "pixtral", "whisper", "tts", "comfyui"]
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                services_status[service_names[i]] = {
                    "status": "offline",
                    "error": str(result)
                }
            else:
                services_status[service_names[i]] = result
        
        # Statistiques globales
        online_services = sum(1 for s in services_status.values() if s.get('status') == 'online')
        total_services = len(services_status)
        
        return {
            "services": services_status,
            "summary": {
                "online": online_services,
                "total": total_services,
                "health_percentage": (online_services / total_services) * 100,
                "timestamp": time.time()
            }
        }
    
    async def _check_service(self, service_name: str, endpoint: str) -> Dict[str, Any]:
        """Vérification d'un service spécifique"""
        try:
            response = await self.client.get(endpoint, timeout=5)
            return {
                "status": "online" if response.status_code == 200 else "offline",
                "endpoint": endpoint,
                "response_time": response.elapsed.total_seconds(),
                "status_code": response.status_code
            }
        except Exception as e:
            return {
                "status": "offline",
                "endpoint": endpoint,
                "error": str(e)
            }

# Instance globale
ai_services = OptimizedAIServices()
