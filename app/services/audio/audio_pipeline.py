# app/services/audio/audio_pipeline.py
"""
Pipeline Audio Complet - HemersonAIBuild PlaygroundV1
Whisper V3 → Mistral/Magistral → TTS5
Optimisé pour Docker et performances
"""

import asyncio
import time
import tempfile
import hashlib
from typing import Dict, Any, Optional, BinaryIO
from pathlib import Path
import aiofiles
import httpx
from loguru import logger

from services.rate_limiter.advanced_limiter import CacheManager, RequestType
from services.ai.unified_ai_service import UnifiedAIService

class AudioPipelineService:
    """
    Pipeline Audio Complet avec optimisations avancées
    
    Fonctionnalités :
    - Transcription Whisper V3 haute précision
    - Traitement intelligent du texte
    - Synthèse vocale TTS5 naturelle
    - Cache multi-niveaux pour performances
    - Gestion d'erreurs robuste avec retry
    """
    
    def __init__(self, cache_manager: CacheManager, ai_service: UnifiedAIService):
        self.cache_manager = cache_manager
        self.ai_service = ai_service
        
        # Configuration des endpoints audio
        self.whisper_endpoint = "http://192.168.1.212:8000"
        self.tts_endpoint = "http://192.168.1.214:8001"
        
        # Client HTTP optimisé pour l'audio
        self.audio_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=15.0,
                read=180.0,  # Plus long pour l'audio
                write=120.0,
                pool=300.0
            ),
            limits=httpx.Limits(
                max_keepalive_connections=10,
                max_connections=50,
                keepalive_expiry=60.0
            )
        )
        
        # Métriques spécifiques audio
        self.audio_metrics = {
            "transcriptions_total": 0,
            "transcriptions_successful": 0,
            "synthesis_total": 0,
            "synthesis_successful": 0,
            "pipeline_complete_total": 0,
            "pipeline_complete_successful": 0,
            "average_pipeline_time": 0.0,
            "cache_hits_transcription": 0,
            "cache_hits_synthesis": 0
        }
        
        logger.info("🎵 AudioPipelineService initialisé avec optimisations avancées")
    
    async def transcribe_audio(self,
                              audio_data: bytes,
                              filename: str,
                              language: str = "fr",
                              use_cache: bool = True,
                              **kwargs) -> Dict[str, Any]:
        """
        Transcription audio avec Whisper V3
        
        Args:
            audio_data: Données audio en bytes
            filename: Nom du fichier original
            language: Langue de transcription
            use_cache: Utiliser le cache intelligent
            **kwargs: Paramètres additionnels
        
        Returns:
            Transcription avec métadonnées et confiance
        """
        start_time = time.time()
        self.audio_metrics["transcriptions_total"] += 1
        
        try:
            # Générer une clé de cache basée sur le contenu audio
            if use_cache:
                audio_hash = hashlib.md5(audio_data).hexdigest()
                cache_key = self.cache_manager.generate_cache_key(
                    "transcription", 
                    audio_hash, 
                    {"language": language, "filename": filename}
                )
                
                cached_result = await self.cache_manager.get_cached_response(cache_key)
                if cached_result:
                    self.audio_metrics["cache_hits_transcription"] += 1
                    logger.info(f"💾 Transcription depuis cache: {filename}")
                    return cached_result
            
            # Validation du format audio
            audio_info = self._analyze_audio_format(audio_data, filename)
            if not audio_info["is_valid"]:
                return {
                    "success": False,
                    "error": f"Format audio non supporté: {audio_info['error']}",
                    "transcription": "",
                    "confidence": 0.0
                }
            
            # Préparer les données pour Whisper
            files = {
                "file": (filename, audio_data, audio_info["mime_type"])
            }
            
            data = {
                "model": "whisper-large-v3",
                "language": language,
                "response_format": "verbose_json",
                "temperature": kwargs.get("temperature", 0.1),
                "timestamp_granularities[]": ["segment", "word"]
            }
            
            # Requête de transcription avec retry
            result = await self._make_audio_request(
                f"{self.whisper_endpoint}/v1/audio/transcriptions",
                files=files,
                data=data,
                max_retries=2,
                operation="transcription"
            )
            
            if result["success"]:
                response_data = result["data"]
                
                # Analyser la qualité de transcription
                quality_analysis = self._analyze_transcription_quality(response_data)
                
                transcription_result = {
                    "success": True,
                    "transcription": response_data.get("text", ""),
                    "language": response_data.get("language", language),
                    "confidence": quality_analysis["overall_confidence"],
                    "duration": response_data.get("duration", 0.0),
                    "segments": response_data.get("segments", []),
                    "words": response_data.get("words", []),
                    "processing_time": time.time() - start_time,
                    "audio_info": audio_info,
                    "quality_analysis": quality_analysis,
                    "model": "Whisper Large V3",
                    "gpu": "RTX 3090 (192.168.1.212)"
                }
                
                # Mettre en cache (TTL plus long pour transcriptions)
                if use_cache:
                    await self.cache_manager.cache_response(
                        cache_key,
                        transcription_result,
                        ttl=7200  # 2 heures
                    )
                
                self.audio_metrics["transcriptions_successful"] += 1
                logger.info(f"✅ Transcription réussie: {filename} ({len(response_data.get('text', ''))} chars)")
                
                return transcription_result
            else:
                return {
                    "success": False,
                    "error": result["error"],
                    "transcription": "Erreur de transcription audio.",
                    "confidence": 0.0,
                    "processing_time": time.time() - start_time
                }
                
        except Exception as e:
            logger.error(f"❌ Erreur transcription audio: {e}")
            return {
                "success": False,
                "error": str(e),
                "transcription": f"Erreur lors de la transcription: {str(e)}",
                "confidence": 0.0,
                "processing_time": time.time() - start_time
            }
    
    async def synthesize_speech(self,
                               text: str,
                               voice: str = "female",
                               language: str = "fr",
                               use_cache: bool = True,
                               **kwargs) -> Dict[str, Any]:
        """
        Synthèse vocale avec TTS5
        
        Args:
            text: Texte à synthétiser
            voice: Voix à utiliser
            language: Langue de synthèse
            use_cache: Utiliser le cache
            **kwargs: Paramètres additionnels
            
        Returns:
            Audio synthétisé avec métadonnées
        """
        start_time = time.time()
        self.audio_metrics["synthesis_total"] += 1
        
        try:
            # Validation et nettoyage du texte
            cleaned_text = self._prepare_text_for_synthesis(text)
            if not cleaned_text:
                return {
                    "success": False,
                    "error": "Texte vide ou invalide pour la synthèse",
                    "audio_data": None
                }
            
            # Cache pour synthèse
            if use_cache:
                cache_key = self.cache_manager.generate_cache_key(
                    "tts_synthesis",
                    cleaned_text,
                    {"voice": voice, "language": language}
                )
                
                cached_result = await self.cache_manager.get_cached_response(cache_key)
                if cached_result:
                    self.audio_metrics["cache_hits_synthesis"] += 1
                    logger.info("💾 Synthèse depuis cache")
                    return cached_result
            
            # Préparer les paramètres TTS5
            tts_payload = {
                "input": cleaned_text,
                "model": "tts-5-hd",
                "voice": voice,
                "language": language,
                "response_format": kwargs.get("format", "wav"),
                "speed": kwargs.get("speed", 1.0),
                "pitch": kwargs.get("pitch", 1.0),
                "volume": kwargs.get("volume", 1.0),
                "sample_rate": kwargs.get("sample_rate", 24000)
            }
            
            # Requête de synthèse
            result = await self._make_audio_request(
                f"{self.tts_endpoint}/v1/audio/speech",
                json_data=tts_payload,
                max_retries=2,
                operation="synthesis",
                expect_binary=True
            )
            
            if result["success"]:
                audio_data = result["data"]
                
                # Analyser l'audio généré
                audio_analysis = self._analyze_generated_audio(audio_data)
                
                synthesis_result = {
                    "success": True,
                    "audio_data": audio_data,
                    "text_input": cleaned_text,
                    "voice": voice,
                    "language": language,
                    "audio_format": tts_payload["response_format"],
                    "audio_size": len(audio_data),
                    "processing_time": time.time() - start_time,
                    "audio_analysis": audio_analysis,
                    "model": "TTS5 HD",
                    "gpu": "RTX 3090 (192.168.1.214)"
                }
                
                # Cache avec TTL adapté à la taille
                if use_cache:
                    cache_ttl = 3600  # 1 heure par défaut
                    if len(audio_data) > 1024 * 1024:  # > 1MB
                        cache_ttl = 1800  # 30 minutes pour gros fichiers
                    
                    await self.cache_manager.cache_response(
                        cache_key,
                        synthesis_result,
                        ttl=cache_ttl
                    )
                
                self.audio_metrics["synthesis_successful"] += 1
                logger.info(f"✅ Synthèse réussie: {len(cleaned_text)} chars → {len(audio_data)} bytes")
                
                return synthesis_result
            else:
                return {
                    "success": False,
                    "error": result["error"],
                    "audio_data": None,
                    "processing_time": time.time() - start_time
                }
                
        except Exception as e:
            logger.error(f"❌ Erreur synthèse vocale: {e}")
            return {
                "success": False,
                "error": str(e),
                "audio_data": None,
                "processing_time": time.time() - start_time
            }
    
    async def process_complete_pipeline(self,
                                      audio_data: bytes,
                                      filename: str,
                                      response_voice: str = "female",
                                      language: str = "fr",
                                      analysis_mode: str = "conversational",
                                      **kwargs) -> Dict[str, Any]:
        """
        Pipeline Audio Complet Optimisé
        
        Audio → Whisper → Mistral/Magistral → TTS5
        
        Args:
            audio_data: Données audio input
            filename: Nom du fichier
            response_voice: Voix pour la réponse
            language: Langue de traitement
            analysis_mode: Mode d'analyse (conversational, cogito, expert)
            **kwargs: Paramètres additionnels
            
        Returns:
            Résultat complet du pipeline avec métriques
        """
        pipeline_start = time.time()
        self.audio_metrics["pipeline_complete_total"] += 1
        
        pipeline_results = {
            "transcription": None,
            "ai_processing": None,
            "speech_synthesis": None
        }
        
        try:
            logger.info(f"🎵 Début pipeline audio complet: {filename} (mode: {analysis_mode})")
            
            # ÉTAPE 1: Transcription avec Whisper
            logger.info("1️⃣ Transcription audio avec Whisper Large V3...")
            transcription_result = await self.transcribe_audio(
                audio_data=audio_data,
                filename=filename,
                language=language,
                **kwargs
            )
            pipeline_results["transcription"] = transcription_result
            
            if not transcription_result["success"]:
                return {
                    "success": False,
                    "error": "Échec de la transcription",
                    "step_failed": "transcription",
                    "pipeline_results": pipeline_results,
                    "total_time": time.time() - pipeline_start
                }
            
            transcription = transcription_result["transcription"]
            logger.info(f"✅ Transcription: '{transcription[:100]}...'")
            
            # ÉTAPE 2: Traitement IA selon le mode
            logger.info(f"2️⃣ Traitement IA en mode {analysis_mode}...")
            
            if analysis_mode == "cogito":
                # Mode Cogito avec Magistral
                ai_result = await self.ai_service.chat_with_cot(
                    message=f"Message audio transcrit: {transcription}",
                    cot_type="cogito",
                    **kwargs
                )
            elif analysis_mode == "expert":
                ai_result = await self.ai_service.chat_with_cot(
                    message=f"Analyse experte demandée pour: {transcription}",
                    cot_type="expert",
                    **kwargs
                )
            else:
                # Mode conversationnel avec Mistral
                conversational_prompt = f"""
L'utilisateur vient d'envoyer un message audio qui a été transcrit.

Transcription: "{transcription}"

Réponds de manière naturelle et conversationnelle, comme si tu avais écouté le message audio directement. Adapte ton style à la spontanéité de la communication orale. Sois chaleureux et engageant.
"""
                ai_result = await self.ai_service.chat_simple(
                    message=conversational_prompt,
                    **kwargs
                )
            
            pipeline_results["ai_processing"] = ai_result
            
            if not ai_result["success"]:
                return {
                    "success": False,
                    "error": "Échec du traitement IA",
                    "step_failed": "ai_processing",
                    "transcription": transcription,
                    "pipeline_results": pipeline_results,
                    "total_time": time.time() - pipeline_start
                }
            
            ai_response = ai_result["response"]
            logger.info(f"✅ Réponse IA: '{ai_response[:100]}...'")
            
            # ÉTAPE 3: Synthèse vocale
            logger.info("3️⃣ Synthèse vocale avec TTS5...")
            synthesis_result = await self.synthesize_speech(
                text=ai_response,
                voice=response_voice,
                language=language,
                **kwargs
            )
            pipeline_results["speech_synthesis"] = synthesis_result
            
            if not synthesis_result["success"]:
                return {
                    "success": False,
                    "error": "Échec de la synthèse vocale",
                    "step_failed": "speech_synthesis",
                    "transcription": transcription,
                    "ai_response": ai_response,
                    "pipeline_results": pipeline_results,
                    "total_time": time.time() - pipeline_start
                }
            
            # Pipeline complet réussi
            total_time = time.time() - pipeline_start
            self.audio_metrics["pipeline_complete_successful"] += 1
            
            # Mise à jour métriques
            current_avg = self.audio_metrics["average_pipeline_time"]
            total_count = self.audio_metrics["pipeline_complete_total"]
            self.audio_metrics["average_pipeline_time"] = (
                (current_avg * (total_count - 1) + total_time) / total_count
            )
            
            logger.info(f"🎉 Pipeline audio complet réussi en {total_time:.2f}s")
            
            return {
                "success": True,
                "original_filename": filename,
                "transcription": transcription,
                "ai_response": ai_response,
                "response_audio": synthesis_result["audio_data"],
                "analysis_mode": analysis_mode,
                "language": language,
                "voice": response_voice,
                "pipeline_results": pipeline_results,
                "performance_metrics": {
                    "total_time": total_time,
                    "transcription_time": transcription_result.get("processing_time", 0),
                    "ai_processing_time": ai_result.get("processing_time", 0),
                    "synthesis_time": synthesis_result.get("processing_time", 0),
                    "transcription_confidence": transcription_result.get("confidence", 0),
                    "original_audio_size": len(audio_data),
                    "response_audio_size": len(synthesis_result["audio_data"])
                },
                "quality_indicators": {
                    "transcription_quality": transcription_result.get("quality_analysis", {}),
                    "audio_output_quality": synthesis_result.get("audio_analysis", {}),
                    "pipeline_efficiency": self._calculate_pipeline_efficiency(total_time, len(audio_data))
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur pipeline audio complet: {e}")
            return {
                "success": False,
                "error": str(e),
                "step_failed": "pipeline_exception",
                "pipeline_results": pipeline_results,
                "total_time": time.time() - pipeline_start
            }
    
    def _analyze_audio_format(self, audio_data: bytes, filename: str) -> Dict[str, Any]:
        """Analyse le format audio et valide la compatibilité"""
        try:
            # Détecter le format basé sur les magic bytes
            if audio_data.startswith(b'RIFF') and b'WAVE' in audio_data[:20]:
                return {
                    "is_valid": True,
                    "format": "wav",
                    "mime_type": "audio/wav",
                    "estimated_duration": self._estimate_wav_duration(audio_data)
                }
            elif audio_data.startswith(b'\xff\xfb') or audio_data.startswith(b'ID3'):
                return {
                    "is_valid": True,
                    "format": "mp3",
                    "mime_type": "audio/mpeg",
                    "estimated_duration": None  # Plus complexe à calculer
                }
            elif audio_data.startswith(b'ftyp'):
                return {
                    "is_valid": True,
                    "format": "m4a",
                    "mime_type": "audio/mp4",
                    "estimated_duration": None
                }
            else:
                return {
                    "is_valid": False,
                    "error": "Format audio non reconnu",
                    "format": "unknown"
                }
        except Exception as e:
            return {
                "is_valid": False,
                "error": f"Erreur d'analyse: {str(e)}",
                "format": "error"
            }
    
    def _estimate_wav_duration(self, wav_data: bytes) -> Optional[float]:
        """Estime la durée d'un fichier WAV"""
        try:
            if len(wav_data) < 44:  # Header WAV minimum
                return None
            
            # Extraire les informations du header WAV
            sample_rate = int.from_bytes(wav_data[24:28], byteorder='little')
            byte_rate = int.from_bytes(wav_data[28:32], byteorder='little')
            data_size = len(wav_data) - 44  # Approximation
            
            if byte_rate > 0:
                duration = data_size / byte_rate
                return duration
            return None
        except:
            return None
    
    def _analyze_transcription_quality(self, whisper_response: Dict) -> Dict[str, Any]:
        """Analyse la qualité de transcription"""
        try:
            segments = whisper_response.get("segments", [])
            
            if not segments:
                return {"overall_confidence": 0.5, "quality": "unknown"}
            
            # Calculer confiance moyenne
            confidences = []
            for segment in segments:
                if "avg_logprob" in segment:
                    # Convertir log prob en confiance approximative
                    confidence = max(0.0, min(1.0, (segment["avg_logprob"] + 1) / 2))
                    confidences.append(confidence)
            
            overall_confidence = sum(confidences) / len(confidences) if confidences else 0.5
            
            # Analyser la qualité textuelle
            text = whisper_response.get("text", "")
            quality_indicators = {
                "text_length": len(text),
                "word_count": len(text.split()),
                "has_punctuation": any(p in text for p in ".,!?;:"),
                "segments_count": len(segments),
                "overall_confidence": overall_confidence
            }
            
            # Déterminer la qualité globale
            if overall_confidence > 0.8:
                quality = "excellent"
            elif overall_confidence > 0.6:
                quality = "good"
            elif overall_confidence > 0.4:
                quality = "fair"
            else:
                quality = "poor"
            
            return {
                "overall_confidence": overall_confidence,
                "quality": quality,
                **quality_indicators
            }
            
        except Exception as e:
            logger.warning(f"Erreur analyse qualité transcription: {e}")
            return {"overall_confidence": 0.5, "quality": "unknown"}
    
    def _prepare_text_for_synthesis(self, text: str) -> str:
        """Prépare et nettoie le texte pour la synthèse"""
        try:
            if not text or not text.strip():
                return ""
            
            # Nettoyage de base
            cleaned = text.strip()
            
            # Limiter la longueur pour TTS
            max_length = 1000
            if len(cleaned) > max_length:
                # Couper au dernier point ou à la limite
                cut_index = cleaned.rfind('.', 0, max_length)
                if cut_index > max_length // 2:
                    cleaned = cleaned[:cut_index + 1]
                else:
                    cleaned = cleaned[:max_length] + "..."
            
            # Échapper les caractères problématiques
            cleaned = cleaned.replace('&', ' et ')
            cleaned = cleaned.replace('<', ' inférieur à ')
            cleaned = cleaned.replace('>', ' supérieur à ')
            
            return cleaned
            
        except Exception as e:
            logger.warning(f"Erreur nettoyage texte: {e}")
            return text[:500] if text else ""
    
    def _analyze_generated_audio(self, audio_data: bytes) -> Dict[str, Any]:
        """Analyse la qualité de l'audio généré"""
        try:
            analysis = {
                "size_bytes": len(audio_data),
                "size_mb": len(audio_data) / (1024 * 1024),
                "estimated_duration": None,
                "quality": "unknown"
            }
            
            # Estimation basique de durée pour WAV
            if audio_data.startswith(b'RIFF'):
                duration = self._estimate_wav_duration(audio_data)
                analysis["estimated_duration"] = duration
                
                # Qualité basée sur la taille vs durée
                if duration and duration > 0:
                    bitrate_estimate = (len(audio_data) * 8) / duration
                    if bitrate_estimate > 192000:
                        analysis["quality"] = "high"
                    elif bitrate_estimate > 128000:
                        analysis["quality"] = "medium"
                    else:
                        analysis["quality"] = "low"
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Erreur analyse audio généré: {e}")
            return {"size_bytes": len(audio_data), "quality": "unknown"}
    
    def _calculate_pipeline_efficiency(self, total_time: float, input_size: int) -> str:
        """Calcule l'efficacité du pipeline"""
        try:
            # Efficacité basée sur le temps et la taille
            mb_per_second = (input_size / (1024 * 1024)) / total_time if total_time > 0 else 0
            
            if mb_per_second > 2.0:
                return "excellent"
            elif mb_per_second > 1.0:
                return "good"
            elif mb_per_second > 0.5:
                return "fair"
            else:
                return "slow"
                
        except:
            return "unknown"
    
    async def _make_audio_request(self,
                                 endpoint: str,
                                 files: Optional[Dict] = None,
                                 data: Optional[Dict] = None,
                                 json_data: Optional[Dict] = None,
                                 max_retries: int = 3,
                                 operation: str = "audio_request",
                                 expect_binary: bool = False) -> Dict[str, Any]:
        """Requête HTTP spécialisée pour l'audio avec retry"""
        
        for attempt in range(max_retries):
            try:
                if files:
                    # Requête multipart pour upload
                    response = await self.audio_client.post(
                        endpoint,
                        files=files,
                        data=data
                    )
                else:
                    # Requête JSON
                    response = await self.audio_client.post(
                        endpoint,
                        json=json_data
                    )
                
                if response.status_code == 200:
                    if expect_binary:
                        return {"success": True, "data": response.content}
                    else:
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
    
    async def get_audio_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques audio détaillées"""
        total_transcriptions = self.audio_metrics["transcriptions_total"]
        total_synthesis = self.audio_metrics["synthesis_total"]
        total_pipelines = self.audio_metrics["pipeline_complete_total"]
        
        return {
            **self.audio_metrics,
            "success_rates": {
                "transcription": (
                    self.audio_metrics["transcriptions_successful"] / max(total_transcriptions, 1)
                ) * 100,
                "synthesis": (
                    self.audio_metrics["synthesis_successful"] / max(total_synthesis, 1)
                ) * 100,
                "pipeline_complete": (
                    self.audio_metrics["pipeline_complete_successful"] / max(total_pipelines, 1)
                ) * 100
            },
            "cache_efficiency": {
                "transcription_hit_rate": (
                    self.audio_metrics["cache_hits_transcription"] / max(total_transcriptions, 1)
                ) * 100,
                "synthesis_hit_rate": (
                    self.audio_metrics["cache_hits_synthesis"] / max(total_synthesis, 1)
                ) * 100
            }
        }
    
    async def close(self):
        """Fermeture propre du service audio"""
        await self.audio_client.aclose()
        logger.info("🔌 AudioPipelineService fermé proprement")
