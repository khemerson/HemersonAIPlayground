# app/services/ai/unified_ai_service.py
"""
Service IA unifié avec Chain of Thought (COT)
Priorité 2 : Cœur métier du playground
"""

import asyncio
import time
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

import httpx
from loguru import logger
from services.rate_limiter.advanced_limiter import CacheManager

@dataclass
class COTStep:
    """Étape de Chain of Thought"""
    step_number: int
    title: str
    content: str
    reasoning: str
    confidence: float

class ModelType(Enum):
    """Types de modèles disponibles"""
    MISTRAL = "mistral"      # Conversation générale
    MAGISTRAL = "magistral"  # Analyse approfondie avec COT

class UnifiedAIService:
    """
    Service IA unifié avec support COT avancé
    
    Fonctionnalités :
    - Chat normal et Chat avec COT
    - Cache intelligent des réponses
    - Monitoring des performances
    - Retry automatique avec backoff
    """
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        
        # Configuration des endpoints (depuis settings)
        self.endpoints = {
            ModelType.MISTRAL: "http://192.168.1.211:11434",
            ModelType.MAGISTRAL: "http://192.168.1.214:11434"
        }
        
        # Client HTTP optimisé
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0,
                read=180.0,  # Timeout plus long pour COT
                write=60.0,
                pool=240.0
            ),
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=100,
                keepalive_expiry=30.0
            )
        )
        
        # Métriques de performance
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "cached_responses": 0,
            "average_response_time": 0.0,
            "cot_requests": 0
        }
        
        logger.info("🤖 UnifiedAIService initialisé avec support COT")
    
    async def chat_simple(self, 
                         message: str, 
                         model: ModelType = ModelType.MISTRAL,
                         use_cache: bool = True,
                         **kwargs) -> Dict[str, Any]:
        """
        Chat simple sans COT
        
        Args:
            message: Message utilisateur
            model: Modèle à utiliser
            use_cache: Utiliser le cache
            **kwargs: Paramètres additionnels
            
        Returns:
            Réponse du modèle avec métadonnées
        """
        start_time = time.time()
        
        try:
            # Vérifier le cache si activé
            if use_cache:
                cache_key = self.cache_manager.generate_cache_key(
                    f"chat_simple_{model.value}", 
                    message, 
                    kwargs
                )
                
                cached_response = await self.cache_manager.get_cached_response(cache_key)
                if cached_response:
                    self.metrics["cached_responses"] += 1
                    logger.info(f"💾 Réponse depuis cache: {model.value}")
                    return cached_response
            
            # Préparer la requête
            payload = {
                "model": model.value,
                "prompt": message,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", 0.7),
                    "num_predict": kwargs.get("max_tokens", 2000),
                    "top_k": 40,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
            }
            
            # Faire la requête avec retry
            result = await self._make_request_with_retry(
                f"{self.endpoints[model]}/api/generate",
                payload
            )
            
            if result["success"]:
                response_data = {
                    "success": True,
                    "response": result["data"].get("response", ""),
                    "model": model.value,
                    "type": "simple_chat",
                    "processing_time": time.time() - start_time,
                    "tokens_used": result["data"].get("eval_count", 0),
                    "cached": False
                }
                
                # Mettre en cache
                if use_cache:
                    await self.cache_manager.cache_response(
                        cache_key, 
                        response_data, 
                        ttl=1800  # 30 minutes
                    )
                
                self._update_metrics(True, time.time() - start_time)
                return response_data
            else:
                self._update_metrics(False, time.time() - start_time)
                return {
                    "success": False,
                    "error": result["error"],
                    "model": model.value,
                    "response": f"Le modèle {model.value} n'est pas disponible actuellement."
                }
                
        except Exception as e:
            logger.error(f"❌ Erreur chat simple: {e}")
            self._update_metrics(False, time.time() - start_time)
            return {
                "success": False,
                "error": str(e),
                "model": model.value,
                "response": "Une erreur est survenue lors du traitement."
            }
    
    async def chat_with_cot(self, 
                           message: str,
                           cot_type: str = "cogito",
                           use_cache: bool = True,
                           **kwargs) -> Dict[str, Any]:
        """
        Chat avec Chain of Thought (COT)
        
        Args:
            message: Message utilisateur
            cot_type: Type de COT (cogito, expert, creative)
            use_cache: Utiliser le cache
            **kwargs: Paramètres additionnels
            
        Returns:
            Réponse avec étapes COT détaillées
        """
        start_time = time.time()
        self.metrics["cot_requests"] += 1
        
        try:
            # Vérifier le cache
            if use_cache:
                cache_key = self.cache_manager.generate_cache_key(
                    f"chat_cot_{cot_type}", 
                    message, 
                    kwargs
                )
                
                cached_response = await self.cache_manager.get_cached_response(cache_key)
                if cached_response:
                    self.metrics["cached_responses"] += 1
                    logger.info(f"💾 COT depuis cache: {cot_type}")
                    return cached_response
            
            # Prompts COT spécialisés
            cot_prompt = self._build_cot_prompt(message, cot_type)
            
            # Requête avec timeout étendu pour COT
            payload = {
                "model": ModelType.MAGISTRAL.value,
                "prompt": cot_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Plus déterministe pour COT
                    "num_predict": 4000,  # Plus de tokens pour le raisonnement
                    "top_k": 10,
                    "top_p": 0.8
                }
            }
            
            result = await self._make_request_with_retry(
                f"{self.endpoints[ModelType.MAGISTRAL]}/api/generate",
                payload,
                max_retries=2  # Moins de retries pour COT (plus long)
            )
            
            if result["success"]:
                raw_response = result["data"].get("response", "")
                
                # Parser les étapes COT
                cot_steps = self._parse_cot_response(raw_response)
                
                response_data = {
                    "success": True,
                    "response": raw_response,
                    "cot_steps": cot_steps,
                    "model": ModelType.MAGISTRAL.value,
                    "type": f"cot_{cot_type}",
                    "processing_time": time.time() - start_time,
                    "tokens_used": result["data"].get("eval_count", 0),
                    "reasoning_quality": self._assess_reasoning_quality(cot_steps),
                    "cached": False
                }
                
                # Cache avec TTL plus long pour COT (plus coûteux)
                if use_cache:
                    await self.cache_manager.cache_response(
                        cache_key, 
                        response_data, 
                        ttl=3600  # 1 heure
                    )
                
                self._update_metrics(True, time.time() - start_time)
                return response_data
            else:
                self._update_metrics(False, time.time() - start_time)
                return {
                    "success": False,
                    "error": result["error"],
                    "model": ModelType.MAGISTRAL.value,
                    "response": "Magistral n'est pas disponible pour l'analyse COT."
                }
                
        except Exception as e:
            logger.error(f"❌ Erreur COT: {e}")
            self._update_metrics(False, time.time() - start_time)
            return {
                "success": False,
                "error": str(e),
                "model": ModelType.MAGISTRAL.value,
                "response": "Erreur lors de l'analyse COT."
            }
    
    def _build_cot_prompt(self, message: str, cot_type: str) -> str:
        """Construit le prompt COT selon le type"""
        base_instruction = """Tu es Magistral, un assistant IA expert en raisonnement structuré.

IMPORTANT : Structure ta réponse avec des étapes de raisonnement claires et numérotées.

Utilise ce format EXACT :

🧠 **ÉTAPE 1 : COMPRÉHENSION**
[Analyse ce que demande l'utilisateur]

🔍 **ÉTAPE 2 : ANALYSE**
[Décompose le problème en sous-parties]

💡 **ÉTAPE 3 : RAISONNEMENT**
[Explique ton processus de réflexion]

⚡ **ÉTAPE 4 : SYNTHÈSE**
[Conclusion et réponse finale]

🎯 **ÉTAPE 5 : VALIDATION**
[Vérifie la cohérence de ta réponse]"""
        
        cot_prompts = {
            "cogito": f"""{base_instruction}

Effectue une analyse COGITO (raisonnement approfondi) de cette requête :
"{message}"

Applique une réflexion métacognitive : pense à ta façon de penser.""",
            
            "expert": f"""{base_instruction}

Mode EXPERT : Analyse technique pointue de cette requête :
"{message}"

Mobilise tes connaissances expertes et propose des solutions concrètes.""",
            
            "creative": f"""{base_instruction}

Mode CRÉATIF : Approche innovante de cette requête :
"{message}"

Explore des angles originaux et des solutions créatives."""
        }
        
        return cot_prompts.get(cot_type, cot_prompts["cogito"])
    
    def _parse_cot_response(self, response: str) -> List[COTStep]:
        """Parse la réponse pour extraire les étapes COT"""
        steps = []
        lines = response.split('\n')
        
        current_step = None
        step_content = []
        step_number = 0
        
        for line in lines:
            # Détecter le début d'une étape
            if line.strip().startswith(('🧠', '🔍', '💡', '⚡', '🎯')) and '**ÉTAPE' in line:
                # Sauvegarder l'étape précédente
                if current_step and step_content:
                    steps.append(COTStep(
                        step_number=step_number,
                        title=current_step,
                        content='\n'.join(step_content),
                        reasoning=self._extract_reasoning(step_content),
                        confidence=self._assess_confidence(step_content)
                    ))
                
                # Nouvelle étape
                step_number += 1
                current_step = line.strip()
                step_content = []
            else:
                if line.strip():
                    step_content.append(line.strip())
        
        # Dernière étape
        if current_step and step_content:
            steps.append(COTStep(
                step_number=step_number,
                title=current_step,
                content='\n'.join(step_content),
                reasoning=self._extract_reasoning(step_content),
                confidence=self._assess_confidence(step_content)
            ))
        
        return steps
    
    def _extract_reasoning(self, content: List[str]) -> str:
        """Extrait le raisonnement d'une étape"""
        # Chercher des indicateurs de raisonnement
        reasoning_indicators = ['parce que', 'car', 'donc', 'ainsi', 'par conséquent']
        reasoning_lines = []
        
        for line in content:
            if any(indicator in line.lower() for indicator in reasoning_indicators):
                reasoning_lines.append(line)
        
        return ' '.join(reasoning_lines) if reasoning_lines else content[0] if content else ""
    
    def _assess_confidence(self, content: List[str]) -> float:
        """Évalue la confiance d'une étape"""
        # Indicateurs de confiance
        high_confidence = ['certainement', 'clairement', 'évidemment', 'sans doute']
        low_confidence = ['peut-être', 'probablement', 'possiblement', 'semble']
        
        text = ' '.join(content).lower()
        
        high_count = sum(1 for indicator in high_confidence if indicator in text)
        low_count = sum(1 for indicator in low_confidence if indicator in text)
        
        # Score de base + ajustements
        base_score = 0.7
        confidence = base_score + (high_count * 0.1) - (low_count * 0.15)
        
        return max(0.1, min(1.0, confidence))
    
    def _assess_reasoning_quality(self, steps: List[COTStep]) -> Dict[str, Any]:
        """Évalue la qualité du raisonnement COT"""
        if not steps:
            return {"quality": "poor", "score": 0.0}
        
        # Métriques de qualité
        avg_confidence = sum(step.confidence for step in steps) / len(steps)
        step_count = len(steps)
        total_content_length = sum(len(step.content) for step in steps)
        
        # Score de qualité
        quality_score = (
            (avg_confidence * 0.4) +
            (min(step_count / 5, 1.0) * 0.3) +
            (min(total_content_length / 1000, 1.0) * 0.3)
        )
        
        quality_levels = [
            (0.8, "excellent"),
            (0.6, "good"),
            (0.4, "fair"),
            (0.0, "poor")
        ]
        
        quality_label = next(label for threshold, label in quality_levels if quality_score >= threshold)
        
        return {
            "quality": quality_label,
            "score": quality_score,
            "step_count": step_count,
            "avg_confidence": avg_confidence,
            "reasoning_depth": total_content_length / max(step_count, 1)
        }
    
    async def _make_request_with_retry(self, 
                                     endpoint: str, 
                                     payload: Dict[str, Any],
                                     max_retries: int = 3) -> Dict[str, Any]:
        """Requête HTTP avec retry et backoff exponentiel"""
        
        for attempt in range(max_retries):
            try:
                response = await self.client.post(
                    endpoint,
                    json=payload,
                    timeout=180  # 3 minutes pour COT
                )
                
                if response.status_code == 200:
                    return {
                        "success": True,
                        "data": response.json()
                    }
                else:
                    logger.warning(f"⚠️ HTTP {response.status_code} (tentative {attempt + 1})")
                    
            except httpx.TimeoutException:
                logger.warning(f"⏰ Timeout (tentative {attempt + 1})")
            except Exception as e:
                logger.error(f"❌ Erreur requête: {e}")
            
            # Backoff exponentiel
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"⏳ Attente {wait_time}s avant retry...")
                await asyncio.sleep(wait_time)
        
        return {
            "success": False,
            "error": f"Échec après {max_retries} tentatives"
        }
    
    def _update_metrics(self, success: bool, response_time: float):
        """Met à jour les métriques de performance"""
        self.metrics["total_requests"] += 1
        
        if success:
            self.metrics["successful_requests"] += 1
        
        # Moyenne mobile du temps de réponse
        total = self.metrics["total_requests"]
        current_avg = self.metrics["average_response_time"]
        self.metrics["average_response_time"] = (
            (current_avg * (total - 1) + response_time) / total
        )
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques de performance"""
        success_rate = (
            (self.metrics["successful_requests"] / max(self.metrics["total_requests"], 1)) * 100
        )
        
        cache_hit_rate = (
            (self.metrics["cached_responses"] / max(self.metrics["total_requests"], 1)) * 100
        )
        
        return {
            **self.metrics,
            "success_rate": success_rate,
            "cache_hit_rate": cache_hit_rate,
            "cot_usage_rate": (
                (self.metrics["cot_requests"] / max(self.metrics["total_requests"], 1)) * 100
            )
        }
    
    async def close(self):
        """Fermeture propre du service"""
        await self.client.aclose()
        logger.info("🔌 UnifiedAIService fermé proprement")
