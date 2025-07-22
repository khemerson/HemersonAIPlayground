# app/services/rate_limiter/advanced_limiter.py
"""
Rate Limiter Avancé pour HemersonAIBuild PlaygroundV1
Priorité absolue : Prévenir les abus sans authentification
"""

import asyncio
import time
import hashlib
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum
import json

import redis.asyncio as redis
from fastapi import Request
from loguru import logger

@dataclass
class RateLimit:
    """Configuration d'une limite de débit"""
    requests: int  # Nombre de requêtes autorisées
    window: int    # Fenêtre de temps en secondes
    burst: int     # Rafale autorisée
    
class RequestType(Enum):
    """Types de requêtes avec limites différenciées"""
    CHAT = "chat"
    AUDIO_PIPELINE = "audio_pipeline"
    IMAGE_ANALYSIS = "image_analysis"
    IMAGE_GENERATION = "image_generation"
    HEALTH_CHECK = "health_check"

class AdvancedRateLimiter:
    """
    Rate Limiter multi-niveaux avec Redis
    
    Fonctionnalités :
    - Limitation par IP, session et type de requête
    - Escalade progressive des restrictions
    - Support burst pour pics de trafic
    - Métriques en temps réel
    """
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        
        # Configuration des limites par type de requête
        # Explication : Limites différenciées selon la charge système
        self.limits = {
            RequestType.CHAT: RateLimit(20, 60, 5),           # 20/min, burst 5
            RequestType.AUDIO_PIPELINE: RateLimit(5, 60, 2),  # 5/min, burst 2 (coûteux)
            RequestType.IMAGE_ANALYSIS: RateLimit(10, 60, 3), # 10/min, burst 3
            RequestType.IMAGE_GENERATION: RateLimit(3, 300, 1), # 3/5min, burst 1 (très coûteux)
            RequestType.HEALTH_CHECK: RateLimit(60, 60, 10),  # 60/min, burst 10
        }
        
        # Limites globales par IP (anti-DDoS)
        self.global_limits = {
            "requests_per_minute": 50,
            "requests_per_hour": 500,
            "requests_per_day": 2000
        }
        
        logger.info("🚦 AdvancedRateLimiter initialisé avec limites multi-niveaux")
    
    async def check_rate_limit(self, 
                             request: Request,
                             request_type: RequestType,
                             user_id: Optional[str] = None) -> Tuple[bool, Dict[str, any]]:
        """
        Vérification complète des limites de débit
        
        Args:
            request: Requête FastAPI pour extraction IP
            request_type: Type de requête
            user_id: ID utilisateur (optionnel)
            
        Returns:
            (is_allowed, limit_info)
        """
        try:
            # Identifier l'utilisateur (IP + session)
            client_ip = self._get_client_ip(request)
            identifier = user_id or client_ip
            
            # Vérifications en parallèle
            checks = await asyncio.gather(
                self._check_specific_limit(identifier, request_type),
                self._check_global_limits(client_ip),
                self._check_burst_protection(identifier, request_type),
                return_exceptions=True
            )
            
            # Analyser les résultats
            specific_allowed, specific_info = checks[0] if not isinstance(checks[0], Exception) else (True, {})
            global_allowed, global_info = checks[1] if not isinstance(checks[1], Exception) else (True, {})
            burst_allowed, burst_info = checks[2] if not isinstance(checks[2], Exception) else (True, {})
            
            # Décision finale
            is_allowed = all([specific_allowed, global_allowed, burst_allowed])
            
            # Informations détaillées
            limit_info = {
                "allowed": is_allowed,
                "identifier": identifier,
                "request_type": request_type.value,
                "specific_limit": specific_info,
                "global_limit": global_info,
                "burst_protection": burst_info,
                "timestamp": time.time()
            }
            
            if not is_allowed:
                logger.warning(f"🚫 Rate limit dépassé: {identifier} - {request_type.value}")
            
            return is_allowed, limit_info
            
        except Exception as e:
            logger.error(f"❌ Erreur rate limiting: {e}")
            # En cas d'erreur, autoriser mais loguer
            return True, {"error": str(e), "fallback": True}
    
    async def _check_specific_limit(self, 
                                  identifier: str, 
                                  request_type: RequestType) -> Tuple[bool, Dict]:
        """Vérification des limites spécifiques au type de requête"""
        limit = self.limits[request_type]
        key = f"rate_limit:specific:{request_type.value}:{identifier}"
        
        current_time = int(time.time())
        window_start = current_time - limit.window
        
        # Pipeline Redis pour atomicité
        pipe = self.redis.pipeline()
        
        # Supprimer les entrées expirées
        pipe.zremrangebyscore(key, 0, window_start)
        
        # Compter les requêtes dans la fenêtre
        pipe.zcard(key)
        
        # Ajouter la requête actuelle
        pipe.zadd(key, {str(current_time): current_time})
        
        # Définir l'expiration
        pipe.expire(key, limit.window)
        
        results = await pipe.execute()
        current_count = results[1]
        
        is_allowed = current_count < limit.requests
        
        return is_allowed, {
            "limit": limit.requests,
            "window": limit.window,
            "current": current_count,
            "remaining": max(0, limit.requests - current_count)
        }
    
    async def _check_global_limits(self, client_ip: str) -> Tuple[bool, Dict]:
        """Vérification des limites globales par IP"""
        limits_to_check = [
            ("minute", 60, self.global_limits["requests_per_minute"]),
            ("hour", 3600, self.global_limits["requests_per_hour"]),
            ("day", 86400, self.global_limits["requests_per_day"])
        ]
        
        for period, seconds, max_requests in limits_to_check:
            key = f"rate_limit:global:{period}:{client_ip}"
            current_time = int(time.time())
            window_start = current_time - seconds
            
            pipe = self.redis.pipeline()
            pipe.zremrangebyscore(key, 0, window_start)
            pipe.zcard(key)
            pipe.zadd(key, {str(current_time): current_time})
            pipe.expire(key, seconds)
            
            results = await pipe.execute()
            current_count = results[1]
            
            if current_count >= max_requests:
                return False, {
                    "period": period,
                    "limit": max_requests,
                    "current": current_count,
                    "reason": f"Global {period} limit exceeded"
                }
        
        return True, {"status": "within_global_limits"}
    
    async def _check_burst_protection(self, 
                                    identifier: str, 
                                    request_type: RequestType) -> Tuple[bool, Dict]:
        """Protection contre les rafales excessives"""
        limit = self.limits[request_type]
        key = f"rate_limit:burst:{request_type.value}:{identifier}"
        
        # Fenêtre de burst (10 secondes)
        burst_window = 10
        current_time = int(time.time())
        window_start = current_time - burst_window
        
        pipe = self.redis.pipeline()
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zcard(key)
        pipe.zadd(key, {str(current_time): current_time})
        pipe.expire(key, burst_window)
        
        results = await pipe.execute()
        burst_count = results[1]
        
        is_allowed = burst_count <= limit.burst
        
        return is_allowed, {
            "burst_limit": limit.burst,
            "burst_window": burst_window,
            "current_burst": burst_count,
            "allowed": is_allowed
        }
    
    def _get_client_ip(self, request: Request) -> str:
        """
        Extraction sécurisée de l'IP client
        Gère les proxies et load balancers
        """
        # Vérifier les headers de proxy
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Prendre la première IP (client original)
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # IP directe
        return request.client.host if request.client else "unknown"
    
    async def get_usage_stats(self, identifier: str) -> Dict[str, any]:
        """Statistiques d'usage pour un identifier"""
        stats = {}
        
        for request_type in RequestType:
            key = f"rate_limit:specific:{request_type.value}:{identifier}"
            count = await self.redis.zcard(key)
            limit = self.limits[request_type]
            
            stats[request_type.value] = {
                "current_usage": count,
                "limit": limit.requests,
                "window_seconds": limit.window,
                "remaining": max(0, limit.requests - count)
            }
        
        return stats
    
    async def reset_limits(self, identifier: str) -> bool:
        """Reset des limites pour un identifier (admin)"""
        try:
            keys_to_delete = []
            
            # Construire les clés à supprimer
            for request_type in RequestType:
                keys_to_delete.extend([
                    f"rate_limit:specific:{request_type.value}:{identifier}",
                    f"rate_limit:burst:{request_type.value}:{identifier}"
                ])
            
            # Supprimer en batch
            if keys_to_delete:
                await self.redis.delete(*keys_to_delete)
                logger.info(f"🔄 Limites reset pour {identifier}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Erreur reset limites: {e}")
            return False

# Intégration FastAPI Cache2
class CacheManager:
    """Gestionnaire de cache avec FastAPI-Cache2 et Redis"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        logger.info("💾 CacheManager initialisé avec Redis")
    
    async def get_cached_response(self, cache_key: str) -> Optional[Dict]:
        """Récupération d'une réponse en cache"""
        try:
            cached = await self.redis.get(cache_key)
            if cached:
                return json.loads(cached)
            return None
        except Exception as e:
            logger.error(f"❌ Erreur get cache: {e}")
            return None
    
    async def cache_response(self, 
                           cache_key: str, 
                           response: Dict, 
                           ttl: int = 3600) -> bool:
        """Mise en cache d'une réponse"""
        try:
            await self.redis.setex(
                cache_key, 
                ttl, 
                json.dumps(response)
            )
            return True
        except Exception as e:
            logger.error(f"❌ Erreur set cache: {e}")
            return False
    
    def generate_cache_key(self, 
                          request_type: str, 
                          content: str, 
                          params: Dict = None) -> str:
        """Génération de clé de cache"""
        # Créer un hash unique basé sur le contenu
        content_hash = hashlib.md5(content.encode()).hexdigest()
        params_hash = hashlib.md5(str(sorted(params.items()) if params else "").encode()).hexdigest()
        
        return f"cache:{request_type}:{content_hash}:{params_hash}"
