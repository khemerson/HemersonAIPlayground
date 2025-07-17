# app/utils/rate_limiter.py
"""
Système de limitation optimisé pour Chainlit
Protection avancée contre les abus
"""

import time
import json
from typing import Dict, Tuple, Optional
from datetime import datetime
import aioredis
from loguru import logger
from .config import settings

class ChainlitRateLimiter:
    """
    Rate limiter optimisé pour Chainlit
    
    Fonctionnalités :
    - Limitations par utilisateur et par type de requête
    - Système de niveaux utilisateur adaptatif
    - Métriques en temps réel
    - Intégration Redis pour performance
    """
    
    def __init__(self):
        """Initialise le rate limiter"""
        self.redis_client = None
        self.local_cache = {}
        self.init_redis()
        
        # Niveaux utilisateur avec leurs limites
        self.user_tiers = {
            'new_user': {
                'rpm': 10,
                'rph': 40,
                'description': 'Nouvel utilisateur',
                'upgrade_threshold': 5
            },
            'regular_user': {
                'rpm': settings.MAX_REQUESTS_PER_MINUTE,
                'rph': settings.MAX_REQUESTS_PER_HOUR,
                'description': 'Utilisateur régulier',
                'upgrade_threshold': 20
            },
            'power_user': {
                'rpm': settings.MAX_REQUESTS_PER_MINUTE + 5,
                'rph': settings.MAX_REQUESTS_PER_HOUR + 50,
                'description': 'Utilisateur avancé',
                'upgrade_threshold': 50
            },
            'suspicious': {
                'rpm': 3,
                'rph': 10,
                'description': 'Activité suspecte',
                'upgrade_threshold': 0
            }
        }
        
        logger.info("✅ Rate limiter Chainlit initialisé")
    
    async def init_redis(self):
        """Initialise la connexion Redis de manière asynchrone"""
        try:
            self.redis_client = await aioredis.from_url(
                f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}",
                encoding="utf-8",
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("✅ Connexion Redis rate limiter établie")
        except Exception as e:
            logger.warning(f"⚠️ Redis non disponible, utilisation du cache local: {e}")
            self.redis_client = None
    
    def determine_user_tier(self, user_id: str, session_data: Dict) -> str:
        """
        Détermine le niveau utilisateur basé sur son comportement
        
        Args:
            user_id: Identifiant unique de l'utilisateur
            session_data: Données de session
            
        Returns:
            str: Niveau utilisateur
        """
        # Vérifier l'historique d'usage
        total_requests = session_data.get('total_requests', 0)
        session_duration = session_data.get('session_duration', 0)
        
        # Détection d'activité suspecte
        if self._detect_suspicious_activity(user_id, session_data):
            return 'suspicious'
        
        # Déterminer le niveau selon l'historique
        if total_requests == 0:
            return 'new_user'
        elif total_requests < self.user_tiers['regular_user']['upgrade_threshold']:
            return 'regular_user'
        else:
            return 'power_user'
    
    def _detect_suspicious_activity(self, user_id: str, session_data: Dict) -> bool:
        """
        Détecte les comportements suspects
        
        Args:
            user_id: Identifiant utilisateur
            session_data: Données de session
            
        Returns:
            bool: True si activité suspecte
        """
        # Patterns suspects à détecter
        request_frequency = session_data.get('request_frequency', 0)
        error_rate = session_data.get('error_rate', 0)
        
        # Trop de requêtes trop rapidement
        if request_frequency > 30:  # Plus de 30 requêtes en peu de temps
            return True
        
        # Taux d'erreur élevé (potentiel bot)
        if error_rate > 0.5:  # Plus de 50% d'erreurs
            return True
        
        return False
    
    async def is_allowed(self, user_id: str, request_type: str = "general", session_data: Dict = None) -> Tuple[bool, Dict]:
        """
        Vérifie si une requête est autorisée
        
        Args:
            user_id: Identifiant unique de l'utilisateur
            request_type: Type de requête (chat, image, audio, etc.)
            session_data: Données de session optionnelles
            
        Returns:
            Tuple[bool, Dict]: (Autorisé, Informations détaillées)
        """
        try:
            # Déterminer le niveau utilisateur
            if session_data is None:
                session_data = {}
            
            user_tier = self.determine_user_tier(user_id, session_data)
            tier_limits = self.user_tiers[user_tier]
            
            # Clés pour le tracking
            now = int(time.time())
            minute_key = f"rate_limit:{user_id}:{request_type}:minute:{now // 60}"
            hour_key = f"rate_limit:{user_id}:{request_type}:hour:{now // 3600}"
            
            # Récupérer les compteurs
            if self.redis_client:
                minute_count = await self._redis_incr(minute_key, 60)
                hour_count = await self._redis_incr(hour_key, 3600)
            else:
                minute_count = self._local_incr(minute_key, 60)
                hour_count = self._local_incr(hour_key, 3600)
            
            # Vérifier les limites
            rpm_exceeded = minute_count > tier_limits['rpm']
            rph_exceeded = hour_count > tier_limits['rph']
            
            is_allowed = not (rpm_exceeded or rph_exceeded)
            
            # Informations détaillées
            limit_info = {
                "is_allowed": is_allowed,
                "user_tier": user_tier,
                "tier_description": tier_limits['description'],
                "request_type": request_type,
                "rpm_used": minute_count,
                "rpm_limit": tier_limits['rpm'],
                "rpm_remaining": max(0, tier_limits['rpm'] - minute_count),
                "rph_used": hour_count,
                "rph_limit": tier_limits['rph'],
                "rph_remaining": max(0, tier_limits['rph'] - hour_count),
                "limit_type": "minute" if rpm_exceeded else ("hour" if rph_exceeded else None),
                "next_reset": {
                    "minute": (now // 60 + 1) * 60,
                    "hour": (now // 3600 + 1) * 3600
                }
            }
            
            # Logging pour le monitoring
            if not is_allowed:
                logger.warning(f"Rate limit atteint pour {user_id} ({user_tier}): {request_type}")
            
            return is_allowed, limit_info
            
        except Exception as e:
            logger.error(f"Erreur rate limiter: {e}")
            # En cas d'erreur, on autorise (mode dégradé)
            return True, {
                "is_allowed": True,
                "error": str(e),
                "user_tier": "unknown"
            }
    
    async def _redis_incr(self, key: str, ttl: int) -> int:
        """Incrémente un compteur Redis avec expiration"""
        try:
            pipe = self.redis_client.pipeline()
            pipe.incr(key)
            pipe.expire(key, ttl)
            results = await pipe.execute()
            return results[0]
        except Exception as e:
            logger.error(f"Erreur Redis incr: {e}")
            return 0
    
    def _local_incr(self, key: str, ttl: int) -> int:
        """Incrémente un compteur local avec expiration"""
        now = time.time()
        
        # Nettoyer les clés expirées
        expired_keys = [k for k, v in self.local_cache.items() if v['expires'] < now]
        for k in expired_keys:
            del self.local_cache[k]
        
        # Incrémenter ou créer
        if key not in self.local_cache:
            self.local_cache[key] = {"count": 0, "expires": now + ttl}
        
        self.local_cache[key]["count"] += 1
        return self.local_cache[key]["count"]
    
    async def record_usage(self, user_id: str, request_type: str, success: bool = True, processing_time: float = 0):
        """
        Enregistre l'utilisation pour les statistiques
        
        Args:
            user_id: Identifiant utilisateur
            request_type: Type de requête
            success: Si la requête a réussi
            processing_time: Temps de traitement
        """
        try:
            usage_data = {
                'timestamp': datetime.now().isoformat(),
                'user_id': user_id,
                'request_type': request_type,
                'success': success,
                'processing_time': processing_time
            }
            
            # Enregistrer dans Redis avec expiration
            if self.redis_client:
                usage_key = f"usage:{user_id}:{int(time.time())}"
                await self.redis_client.setex(usage_key, 86400, json.dumps(usage_data))
            
        except Exception as e:
            logger.error(f"Erreur enregistrement usage: {e}")
    
    async def get_user_stats(self, user_id: str) -> Dict:
        """
        Récupère les statistiques détaillées d'un utilisateur
        
        Args:
            user_id: Identifiant utilisateur
            
        Returns:
            Dict: Statistiques complètes
        """
        try:
            # Récupérer les limites actuelles
            is_allowed, limit_info = await self.is_allowed(user_id)
            
            # Statistiques d'usage si Redis disponible
            usage_stats = {}
            if self.redis_client:
                usage_keys = await self.redis_client.keys(f"usage:{user_id}:*")
                usage_stats = {
                    'total_requests_24h': len(usage_keys),
                    'active_periods': len(set(key.split(':')[2] for key in usage_keys))
                }
            
            return {
                "user_id": user_id,
                "current_limits": limit_info,
                "usage_stats": usage_stats,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erreur récupération stats: {e}")
            return {"error": str(e)}
    
    async def cleanup_expired_entries(self):
        """Nettoie les entrées expirées (tâche de maintenance)"""
        try:
            if self.redis_client:
                # Redis gère l'expiration automatiquement
                pass
            else:
                # Nettoyer le cache local
                now = time.time()
                expired_keys = [k for k, v in self.local_cache.items() if v['expires'] < now]
                for k in expired_keys:
                    del self.local_cache[k]
                
                if expired_keys:
                    logger.info(f"Nettoyé {len(expired_keys)} entrées expirées")
                    
        except Exception as e:
            logger.error(f"Erreur nettoyage: {e}")

# Instance globale
rate_limiter = ChainlitRateLimiter()
