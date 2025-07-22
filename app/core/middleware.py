# app/core/middleware.py
"""
Middleware d'intégration Rate Limiting + Cache
"""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import time

from services.rate_limiter.advanced_limiter import AdvancedRateLimiter, RequestType, CacheManager

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware de rate limiting et cache"""
    
    def __init__(self, app, rate_limiter: AdvancedRateLimiter, cache_manager: CacheManager):
        super().__init__(app)
        self.rate_limiter = rate_limiter
        self.cache_manager = cache_manager
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Déterminer le type de requête
        request_type = self._determine_request_type(request)
        
        # Vérifier les limites
        is_allowed, limit_info = await self.rate_limiter.check_rate_limit(
            request, 
            request_type
        )
        
        if not is_allowed:
            # Retourner erreur 429 avec détails
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "details": limit_info,
                    "retry_after": self._calculate_retry_after(limit_info)
                },
                headers={"Retry-After": str(self._calculate_retry_after(limit_info))}
            )
        
        # Traiter la requête
        response = await call_next(request)
        
        # Ajouter headers de rate limiting
        response.headers["X-RateLimit-Limit"] = str(limit_info.get("specific_limit", {}).get("limit", 0))
        response.headers["X-RateLimit-Remaining"] = str(limit_info.get("specific_limit", {}).get("remaining", 0))
        response.headers["X-RateLimit-Reset"] = str(int(time.time() + 60))
        
        return response
    
    def _determine_request_type(self, request: Request) -> RequestType:
        """Détermine le type de requête basé sur le path"""
        path = request.url.path
        
        if "/chat" in path:
            return RequestType.CHAT
        elif "/audio" in path:
            return RequestType.AUDIO_PIPELINE
        elif "/image-analysis" in path:
            return RequestType.IMAGE_ANALYSIS
        elif "/image-generation" in path:
            return RequestType.IMAGE_GENERATION
        elif "/health" in path:
            return RequestType.HEALTH_CHECK
        else:
            return RequestType.CHAT  # Default
    
    def _calculate_retry_after(self, limit_info: Dict) -> int:
        """Calcule le délai avant retry"""
        if "specific_limit" in limit_info:
            return limit_info["specific_limit"].get("window", 60)
        return 60
