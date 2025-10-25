"""
Cache management endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict

from ....services.cache.semantic_cache import SemanticCache
from ...deps import get_semantic_cache

router = APIRouter()


@router.get("/stats", response_model=Dict)
async def get_cache_stats(
    semantic_cache: SemanticCache = Depends(get_semantic_cache)
) -> Dict:
    """Получение статистики кэша"""
    try:
        stats = semantic_cache.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cache stats: {str(e)}"
        )


@router.delete("/clear", response_model=Dict)
async def clear_cache(
    semantic_cache: SemanticCache = Depends(get_semantic_cache)
) -> Dict:
    """Очистка всего кэша"""
    try:
        success = semantic_cache.clear()
        
        if success:
            return {
                "success": True,
                "message": "Cache cleared successfully"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to clear cache"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear cache: {str(e)}"
        )


@router.get("/health", response_model=Dict)
async def cache_health_check(
    semantic_cache: SemanticCache = Depends(get_semantic_cache)
) -> Dict:
    """Проверка работоспособности Redis"""
    try:
        # Тестовая операция
        semantic_cache.redis_client.ping()
        
        return {
            "status": "healthy",
            "message": "Redis is operational"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Redis unhealthy: {str(e)}"
        )
