"""
Health check endpoints
"""
from fastapi import APIRouter, Depends, status
from typing import Dict

from neo4jrag.domain.schemas.response import HealthResponse
from neo4jrag.services.neo4j.neo4j_connector import Neo4jConnector
from neo4jrag.services.ollama.ollama_loader import OllamaLoader
from neo4jrag.services.cache.semantic_cache import SemanticCache
from neo4jrag.api.deps import (
    get_neo4j_connector,
    get_ollama_loader,
    get_semantic_cache,
    get_components
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check(
    neo4j_connector: Neo4jConnector = Depends(get_neo4j_connector),
    ollama_loader: OllamaLoader = Depends(get_ollama_loader),
    semantic_cache: SemanticCache = Depends(get_semantic_cache)
) -> HealthResponse:
    """
    Проверка здоровья системы
    
    Проверяет:
    - Доступность Neo4j
    - Доступность Ollama
    - Доступность Redis
    - Статус инициализации компонентов
    """
    neo4j_healthy = False
    ollama_healthy = False
    redis_healthy = False
    
    # Проверка Neo4j
    try:
        neo4j_connector.execute_query("RETURN 1")
        neo4j_healthy = True
    except Exception as e:
        neo4j_healthy = False
    
    # Проверка Ollama
    try:
        test_embedding = ollama_loader.embed_text("test")
        ollama_healthy = len(test_embedding) > 0
    except Exception as e:
        ollama_healthy = False
    
    # Проверка Redis
    try:
        semantic_cache.redis_client.ping()
        redis_healthy = True
    except Exception as e:
        redis_healthy = False
    
    overall_status = "healthy" if (neo4j_healthy and ollama_healthy and redis_healthy) else "unhealthy"
    
    return HealthResponse(
        status=overall_status,
        components={
            "neo4j": "healthy" if neo4j_healthy else "unhealthy",
            "ollama": "healthy" if ollama_healthy else "unhealthy",
            "redis": "healthy" if redis_healthy else "unhealthy"
        },
        version="1.0.0"
    )


@router.get("/ready")
async def readiness_check(
    components: Dict = Depends(get_components)
) -> Dict:
    """Проверка готовности к обработке запросов"""
    all_ready = all([
        components.get("neo4j_connector"),
        components.get("ollama_loader"),
        components.get("rag_pipeline"),
        components.get("semantic_cache")
    ])
    
    return {
        "ready": all_ready,
        "message": "System is ready" if all_ready else "System is initializing"
    }


@router.get("/live")
async def liveness_check() -> Dict:
    """Проверка живучести приложения (для Kubernetes)"""
    return {"status": "alive"}
