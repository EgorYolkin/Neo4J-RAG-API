"""
API v1 Router
"""
from fastapi import APIRouter

from neo4jrag.api.v1.endpoints import health, query, documents, stats, cache

api_router = APIRouter()

# Подключение endpoint модулей
api_router.include_router(health.router, tags=["Health"])
api_router.include_router(query.router, prefix="/query", tags=["Query"])
api_router.include_router(documents.router, prefix="/documents", tags=["Documents"])
api_router.include_router(stats.router, prefix="/stats", tags=["Statistics"])
api_router.include_router(cache.router, prefix="/cache", tags=["Cache"])