"""
FastAPI Application Entry Point
"""
from fastapi import FastAPI
from contextlib import asynccontextmanager
import os

from neo4jrag.api.v1.router import api_router
from neo4jrag.api.v1.middleware.setup_middleware import setup_middleware
from neo4jrag.core.events import (
    startup_event,
    shutdown_event,
    warm_up_event,
    initialize_sample_data
)
from neo4jrag.config import Config
from neo4jrag.utils.logger import setup_logging

# Настройка логирования
setup_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_file=os.getenv("LOG_FILE", "logs/app.log")
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager для startup/shutdown"""
    # ===== STARTUP =====
    config = Config.from_env()
    
    # Инициализация компонентов
    components = await startup_event(config)
    
    # Сохраняем в app state
    app.state.config = config
    app.state.components = components
    app.state.neo4j_connector = components["neo4j_connector"]
    app.state.ollama_loader = components["ollama_loader"]
    app.state.graph_builder = components["graph_builder"]
    app.state.vector_store = components["vector_store"]
    app.state.rag_pipeline = components["rag_pipeline"]
    app.state.semantic_cache = components.get("semantic_cache")  # ✅ Добавлено
    
    # Прогрев системы
    await warm_up_event(components)
    
    # Инициализация примеров данных (только в dev режиме)
    if os.getenv("ENVIRONMENT", "development") == "development":
        await initialize_sample_data(components)
    
    yield
    
    # ===== SHUTDOWN =====
    await shutdown_event(components)


# Создание FastAPI приложения
app = FastAPI(
    title="Neo4j GraphRAG API",
    description="""
    Production-ready GraphRAG system with:
    - LangChain & LangGraph for RAG workflows
    - Neo4j for knowledge graph storage
    - Ollama for local LLM inference
    - Vector similarity search
    - Semantic caching with Redis
    - Hybrid retrieval strategies
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Middleware
setup_middleware(app)

# Роутеры
app.include_router(api_router, prefix="/api/v1")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "name": "Neo4j GraphRAG API",
        "version": "1.0.0",
        "status": "running",
        "features": [
            "Neo4j Knowledge Graph",
            "Semantic Caching (Redis)",
            "Vector Search",
            "Hybrid Retrieval",
            "LangGraph Workflows"
        ],
        "docs": {
            "swagger": "/docs",
            "redoc": "/redoc"
        },
        "endpoints": {
            "health": "/api/v1/health",
            "query": "/api/v1/query",
            "documents": "/api/v1/documents",
            "stats": "/api/v1/stats",
            "cache": "/api/v1/cache"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
