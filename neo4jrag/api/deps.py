"""
FastAPI Dependencies
"""
from fastapi import Request, HTTPException
from typing import Dict

from ..services.neo4j.neo4j_connector import Neo4jConnector
from ..services.ollama.ollama_loader import OllamaLoader
from ..services.neo4j.graph_builder import GraphBuilder
from ..services.neo4j.vector_store import VectorStore
from ..services.ollama.rag_pipeline import RAGPipeline
from ..services.cache.semantic_cache import SemanticCache


async def get_components(request: Request) -> Dict:
    """Dependency для получения всех компонентов"""
    if not hasattr(request.app.state, "components"):
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return request.app.state.components


async def get_neo4j_connector(request: Request) -> Neo4jConnector:
    """Dependency для получения Neo4j коннектора"""
    if not hasattr(request.app.state, "neo4j_connector"):
        raise HTTPException(status_code=503, detail="Neo4j not initialized")
    
    return request.app.state.neo4j_connector


async def get_ollama_loader(request: Request) -> OllamaLoader:
    """Dependency для получения Ollama загрузчика"""
    if not hasattr(request.app.state, "ollama_loader"):
        raise HTTPException(status_code=503, detail="Ollama not initialized")
    
    return request.app.state.ollama_loader


async def get_graph_builder(request: Request) -> GraphBuilder:
    """Dependency для получения Graph Builder"""
    if not hasattr(request.app.state, "graph_builder"):
        raise HTTPException(status_code=503, detail="Graph Builder not initialized")
    
    return request.app.state.graph_builder


async def get_vector_store(request: Request) -> VectorStore:
    """Dependency для получения Vector Store"""
    if not hasattr(request.app.state, "vector_store"):
        raise HTTPException(status_code=503, detail="Vector Store not initialized")
    
    return request.app.state.vector_store


async def get_rag_pipeline(request: Request) -> RAGPipeline:
    """Dependency для получения RAG Pipeline"""
    if not hasattr(request.app.state, "rag_pipeline"):
        raise HTTPException(status_code=503, detail="RAG Pipeline not initialized")
    
    return request.app.state.rag_pipeline


async def get_semantic_cache(request: Request) -> SemanticCache:
    """Dependency для получения Semantic Cache"""
    if not hasattr(request.app.state, "semantic_cache"):
        raise HTTPException(status_code=503, detail="Semantic Cache not initialized")
    
    return request.app.state.semantic_cache
