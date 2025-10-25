"""
Query endpoints для RAG запросов
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.params import Query as FastAPIQuery
from typing import Dict, List

from neo4jrag.domain.schemas.request import QueryRequest, BatchQueryRequest
from neo4jrag.domain.schemas.response import QueryResponse, BatchQueryResponse, SourceInfo
from neo4jrag.services.cache.semantic_cache import SemanticCache
from neo4jrag.services.ollama.ollama_loader import OllamaLoader
from neo4jrag.services.ollama.rag_pipeline import RAGPipeline
from neo4jrag.services.neo4j.neo4j_connector import Neo4jConnector
from neo4jrag.services.neo4j.vector_store import VectorStore
from neo4jrag.api.deps import (
    get_ollama_loader,
    get_rag_pipeline,
    get_semantic_cache,
    get_neo4j_connector,
    get_vector_store
)

router = APIRouter()


@router.post("/", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def query_rag(
    request: QueryRequest,
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline),
    ollama_loader: OllamaLoader = Depends(get_ollama_loader),
    semantic_cache: SemanticCache = Depends(get_semantic_cache)
) -> QueryResponse:
    """
    Выполнение RAG запроса с семантическим кэшированием
    
    - Сначала проверяет кэш на наличие похожих запросов
    - Если найден (similarity > threshold) → возвращает кэшированный ответ
    - Иначе выполняет RAG и кэширует результат
    """
    try:
        # 1. Генерируем эмбеддинг запроса
        query_embedding = ollama_loader.embed_text(request.question)
        
        # 2. Проверяем кэш
        cached_result = semantic_cache.get(request.question, query_embedding)
        
        if cached_result:
            # Cache HIT - возвращаем кэшированный результат
            return QueryResponse(
                question=request.question,
                answer=cached_result["answer"],
                sources=[
                    SourceInfo(
                        text=src.get("text", ""),
                        score=src.get("score", 0.0),
                        doc_title=src.get("doc_title", "Unknown")
                    )
                    for src in cached_result.get("sources", [])
                ],
                search_type=cached_result["search_type"],
                processing_steps=cached_result["processing_steps"] + ["✓ Retrieved from cache"],
                cached=True,
                cache_similarity=cached_result.get("similarity", 0.0),
                original_query=cached_result.get("original_query")
            )
        
        # 3. Cache MISS - выполняем RAG запрос
        result = rag_pipeline.ask(
            question=request.question,
            verbose=False
        )
        
        # 4. Формируем источники
        sources = [
            {
                "text": ctx.get("text", ""),
                "score": ctx.get("score", 0.0),
                "doc_title": ctx.get("doc_title", "Unknown")
            }
            for ctx in result.get("context", [])
        ]
        
        # 5. Кэшируем результат
        semantic_cache.set(
            query=request.question,
            embedding=query_embedding,
            answer=result["answer"],
            sources=sources,
            search_type=result.get("search_type", "hybrid"),
            processing_steps=result.get("steps", [])
        )
        
        # 6. Возвращаем результат
        return QueryResponse(
            question=request.question,
            answer=result["answer"],
            sources=[
                SourceInfo(
                    text=src["text"],
                    score=src["score"],
                    doc_title=src["doc_title"]
                )
                for src in sources
            ],
            search_type=result.get("search_type", "hybrid"),
            processing_steps=result.get("steps", []),
            cached=False
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}"
        )


@router.post("/batch", response_model=BatchQueryResponse)
async def batch_query_rag(
    request: BatchQueryRequest,
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
) -> BatchQueryResponse:
    """Пакетная обработка нескольких вопросов"""
    try:
        results = []
        
        for question in request.questions:
            result = rag_pipeline.ask(
                question=question,
                verbose=False
            )
            
            results.append(QueryResponse(
                question=question,
                answer=result["answer"],
                sources=[
                    SourceInfo(
                        text=ctx.get("text", ""),
                        score=ctx.get("score", 0.0),
                        doc_title=ctx.get("doc_title", "Unknown")
                    )
                    for ctx in result.get("context", [])
                ],
                search_type=result.get("search_type", "hybrid"),
                processing_steps=result.get("steps", []),
                cached=False
            ))
        
        return BatchQueryResponse(
            results=results,
            total=len(results)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch query failed: {str(e)}"
        )


@router.get("/similar", response_model=List[Dict])
async def find_similar_chunks(
    text: str = FastAPIQuery(..., min_length=1, max_length=1000),
    k: int = FastAPIQuery(5, ge=1, le=20),
    vector_store: VectorStore = Depends(get_vector_store)
) -> List[Dict]:
    """
    Поиск похожих чанков без генерации ответа
    
    Полезно для:
    - Дебага векторного поиска
    - Получения релевантных отрывков без LLM
    """
    try:
        results = vector_store.similarity_search(text, k=k)
        
        return [
            {
                "chunk_id": result["chunk_id"],
                "text": result["text"],
                "score": result["score"]
            }
            for result in results
        ]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Similarity search failed: {str(e)}"
        )


@router.get("/context/{chunk_id}", response_model=Dict)
async def get_chunk_context(
    chunk_id: str,
    neo4j_connector: Neo4jConnector = Depends(get_neo4j_connector)
) -> Dict:
    """Получение контекста вокруг конкретного чанка (prev/next)"""
    try:
        query = """
        MATCH (c:Chunk {id: $chunk_id})
        OPTIONAL MATCH (prev:Chunk)-[:NEXT]->(c)
        OPTIONAL MATCH (c)-[:NEXT]->(next:Chunk)
        OPTIONAL MATCH (d:Document)-[:HAS_CHUNK]->(c)
        RETURN 
            c.text as current,
            c.position as position,
            prev.text as previous,
            next.text as next,
            d.title as document_title,
            d.id as document_id
        LIMIT 1
        """
        
        result = neo4j_connector.execute_query(query, {"chunk_id": chunk_id})
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chunk {chunk_id} not found"
            )
        
        return result[0]
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get chunk context: {str(e)}"
        )
