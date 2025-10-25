"""
Query endpoints с user isolation
"""
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, List, Optional

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
    get_vector_store,
)

router = APIRouter()


@router.post("/", response_model=QueryResponse)
async def query_rag(
    request: QueryRequest,
    vector_store: VectorStore = Depends(get_vector_store),
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline),
    ollama_loader: OllamaLoader = Depends(get_ollama_loader),
    semantic_cache: SemanticCache = Depends(get_semantic_cache)
) -> QueryResponse:
    """
    RAG запрос с семантическим кэшированием
    
    - Если пользователь аутентифицирован → поиск только в его документах
    - Если нет → поиск по всем документам (если разрешено в конфиге)
    """
    try:
        user_id = request.user_id
        
        # 1. Генерируем эмбеддинг
        query_embedding = ollama_loader.embed_text(request.question)
        
        # 2. Проверяем кэш (с учётом user_id)
        cache_key = f"{user_id}:{request.question}" if user_id else request.question
        cached_result = semantic_cache.get(cache_key, query_embedding)
        
        if cached_result:
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
                cache_similarity=cached_result.get("similarity", 0.0)
            )
        
        # 3. Выполняем поиск с фильтрацией по user_id
        search_results = vector_store.hybrid_search(
            query=request.question,
            k=request.top_k or 3,
            user_id=user_id  # ✅ Фильтруем по пользователю
        )
        
        # 4. Генерируем ответ
        # (Здесь нужно обновить rag_pipeline чтобы принимал готовые результаты)
        # Для простоты используем поиск напрямую
        
        if not search_results:
            return QueryResponse(
                question=request.question,
                answer="К сожалению, я не нашёл релевантной информации в ваших документах.",
                sources=[],
                search_type="hybrid",
                processing_steps=["Поиск не дал результатов"],
                cached=False
            )
        
        # Формируем контекст для LLM
        context_text = "\n\n".join([
            f"Источник {idx+1} (релевантность: {res['score']:.2f}):\n{res['text']}"
            for idx, res in enumerate(search_results)
        ])
        
        # Генерируем ответ
        prompt = f"""На основе предоставленного контекста ответь на вопрос пользователя.

Контекст:
{context_text}

Вопрос: {request.question}

Ответ:"""
        
        answer = ollama_loader.llm.invoke(prompt)
        
        # 5. Кэшируем результат
        sources_for_cache = [
            {
                "text": res["text"],
                "score": res["score"],
                "doc_title": res.get("doc_title", "Unknown")
            }
            for res in search_results
        ]
        
        semantic_cache.set(
            query=cache_key,
            embedding=query_embedding,
            answer=answer,
            sources=sources_for_cache,
            search_type="hybrid",
            processing_steps=["Hybrid search", "LLM generation"]
        )
        
        # 6. Возвращаем результат
        return QueryResponse(
            question=request.question,
            answer=answer,
            sources=[
                SourceInfo(
                    text=src["text"],
                    score=src["score"],
                    doc_title=src["doc_title"]
                )
                for src in sources_for_cache
            ],
            search_type="hybrid",
            processing_steps=["Hybrid search", "LLM generation"],
            cached=False
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}"
        )
