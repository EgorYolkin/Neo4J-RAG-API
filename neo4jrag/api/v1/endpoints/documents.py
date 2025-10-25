"""
Document management with user isolation
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import List, Optional
import uuid
from datetime import datetime

from neo4jrag.domain.schemas.request import DocumentCreateRequest
from neo4jrag.domain.schemas.response import (
    DocumentResponse,
    DocumentListResponse,
    DocumentDeleteResponse
)
from neo4jrag.services.neo4j.neo4j_connector import Neo4jConnector
from neo4jrag.services.neo4j.graph_builder import GraphBuilder
from neo4jrag.services.neo4j.vector_store import VectorStore
from ...deps import (
    get_neo4j_connector,
    get_graph_builder,
    get_vector_store
)

router = APIRouter()


@router.post("/", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def create_document(
    request: DocumentCreateRequest,
    graph_builder: GraphBuilder = Depends(get_graph_builder),
    vector_store: VectorStore = Depends(get_vector_store)
) -> DocumentResponse:
    """Создание нового документа (только для аутентифицированных пользователей)"""
    try:
        doc_id = f"doc_{uuid.uuid4().hex[:8]}"
        user_id = request.user_id

        # Добавление документа с user_id
        chunks_count = graph_builder.add_document(
            doc_id=doc_id,
            title=request.title,
            content=request.content,
            user_id=user_id,  # ✅ Привязываем к пользователю
            metadata=request.metadata
        )
        
        # Генерация эмбеддингов для новых чанков этого пользователя
        vector_store.generate_embeddings(user_id=user_id)
        
        return DocumentResponse(
            id=doc_id,
            title=request.title,
            preview=request.content[:200],
            chunks_count=chunks_count,
            created_at=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create document: {str(e)}"
        )


@router.get("/", response_model=DocumentListResponse)
async def list_documents(
    user_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    neo4j_connector: Neo4jConnector = Depends(get_neo4j_connector)
) -> DocumentListResponse:
    """Получение списка документов пользователя"""
    try:        
        # Получаем ТОЛЬКО документы этого пользователя
        query = """
        MATCH (d:Document {user_id: $user_id})
        OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
        WITH d, count(c) as chunks_count
        RETURN d.id as id,
               d.title as title,
               d.preview as preview,
               chunks_count
        ORDER BY d.created_at DESC
        SKIP $skip
        LIMIT $limit
        """
        
        results = neo4j_connector.execute_query(query, {
            "user_id": user_id,
            "skip": skip,
            "limit": limit
        })
        
        # Общее количество документов пользователя
        count_query = "MATCH (d:Document {user_id: $user_id}) RETURN count(d) as total"
        total_result = neo4j_connector.execute_query(count_query, {"user_id": user_id})
        total = total_result[0]["total"] if total_result else 0
        
        documents = [
            DocumentResponse(
                id=doc["id"],
                title=doc["title"],
                preview=doc["preview"],
                chunks_count=doc["chunks_count"]
            )
            for doc in results
        ]
        
        return DocumentListResponse(
            documents=documents,
            total=total,
            skip=skip,
            limit=limit
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}"
        )


@router.delete("/{document_id}", response_model=DocumentDeleteResponse)
async def delete_document(
    user_id: int,
    document_id: str,
    neo4j_connector: Neo4jConnector = Depends(get_neo4j_connector)
) -> DocumentDeleteResponse:
    """Удаление документа (только своего)"""
    try:
        # Проверяем, что документ принадлежит пользователю
        check_query = """
        MATCH (d:Document {id: $doc_id, user_id: $user_id})
        RETURN d
        LIMIT 1
        """
        exists = neo4j_connector.execute_query(check_query, {
            "doc_id": document_id,
            "user_id": user_id
        })
        
        if not exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found or access denied"
            )
        
        # Удаляем документ
        delete_query = """
        MATCH (d:Document {id: $doc_id, user_id: $user_id})
        OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
        DETACH DELETE d, c
        RETURN count(c) as deleted_chunks
        """
        
        result = neo4j_connector.execute_write(delete_query, {
            "doc_id": document_id,
            "user_id": user_id
        })
        deleted_chunks = result[0]["deleted_chunks"] if result else 0
        
        return DocumentDeleteResponse(
            id=document_id,
            deleted=True,
            message=f"Document and {deleted_chunks} chunks deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )
