"""
Document management endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import List, Optional
import uuid
from datetime import datetime

from ....domain.schemas.request import DocumentCreateRequest
from ....domain.schemas.response import (
    DocumentResponse,
    DocumentListResponse,
    DocumentDeleteResponse
)
from ....services.neo4j.neo4j_connector import Neo4jConnector
from ....services.neo4j.graph_builder import GraphBuilder
from ....services.neo4j.vector_store import VectorStore
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
    """Создание нового документа"""
    try:
        doc_id = f"doc_{uuid.uuid4().hex[:8]}"
        
        # Добавление документа в граф
        chunks_count = graph_builder.add_document(
            doc_id=doc_id,
            title=request.title,
            content=request.content,
            metadata=request.metadata
        )
        
        # Генерация эмбеддингов для новых чанков
        vector_store.generate_embeddings()
        
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
    skip: int = Query(0, ge=0, description="Пропустить N документов"),
    limit: int = Query(10, ge=1, le=100, description="Максимум документов"),
    neo4j_connector: Neo4jConnector = Depends(get_neo4j_connector)
) -> DocumentListResponse:
    """Получение списка всех документов"""
    try:
        query = """
        MATCH (d:Document)
        OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
        WITH d, count(c) as chunks_count
        RETURN d.id as id,
               d.title as title,
               d.preview as preview,
               chunks_count
        ORDER BY d.title
        SKIP $skip
        LIMIT $limit
        """
        
        results = neo4j_connector.execute_query(query, {
            "skip": skip,
            "limit": limit
        })
        
        # Общее количество
        count_query = "MATCH (d:Document) RETURN count(d) as total"
        total_result = neo4j_connector.execute_query(count_query)
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


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    neo4j_connector: Neo4jConnector = Depends(get_neo4j_connector)
) -> DocumentResponse:
    """Получение информации о конкретном документе"""
    try:
        query = """
        MATCH (d:Document {id: $doc_id})
        OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
        WITH d, count(c) as chunks_count
        RETURN d.id as id,
               d.title as title,
               d.preview as preview,
               chunks_count
        LIMIT 1
        """
        
        result = neo4j_connector.execute_query(query, {"doc_id": document_id})
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )
        
        doc = result[0]
        return DocumentResponse(
            id=doc["id"],
            title=doc["title"],
            preview=doc["preview"],
            chunks_count=doc["chunks_count"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get document: {str(e)}"
        )


@router.delete("/{document_id}", response_model=DocumentDeleteResponse)
async def delete_document(
    document_id: str,
    neo4j_connector: Neo4jConnector = Depends(get_neo4j_connector)
) -> DocumentDeleteResponse:
    """Удаление документа и всех его чанков"""
    try:
        # Проверяем существование
        check_query = "MATCH (d:Document {id: $doc_id}) RETURN d LIMIT 1"
        exists = neo4j_connector.execute_query(check_query, {"doc_id": document_id})
        
        if not exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )
        
        # Удаляем документ и связанные чанки
        delete_query = """
        MATCH (d:Document {id: $doc_id})
        OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
        DETACH DELETE d, c
        RETURN count(c) as deleted_chunks
        """
        
        result = neo4j_connector.execute_write(delete_query, {"doc_id": document_id})
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
