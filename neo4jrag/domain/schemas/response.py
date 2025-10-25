"""
Response Pydantic schemas
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class SourceInfo(BaseModel):
    """Информация об источнике"""
    text: str
    score: float
    doc_title: str


class QueryResponse(BaseModel):
    """Ответ на RAG запрос"""
    question: str
    answer: str
    sources: List[SourceInfo]
    search_type: str
    processing_steps: List[str]
    cached: bool = False
    cache_similarity: Optional[float] = None
    original_query: Optional[str] = None 

    
    class Config:
        schema_extra = {
            "example": {
                "question": "Что такое машинное обучение?",
                "answer": "Машинное обучение — это...",
                "sources": [
                    {
                        "text": "Машинное обучение...",
                        "score": 0.95,
                        "doc_title": "Введение в ML"
                    }
                ],
                "search_type": "hybrid",
                "processing_steps": ["Маршрут: Гибридный поиск", "Найдено 3 результата"]
            }
        }


class DocumentResponse(BaseModel):
    """Ответ с информацией о документе"""
    id: str
    title: str
    preview: str
    chunks_count: int
    created_at: Optional[str] = None


class StatsResponse(BaseModel):
    """Статистика графа"""
    nodes: Dict[str, int]
    relationships: Dict[str, int]
    total_documents: int
    total_chunks: int

class HealthResponse(BaseModel):
    """Ответ health check"""
    status: str
    components: Dict[str, str]
    version: str


class DocumentListResponse(BaseModel):
    """Список документов"""
    documents: List[DocumentResponse]
    total: int
    skip: int
    limit: int


class DocumentDeleteResponse(BaseModel):
    """Ответ на удаление"""
    id: str
    deleted: bool
    message: str


class BatchQueryResponse(BaseModel):
    """Результат пакетного запроса"""
    results: List[QueryResponse]
    total: int


class GraphSchemaResponse(BaseModel):
    """Схема графа"""
    node_labels: List[str]
    relationship_types: List[str]
    constraints: List[Dict[str, str]]
    indexes: List[Dict[str, str]]
