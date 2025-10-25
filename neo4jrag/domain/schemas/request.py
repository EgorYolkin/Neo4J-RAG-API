"""
Request Pydantic schemas
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List, Required


class QueryRequest(BaseModel):
    """RAG запрос"""
    user_id: int = Field(..., description="ID пользователя")
    question: str = Field(..., min_length=1, max_length=1000, description="Вопрос пользователя")
    top_k: Optional[int] = Field(3, ge=1, le=10, description="Количество результатов")
    search_type: Optional[str] = Field("hybrid", pattern="^(vector|hybrid)$", description="Тип поиска")
    
    class Config:
        schema_extra = {
            "example": {
                "question": "Что такое машинное обучение?",
                "top_k": 3,
                "search_type": "hybrid"
            }
        }


class DocumentCreateRequest(BaseModel):
    """Создание документа"""
    user_id: int = Field(..., description="ID пользователя")
    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=10)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    class Config:
        schema_extra = {
            "example": {
                "title": "Введение в AI",
                "content": "Искусственный интеллект...",
                "metadata": {"author": "John Doe", "date": "2025-10-25"}
            }
        }

class DocumentUpdateRequest(BaseModel):
    """Обновление документа"""
    user_id: int = Field(..., description="ID пользователя")
    title: Optional[str] = Field(None, min_length=1, max_length=500)
    content: Optional[str] = Field(None, min_length=10)
    metadata: Optional[Dict[str, Any]] = None


class BatchQueryRequest(BaseModel):
    """Пакетный запрос"""
    user_id: int = Field(..., description="ID пользователя")
    questions: List[str] = Field(..., min_items=1, max_items=10)
    top_k: Optional[int] = Field(3, ge=1, le=10)
    search_type: Optional[str] = Field("hybrid", pattern="^(vector|hybrid)$")
