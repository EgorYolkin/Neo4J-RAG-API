"""
Custom exceptions
"""


class RAGException(Exception):
    """Base RAG exception"""
    pass


class RAGInitializationError(RAGException):
    """Initialization failed"""
    pass


class Neo4jConnectionError(RAGException):
    """Neo4j connection failed"""
    pass


class OllamaConnectionError(RAGException):
    """Ollama connection failed"""
    pass


class DocumentNotFoundError(RAGException):
    """Document not found"""
    pass


class QueryProcessingError(RAGException):
    """Query processing failed"""
    pass
