"""
Custom exceptions
"""

class Neo4jException(Exception):
    """Base Neo4j exception"""
    pass

class ServiceUnavailable(Neo4jException):
    """Neo4j service is unavailable"""
    pass


class AuthError(Neo4jException):
    """Authentication failed"""
    pass

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
