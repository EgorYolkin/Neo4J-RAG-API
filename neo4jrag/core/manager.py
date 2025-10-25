"""
Центральный менеджер RAG системы
"""
from typing import Optional
import logging

from neo4jrag.config import Config
from neo4jrag.services.neo4j.neo4j_connector import Neo4jConnector
from neo4jrag.services.neo4j.graph_builder import GraphBuilder
from neo4jrag.services.neo4j.vector_store import VectorStore
from neo4jrag.services.ollama.ollama_loader import OllamaLoader
from neo4jrag.services.ollama.rag_pipeline import RAGPipeline
from neo4jrag.core.exceptions import RAGInitializationError

logger = logging.getLogger(__name__)


class RAGManager:
    """Управление жизненным циклом RAG компонентов"""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Компоненты
        self.neo4j_connector: Optional[Neo4jConnector] = None
        self.ollama_loader: Optional[OllamaLoader] = None
        self.graph_builder: Optional[GraphBuilder] = None
        self.vector_store: Optional[VectorStore] = None
        self.rag_pipeline: Optional[RAGPipeline] = None
        
        self._initialized = False
    
    async def initialize(self) -> None:
        """Инициализация всех компонентов"""
        try:
            logger.info("Initializing RAG Manager...")
            
            # Neo4j
            self.neo4j_connector = Neo4jConnector(
                uri=self.config.neo4j.uri,
                username=self.config.neo4j.username,
                password=self.config.neo4j.password,
                database=self.config.neo4j.database
            )
            self.neo4j_connector.connect()
            
            # Ollama
            self.ollama_loader = OllamaLoader(
                base_url=self.config.ollama.base_url,
                model=self.config.ollama.model,
                embedding_model=self.config.ollama.embedding_model,
                temperature=self.config.ollama.temperature
            )
            self.ollama_loader.load_llm()
            self.ollama_loader.load_embeddings()
            
            # Graph Builder
            self.graph_builder = GraphBuilder(
                connector=self.neo4j_connector,
                chunk_size=self.config.rag.chunk_size,
                chunk_overlap=self.config.rag.chunk_overlap
            )
            self.graph_builder.setup_schema()
            
            # Vector Store
            self.vector_store = VectorStore(
                connector=self.neo4j_connector,
                ollama=self.ollama_loader,
                index_name=self.config.rag.vector_index_name,
                dimensions=self.config.rag.embedding_dimension
            )
            self.vector_store.create_vector_index()
            
            # RAG Pipeline
            self.rag_pipeline = RAGPipeline(
                vector_store=self.vector_store,
                ollama=self.ollama_loader
            )
            
            self._initialized = True
            logger.info("✓ RAG Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG Manager: {e}")
            raise RAGInitializationError(f"Initialization failed: {e}")
    
    async def cleanup(self) -> None:
        """Очистка ресурсов"""
        logger.info("Cleaning up RAG Manager...")
        
        if self.neo4j_connector:
            self.neo4j_connector.close()
        
        self._initialized = False
        logger.info("✓ RAG Manager cleaned up")
    
    @property
    def is_initialized(self) -> bool:
        """Проверка инициализации"""
        return self._initialized
