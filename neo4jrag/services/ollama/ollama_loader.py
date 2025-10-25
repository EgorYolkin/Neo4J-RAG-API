from langchain_ollama import OllamaLLM, OllamaEmbeddings
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class OllamaLoader:
    """Loader for Ollama LLM and Embedding models."""
    
    def __init__(self, base_url: str, model: str, embedding_model: str, temperature: float = 0.0):
        self.base_url = base_url
        self.model = model
        self.embedding_model = embedding_model
        self.temperature = temperature
        
        self.llm: Optional[OllamaLLM] = None
        self.embeddings: Optional[OllamaEmbeddings] = None
    
    def load_llm(self) -> OllamaLLM:
        """Загрузка LLM модели"""
        try:
            self.llm = OllamaLLM(
                model=self.model,
                base_url=self.base_url,
                temperature=self.temperature
            )
            logger.info(f"✓ Loaded LLM model: {self.model}")
            return self.llm
        except Exception as e:
            logger.error(f"❌ Failed to load LLM: {e}")
            raise
    
    def load_embeddings(self) -> OllamaEmbeddings:
        """Загрузка модели эмбеддингов"""
        try:
            self.embeddings = OllamaEmbeddings(
                model=self.embedding_model,
                base_url=self.base_url
            )
            logger.info(f"✓ Loaded embedding model: {self.embedding_model}")
            return self.embeddings
        except Exception as e:
            logger.error(f"❌ Failed to load embeddings: {e}")
            raise
    
    def embed_text(self, text: str) -> List[float]:
        """Генерация эмбеддинга для текста"""
        if not self.embeddings:
            self.load_embeddings()
        return self.embeddings.embed_query(text)
    
    def generate(self, prompt: str) -> str:
        """Генерация текста"""
        if not self.llm:
            self.load_llm()
        return self.llm.invoke(prompt)
