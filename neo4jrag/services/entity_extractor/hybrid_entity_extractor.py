"""
Hybrid Entity Extraction (spaCy fallback to LLM)
"""
from typing import List, Dict, Tuple
import logging

from neo4jrag.services.entity_extractor.fast_entity_extractor import FastEntityExtractor
from neo4jrag.services.entity_extractor.llm_entity_extractor import LLMEntityExtractor
from neo4jrag.services.neo4j.neo4j_connector import Neo4jConnector

logger = logging.getLogger(__name__)


class HybridEntityExtractor:
    """
    Гибридный экстрактор с fallback:
    1. Пробуем spaCy (быстро)
    2. Если не нашло ничего → используем LLM (медленнее, но точнее)
    """
    
    def __init__(self, neo4j_connector: Neo4jConnector, language: str = "ru"):
        self.connector = neo4j_connector
        
        # Пробуем инициализировать spaCy
        self.use_spacy = False
        try:
            self.spacy_extractor = FastEntityExtractor(neo4j_connector, language)
            self.use_spacy = True
            logger.info("✓ spaCy extractor initialized")
        except Exception as e:
            logger.warning(f"⚠ spaCy not available: {e}")
        
        # Инициализируем LLM extractor как fallback
        try:
            self.llm_extractor = LLMEntityExtractor(
                neo4j_connector=neo4j_connector,
                model="qwen2:1.5b",  # Быстрая модель
                language=language
            )
            logger.info("✓ LLM extractor initialized (fallback)")
        except Exception as e:
            logger.error(f"❌ LLM extractor failed: {e}")
            raise
    
    def extract_entities_fast(self, text: str) -> List[Dict]:
        """Извлечение сущностей с fallback"""
        # Сначала пробуем spaCy
        if self.use_spacy:
            entities = self.spacy_extractor.extract_entities_fast(text)
            
            if len(entities) > 0:
                logger.info(f"✓ Found {len(entities)} entities with spaCy")
                return entities
            else:
                logger.info("⚠ spaCy found nothing, using LLM...")
        
        # Fallback на LLM
        return self.llm_extractor.extract_entities_fast(text)
    
    def create_knowledge_graph(
        self,
        text: str,
        document_id: str,
        user_id: str
    ) -> Tuple[int, int]:
        """Создание графа с гибридным подходом"""
        # Пробуем spaCy
        if self.use_spacy:
            entities = self.spacy_extractor.extract_entities_fast(text)
            
            if len(entities) > 0:
                logger.info(f"✓ Using spaCy ({len(entities)} entities)")
                return self.spacy_extractor.create_knowledge_graph(text, document_id, user_id)
        
        # Используем LLM
        logger.info("✓ Using LLM extractor")
        return self.llm_extractor.create_knowledge_graph(text, document_id, user_id)
