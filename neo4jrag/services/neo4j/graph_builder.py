"""
Graph builder с поддержкой entity extraction
"""
from typing import List, Dict, Optional, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json  # ✅ ДОБАВЛЕНО
import logging

from neo4jrag.services.neo4j.neo4j_connector import Neo4jConnector  # ✅ Полный импорт
from neo4jrag.services.entity_extractor.hybrid_entity_extractor import HybridEntityExtractor  # ✅ Полный импорт

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Построение графа знаний с извлечением сущностей"""
    
    def __init__(
        self, 
        connector: Neo4jConnector, 
        chunk_size: int = 500, 
        chunk_overlap: int = 50,
        entity_extractor: Optional[HybridEntityExtractor] = None
    ):
        self.connector = connector
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.entity_extractor = entity_extractor
        logger.info(f"✓ GraphBuilder initialized (entity extraction: {entity_extractor is not None})")
    
    def setup_schema(self) -> None:
        """Создание constraints и индексов"""
        constraints = [
            """
            CREATE CONSTRAINT unique_user IF NOT EXISTS
            FOR (u:User) REQUIRE u.id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT unique_document IF NOT EXISTS
            FOR (d:Document) REQUIRE d.id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT unique_chunk IF NOT EXISTS
            FOR (c:Chunk) REQUIRE c.id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT unique_entity IF NOT EXISTS
            FOR (e:Entity) REQUIRE (e.name, e.user_id) IS UNIQUE
            """
        ]
        
        for constraint in constraints:
            try:
                self.connector.execute_write(constraint)
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.warning(f"Constraint issue: {e}")
        
        # Индексы
        indexes = [
            """
            CREATE INDEX document_user_idx IF NOT EXISTS
            FOR (d:Document) ON (d.user_id)
            """,
            """
            CREATE INDEX entity_user_idx IF NOT EXISTS
            FOR (e:Entity) ON (e.user_id)
            """,
            """
            CREATE INDEX entity_type_idx IF NOT EXISTS
            FOR (e:Entity) ON (e.type)
            """
        ]
        
        for index in indexes:
            try:
                self.connector.execute_write(index)
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.warning(f"Index issue: {e}")
        
        logger.info("✓ Schema setup complete")
    
    def add_document(
        self,
        doc_id: str,
        title: str,
        content: str,
        user_id: str,
        metadata: Dict = None,
        extract_entities: bool = False
    ) -> int:
        """
        Добавление документа с опциональным извлечением сущностей
        """
        metadata = metadata or {}
        
        # Создание документа
        doc_query = """
        MERGE (u:User {id: $user_id})
        MERGE (d:Document {id: $doc_id})
        SET d.title = $title,
            d.content = $content,
            d.preview = $preview,
            d.user_id = $user_id,
            d.metadata_json = $metadata_json,
            d.created_at = datetime()
        MERGE (u)-[:OWNS]->(d)
        RETURN d
        """
        
        self.connector.execute_write(doc_query, {
            "user_id": user_id,
            "doc_id": doc_id,
            "title": title,
            "content": content,
            "preview": content[:200],
            "metadata_json": json.dumps(metadata)  # ✅ Теперь json доступен
        })
        
        # Разбиение на чанки
        chunks = self.text_splitter.split_text(content)
        
        for idx, chunk_text in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{idx}"
            
            chunk_query = """
            MATCH (d:Document {id: $doc_id})
            MERGE (c:Chunk {id: $chunk_id})
            SET c.text = $text,
                c.position = $position,
                c.user_id = $user_id,
                c.length = $length
            MERGE (d)-[:HAS_CHUNK]->(c)
            """
            
            self.connector.execute_write(chunk_query, {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "text": chunk_text,
                "position": idx,
                "user_id": user_id,
                "length": len(chunk_text)
            })
            
            # Связь с предыдущим чанком
            if idx > 0:
                prev_chunk_id = f"{doc_id}_chunk_{idx-1}"
                next_query = """
                MATCH (c1:Chunk {id: $prev_id})
                MATCH (c2:Chunk {id: $curr_id})
                MERGE (c1)-[:NEXT]->(c2)
                """
                self.connector.execute_write(next_query, {
                    "prev_id": prev_chunk_id,
                    "curr_id": chunk_id
                })
        
        logger.info(f"✓ Added document '{title}' for user {user_id} with {len(chunks)} chunks")
        
        # Извлечение сущностей
        if extract_entities and self.entity_extractor:
            logger.info(f"🧠 Extracting entities from document '{title}'...")
            try:
                text_sample = content[:3000] if len(content) > 3000 else content
                
                entities_count, relationships_count = self.entity_extractor.create_knowledge_graph(
                    text=text_sample,
                    document_id=doc_id,
                    user_id=user_id
                )
                
                logger.info(f"✓ Extracted {entities_count} entities and {relationships_count} relationships")
            except Exception as e:
                logger.error(f"❌ Entity extraction failed: {e}")
        
        return len(chunks)
