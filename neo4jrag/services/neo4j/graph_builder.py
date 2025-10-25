from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from neo4jrag.services.neo4j.neo4j_connector import Neo4jConnector
import logging
import json

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Построение графа знаний из документов"""
    
    def __init__(self, connector: Neo4jConnector, chunk_size: int = 500, chunk_overlap: int = 50):
        self.connector = connector
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def setup_schema(self) -> None:
        """Создание constraints и индексов"""
        constraints = [
            """
            CREATE CONSTRAINT unique_document IF NOT EXISTS
            FOR (d:Document) REQUIRE d.id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT unique_chunk IF NOT EXISTS
            FOR (c:Chunk) REQUIRE c.id IS UNIQUE
            """
        ]
        
        for constraint in constraints:
            try:
                self.connector.execute_write(constraint)
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.warning(f"Constraint issue: {e}")
        
        logger.info("✓ Schema setup complete")
    
    def add_document(
        self,
        doc_id: str,
        title: str,
        content: str,
        metadata: Dict = None
    ) -> int:
        """Добавление документа с разбиением на чанки"""
        metadata = metadata or {}
        
        # Создание документа
        # Сериализуем metadata в JSON строку, т.к. Neo4j не поддерживает вложенные словари
        doc_query = """
        MERGE (d:Document {id: $doc_id})
        SET d.title = $title,
            d.content = $content,
            d.preview = $preview,
            d.metadata_json = $metadata_json
        RETURN d
        """
        
        self.connector.execute_write(doc_query, {
            "doc_id": doc_id,
            "title": title,
            "content": content,
            "preview": content[:200],
            "metadata_json": json.dumps(metadata)  # ✅ Конвертируем в JSON строку
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
                c.length = $length
            MERGE (d)-[:HAS_CHUNK]->(c)
            """
            
            self.connector.execute_write(chunk_query, {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "text": chunk_text,
                "position": idx,
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
        
        logger.info(f"✓ Added document '{title}' with {len(chunks)} chunks")
        return len(chunks)
    
    def add_documents_batch(
        self,
        documents: List[Dict[str, str]]
    ) -> Dict[str, int]:
        """
        Пакетное добавление документов
        
        Args:
            documents: Список словарей с ключами: id, title, content, metadata
        
        Returns:
            Статистика добавления
        """
        stats = {"total_docs": 0, "total_chunks": 0}
        
        for doc in documents:
            chunk_count = self.add_document(
                doc_id=doc["id"],
                title=doc["title"],
                content=doc["content"],
                metadata=doc.get("metadata", {})
            )
            stats["total_docs"] += 1
            stats["total_chunks"] += chunk_count
        
        logger.info(f"✓ Added {stats['total_docs']} documents with {stats['total_chunks']} chunks")
        return stats
