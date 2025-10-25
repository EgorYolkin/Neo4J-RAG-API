"""
Graph builder with multi-user support
"""
from typing import List, Dict, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from neo4jrag.services.neo4j.neo4j_connector import Neo4jConnector
import logging

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Построение графа знаний с поддержкой многопользовательского режима"""
    
    def __init__(self, connector: Neo4jConnector, chunk_size: int = 500, chunk_overlap: int = 50):
        self.connector = connector
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def setup_schema(self) -> None:
        """Создание constraints и индексов с поддержкой user_id"""
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
            """
        ]
        
        for constraint in constraints:
            try:
                self.connector.execute_write(constraint)
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.warning(f"Constraint issue: {e}")
        
        # Создаём индекс для быстрого поиска по user_id
        try:
            self.connector.execute_write("""
                CREATE INDEX document_user_idx IF NOT EXISTS
                FOR (d:Document) ON (d.user_id)
            """)
        except Exception as e:
            if "already exists" not in str(e).lower():
                logger.warning(f"Index issue: {e}")
        
        logger.info("✓ Schema setup complete with multi-user support")
    
    def create_user(self, user_id: str, username: str, email: str) -> None:
        """Создание узла пользователя"""
        query = """
        MERGE (u:User {id: $user_id})
        SET u.username = $username,
            u.email = $email,
            u.created_at = datetime()
        RETURN u
        """
        
        self.connector.execute_write(query, {
            "user_id": user_id,
            "username": username,
            "email": email
        })
        
        logger.info(f"✓ User '{username}' created/updated")
    
    def add_document(
        self,
        doc_id: str,
        title: str,
        content: str,
        user_id: str,  # ✅ Добавлен user_id
        metadata: Dict = None
    ) -> int:
        """Добавление документа с привязкой к пользователю"""
        metadata = metadata or {}
        
        # Создание документа с user_id
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
        
        import json
        self.connector.execute_write(doc_query, {
            "user_id": user_id,
            "doc_id": doc_id,
            "title": title,
            "content": content,
            "preview": content[:200],
            "metadata_json": json.dumps(metadata)
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
                c.user_id = $user_id
            MERGE (d)-[:HAS_CHUNK]->(c)
            """
            
            self.connector.execute_write(chunk_query, {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "text": chunk_text,
                "position": idx,
                "user_id": user_id
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
        return len(chunks)
