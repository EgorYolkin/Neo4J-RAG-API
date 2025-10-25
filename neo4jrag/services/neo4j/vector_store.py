from typing import List, Dict
from .neo4j_connector import Neo4jConnector
from ..ollama.ollama_loader import OllamaLoader
import logging

logger = logging.getLogger(__name__)


class VectorStore:
    """Управление векторными эмбеддингами"""
    
    def __init__(
        self,
        connector: Neo4jConnector,
        ollama: OllamaLoader,
        index_name: str = "chunk_embeddings",
        dimensions: int = 768
    ):
        self.connector = connector
        self.ollama = ollama
        self.index_name = index_name
        self.dimensions = dimensions
    
    def create_vector_index(self) -> None:
        """Создание векторного индекса"""
        query = f"""
        CREATE VECTOR INDEX {self.index_name} IF NOT EXISTS
        FOR (c:Chunk)
        ON c.embedding
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {self.dimensions},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """
        
        try:
            self.connector.execute_write(query)
            logger.info(f"✓ Vector index '{self.index_name}' created")
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info(f"✓ Vector index '{self.index_name}' already exists")
            else:
                raise
    
    def generate_embeddings(self) -> int:
        """Генерация эмбеддингов для всех чанков"""
        chunks = self.connector.execute_query("""
            MATCH (c:Chunk)
            WHERE c.embedding IS NULL
            RETURN c.id as id, c.text as text
        """)
        
        if not chunks:
            logger.info("✓ All chunks have embeddings")
            return 0
        
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        
        for idx, chunk in enumerate(chunks):
            embedding = self.ollama.embed_text(chunk["text"])
            
            self.connector.execute_write("""
                MATCH (c:Chunk {id: $chunk_id})
                SET c.embedding = $embedding
            """, {"chunk_id": chunk["id"], "embedding": embedding})
            
            if (idx + 1) % 10 == 0:
                logger.info(f"  Processed {idx + 1}/{len(chunks)}")
        
        logger.info(f"✓ Generated {len(chunks)} embeddings")
        return len(chunks)
    
    def similarity_search(self, query: str, k: int = 3) -> List[Dict]:
        """Векторный поиск"""
        query_embedding = self.ollama.embed_text(query)
        
        results = self.connector.execute_query(f"""
            CALL db.index.vector.queryNodes(
                '{self.index_name}',
                {k},
                $query_embedding
            )
            YIELD node, score
            RETURN node.id as chunk_id,
                   node.text as text,
                   score
            ORDER BY score DESC
        """, {"query_embedding": query_embedding})
        
        return results
    
    def hybrid_search(self, query: str, k: int = 3) -> List[Dict]:
        """Гибридный поиск с контекстом"""
        vector_results = self.similarity_search(query, k)
        
        enriched = []
        for result in vector_results:
            context = self.connector.execute_query("""
                MATCH (c:Chunk {id: $chunk_id})
                OPTIONAL MATCH (prev:Chunk)-[:NEXT]->(c)
                OPTIONAL MATCH (c)-[:NEXT]->(next:Chunk)
                OPTIONAL MATCH (d:Document)-[:HAS_CHUNK]->(c)
                RETURN 
                    c.text as current,
                    prev.text as prev,
                    next.text as next,
                    d.title as doc_title
                LIMIT 1
            """, {"chunk_id": result["chunk_id"]})
            
            if context:
                ctx = context[0]
                combined = ""
                if ctx.get("prev"):
                    combined += f"[Предыдущий]: {ctx['prev']}\n\n"
                combined += f"[Основной]: {ctx['current']}"
                if ctx.get("next"):
                    combined += f"\n\n[Следующий]: {ctx['next']}"
                
                enriched.append({
                    "text": combined,
                    "doc_title": ctx.get("doc_title", "Unknown"),
                    "score": result["score"]
                })
        
        return enriched
