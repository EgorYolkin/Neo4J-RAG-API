from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Neo4jConnector:
    """Neo4j Connector for managing connections and executing queries."""
    
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.driver: Optional[GraphDatabase.driver] = None
    
    def connect(self) -> None:
        """Установка соединения с Neo4j"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            self.driver.verify_connectivity()
            logger.info(f"✓ Connected to Neo4j at {self.uri}")
        except AuthError as e:
            logger.error(f"❌ Authentication failed: {e}")
            raise
        except ServiceUnavailable as e:
            logger.error(f"❌ Neo4j service unavailable: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to connect: {e}")
            raise
    
    def close(self) -> None:
        """Закрытие соединения"""
        if self.driver:
            self.driver.close()
            logger.info("✓ Neo4j connection closed")
    
    @contextmanager
    def session(self):
        """Context manager для сессий"""
        if not self.driver:
            raise RuntimeError("Driver not initialized. Call connect() first.")
        
        session = self.driver.session(database=self.database)
        try:
            yield session
        finally:
            session.close()
    
    def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Выполнение read-запроса"""
        if not self.driver:
            raise RuntimeError("Driver not initialized. Call connect() first.")
        
        parameters = parameters or {}
        
        with self.session() as session:
            result = session.run(query, parameters)
            return [record.data() for record in result]
    
    def execute_write(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Выполнение write-запроса в транзакции"""
        if not self.driver:
            raise RuntimeError("Driver not initialized. Call connect() first.")
        
        parameters = parameters or {}
        
        def _write_tx(tx):
            result = tx.run(query, parameters)
            return [record.data() for record in result]
        
        with self.session() as session:
            return session.execute_write(_write_tx)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики графа"""
        stats = {}
        
        node_counts = self.execute_query("""
            MATCH (n)
            RETURN labels(n)[0] as label, count(n) as count
            ORDER BY count DESC
        """)
        stats["nodes"] = {item["label"]: item["count"] for item in node_counts if item["label"]}
        
        rel_counts = self.execute_query("""
            MATCH ()-[r]->()
            RETURN type(r) as type, count(r) as count
            ORDER BY count DESC
        """)
        stats["relationships"] = {item["type"]: item["count"] for item in rel_counts}
        
        return stats
    
    def __enter__(self):
        """Context manager вход"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager выход"""
        self.close()
