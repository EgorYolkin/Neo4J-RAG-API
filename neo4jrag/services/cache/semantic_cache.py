"""
Semantic Caching with Redis
"""
import redis
import numpy as np
import json
import hashlib
import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Запись в кэше"""
    query: str
    embedding: List[float]
    answer: str
    sources: List[Dict]
    search_type: str
    processing_steps: List[str]
    timestamp: float


class SemanticCache:
    """
    Семантическое кэширование на основе векторного сходства
    
    Принцип работы:
    1. Запрос преобразуется в эмбеддинг
    2. Ищутся похожие эмбеддинги в кэше (cosine similarity)
    3. Если similarity > threshold → cache hit, возвращаем кэшированный ответ
    4. Иначе → cache miss, выполняем запрос и кэшируем результат
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        password: Optional[str] = None,
        db: int = 0,
        ttl: int = 3600,
        similarity_threshold: float = 0.95,
        max_cache_size: int = 10000
    ):
        self.ttl = ttl
        self.similarity_threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        
        # Подключение к Redis
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            password=password,
            db=db,
            decode_responses=False  # False для работы с binary
        )
        
        # Проверка подключения
        try:
            self.redis_client.ping()
            logger.info(f"✓ Connected to Redis at {host}:{port}")
        except redis.ConnectionError as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            raise
        
        # Ключи для разных типов данных
        self.EMBEDDINGS_KEY = "semantic_cache:embeddings"
        self.QUERIES_KEY = "semantic_cache:queries"
        self.ANSWERS_KEY = "semantic_cache:answers"
        self.STATS_KEY = "semantic_cache:stats"
    
    def _generate_query_id(self, query: str) -> str:
        """Генерация уникального ID для запроса"""
        return hashlib.md5(query.encode('utf-8')).hexdigest()
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Вычисление косинусного сходства между векторами"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _serialize_embedding(self, embedding: List[float]) -> bytes:
        """Сериализация эмбеддинга в bytes"""
        return np.array(embedding, dtype=np.float32).tobytes()
    
    def _deserialize_embedding(self, data: bytes) -> np.ndarray:
        """Десериализация эмбеддинга из bytes"""
        return np.frombuffer(data, dtype=np.float32)
    
    def set(
        self,
        query: str,
        embedding: List[float],
        answer: str,
        sources: List[Dict],
        search_type: str,
        processing_steps: List[str]
    ) -> bool:
        """
        Сохранение результата в кэш
        
        Args:
            query: Оригинальный запрос
            embedding: Эмбеддинг запроса
            answer: Ответ RAG-системы
            sources: Источники для ответа
            search_type: Тип поиска
            processing_steps: Шаги обработки
        
        Returns:
            bool: Успешно ли сохранено
        """
        try:
            query_id = self._generate_query_id(query)
            
            # Проверяем размер кэша
            current_size = self.redis_client.zcard(self.EMBEDDINGS_KEY)
            if current_size >= self.max_cache_size:
                # Удаляем самые старые записи (FIFO)
                self.redis_client.zremrangebyrank(self.EMBEDDINGS_KEY, 0, 0)
            
            # Сохраняем эмбеддинг в Sorted Set (score = timestamp)
            import time
            timestamp = time.time()
            
            self.redis_client.zadd(
                self.EMBEDDINGS_KEY,
                {query_id: timestamp}
            )
            
            # Сохраняем сам эмбеддинг
            embedding_key = f"{self.EMBEDDINGS_KEY}:{query_id}"
            self.redis_client.setex(
                embedding_key,
                self.ttl,
                self._serialize_embedding(embedding)
            )
            
            # Сохраняем запрос
            query_key = f"{self.QUERIES_KEY}:{query_id}"
            self.redis_client.setex(query_key, self.ttl, query)
            
            # Сохраняем ответ и метаданные
            answer_data = {
                "answer": answer,
                "sources": sources,
                "search_type": search_type,
                "processing_steps": processing_steps,
                "timestamp": timestamp
            }
            answer_key = f"{self.ANSWERS_KEY}:{query_id}"
            self.redis_client.setex(
                answer_key,
                self.ttl,
                json.dumps(answer_data)
            )
            
            # Обновляем статистику
            self.redis_client.hincrby(self.STATS_KEY, "total_cached", 1)
            
            logger.debug(f"Cached query: {query[:50]}... (ID: {query_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache query: {e}")
            return False
    
    def get(
        self,
        query: str,
        embedding: List[float]
    ) -> Optional[Dict[str, Any]]:
        """
        Поиск похожего запроса в кэше
        
        Args:
            query: Запрос для поиска
            embedding: Эмбеддинг запроса
        
        Returns:
            Кэшированный результат или None
        """
        try:
            query_embedding = np.array(embedding, dtype=np.float32)
            
            # Получаем все кэшированные эмбеддинги
            cached_ids = self.redis_client.zrange(self.EMBEDDINGS_KEY, 0, -1)
            
            if not cached_ids:
                logger.debug("Cache is empty")
                self.redis_client.hincrby(self.STATS_KEY, "total_misses", 1)
                return None
            
            best_similarity = 0.0
            best_match_id = None
            
            # Ищем наиболее похожий эмбеддинг
            for cached_id in cached_ids:
                cached_id_str = cached_id.decode('utf-8')
                embedding_key = f"{self.EMBEDDINGS_KEY}:{cached_id_str}"
                
                cached_embedding_bytes = self.redis_client.get(embedding_key)
                if not cached_embedding_bytes:
                    continue
                
                cached_embedding = self._deserialize_embedding(cached_embedding_bytes)
                similarity = self._cosine_similarity(query_embedding, cached_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = cached_id_str
            
            # Проверяем порог
            if best_similarity < self.similarity_threshold:
                logger.debug(f"Best similarity {best_similarity:.3f} below threshold {self.similarity_threshold}")
                self.redis_client.hincrby(self.STATS_KEY, "total_misses", 1)
                return None
            
            # Cache hit!
            logger.info(f"Cache HIT! Similarity: {best_similarity:.3f} (query: {query[:50]}...)")
            self.redis_client.hincrby(self.STATS_KEY, "total_hits", 1)
            
            # Получаем кэшированный ответ
            answer_key = f"{self.ANSWERS_KEY}:{best_match_id}"
            answer_data_json = self.redis_client.get(answer_key)
            
            if not answer_data_json:
                return None
            
            answer_data = json.loads(answer_data_json)
            answer_data["cached"] = True
            answer_data["similarity"] = float(best_similarity)
            answer_data["original_query"] = self.redis_client.get(
                f"{self.QUERIES_KEY}:{best_match_id}"
            ).decode('utf-8')
            
            return answer_data
            
        except Exception as e:
            logger.error(f"Failed to get from cache: {e}")
            self.redis_client.hincrby(self.STATS_KEY, "total_errors", 1)
            return None
    
    def clear(self) -> bool:
        """Очистка всего кэша"""
        try:
            self.redis_client.delete(
                self.EMBEDDINGS_KEY,
                self.QUERIES_KEY,
                self.ANSWERS_KEY,
                self.STATS_KEY
            )
            
            # Удаляем все связанные ключи
            for key in self.redis_client.scan_iter(f"{self.EMBEDDINGS_KEY}:*"):
                self.redis_client.delete(key)
            for key in self.redis_client.scan_iter(f"{self.QUERIES_KEY}:*"):
                self.redis_client.delete(key)
            for key in self.redis_client.scan_iter(f"{self.ANSWERS_KEY}:*"):
                self.redis_client.delete(key)
            
            logger.info("✓ Cache cleared")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики кэша"""
        try:
            stats = self.redis_client.hgetall(self.STATS_KEY)
            
            total_hits = int(stats.get(b"total_hits", 0))
            total_misses = int(stats.get(b"total_misses", 0))
            total_cached = int(stats.get(b"total_cached", 0))
            total_errors = int(stats.get(b"total_errors", 0))
            
            total_requests = total_hits + total_misses
            hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0.0
            
            cache_size = self.redis_client.zcard(self.EMBEDDINGS_KEY)
            
            return {
                "cache_size": cache_size,
                "max_cache_size": self.max_cache_size,
                "total_cached": total_cached,
                "total_hits": total_hits,
                "total_misses": total_misses,
                "total_errors": total_errors,
                "total_requests": total_requests,
                "hit_rate": round(hit_rate, 2),
                "similarity_threshold": self.similarity_threshold,
                "ttl_seconds": self.ttl
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}
    
    def close(self) -> None:
        """Закрытие соединения с Redis"""
        try:
            self.redis_client.close()
            logger.info("✓ Redis connection closed")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")
