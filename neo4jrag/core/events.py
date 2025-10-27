"""
Application lifecycle events (startup/shutdown)
"""
import logging
from typing import Optional

from neo4jrag.services.cache.semantic_cache import SemanticCache

from neo4jrag.config import Config
from neo4jrag.services.neo4j.neo4j_connector import Neo4jConnector
from neo4jrag.services.neo4j.graph_builder import GraphBuilder
from neo4jrag.services.neo4j.vector_store import VectorStore
from neo4jrag.services.ollama.ollama_loader import OllamaLoader
from neo4jrag.services.ollama.rag_pipeline import RAGPipeline
from neo4jrag.services.entity_extractor.hybrid_entity_extractor import HybridEntityExtractor
from .exceptions import RAGInitializationError

logger = logging.getLogger(__name__)


async def startup_event(config: Config) -> dict:
    """
    Startup event handler
    
    Инициализирует все компоненты системы:
    1. Подключение к Neo4j
    2. Загрузка Ollama моделей
    3. Настройка схемы графа
    4. Создание векторного индекса
    5. Инициализация RAG pipeline
    
    Returns:
        dict: Словарь с инициализированными компонентами
    """
    logger.info("=" * 80)
    logger.info("🚀 Starting Neo4j RAG System")
    logger.info("=" * 80)
    
    components = {}
    
    try:
        # 1. Neo4j Connection
        logger.info("📊 Connecting to Neo4j...")
        neo4j_connector = Neo4jConnector(
            uri=config.neo4j.uri,
            username=config.neo4j.username,
            password=config.neo4j.password,
            database=config.neo4j.database
        )
        neo4j_connector.connect()
        components["neo4j_connector"] = neo4j_connector
        logger.info(f"✓ Connected to Neo4j at {config.neo4j.uri}")
        
        # 2. Ollama Models
        logger.info("🤖 Loading Ollama models...")
        ollama_loader = OllamaLoader(
            base_url=config.ollama.base_url,
            model=config.ollama.model,
            embedding_model=config.ollama.embedding_model,
            temperature=config.ollama.temperature
        )
        ollama_loader.load_llm()
        ollama_loader.load_embeddings()
        components["ollama_loader"] = ollama_loader
        logger.info(f"✓ Loaded LLM: {config.ollama.model}")
        logger.info(f"✓ Loaded Embeddings: {config.ollama.embedding_model}")
        
        # 3. Graph Builder
        logger.info("📚 Setting up Graph Builder...")
        graph_builder = GraphBuilder(
            connector=neo4j_connector,
            chunk_size=config.rag.chunk_size,
            chunk_overlap=config.rag.chunk_overlap
        )
        graph_builder.setup_schema()
        components["graph_builder"] = graph_builder
        logger.info("✓ Graph schema configured")
        
        # 4. Vector Store
        logger.info("🔍 Setting up Vector Store...")
        vector_store = VectorStore(
            connector=neo4j_connector,
            ollama=ollama_loader,
            index_name=config.rag.vector_index_name,
            dimensions=config.rag.embedding_dimension
        )
        vector_store.create_vector_index()
        components["vector_store"] = vector_store
        logger.info(f"✓ Vector index '{config.rag.vector_index_name}' ready")
        
        # 5. RAG Pipeline
        logger.info("⚙️ Initializing RAG Pipeline...")
        rag_pipeline = RAGPipeline(
            vector_store=vector_store,
            ollama=ollama_loader
        )
        components["rag_pipeline"] = rag_pipeline
        logger.info("✓ RAG Pipeline initialized")

        logger.info("💾 Connecting to Redis...")
        semantic_cache = SemanticCache(
            host=config.redis.host,
            port=config.redis.port,
            password=config.redis.password,
            db=config.redis.db,
            ttl=config.redis.cache_ttl,
            similarity_threshold=config.redis.semantic_cache_threshold,
            max_cache_size=config.redis.max_cache_size
        )
        components["semantic_cache"] = semantic_cache
        
        # Статистика кэша
        cache_stats = semantic_cache.get_stats()
        logger.info(f"✓ Redis connected (cache size: {cache_stats.get('cache_size', 0)})")

        # 6. Hybrid Entity Extractor
        logger.info("🧩 Initializing Hybrid Entity Extractor...")
        hybrid_extractor = HybridEntityExtractor(
            neo4j_connector=neo4j_connector,
            language="ru"  # или "en"
        )
        components["entity_extractor"] = hybrid_extractor
        logger.info("✓ Hybrid Entity Extractor initialized")

        
        # 6. Initial Statistics
        logger.info("\n📊 System Statistics:")
        try:
            stats = neo4j_connector.get_statistics()
            for node_type, count in stats.get("nodes", {}).items():
                logger.info(f"  • {node_type}: {count} nodes")
            for rel_type, count in stats.get("relationships", {}).items():
                logger.info(f"  • {rel_type}: {count} relationships")
            
            # Проверка эмбеддингов
            embeddings_query = """
            MATCH (c:Chunk)
            RETURN 
                count(c) as total,
                count(c.embedding) as with_embeddings
            """
            emb_result = neo4j_connector.execute_query(embeddings_query)
            if emb_result:
                total = emb_result[0]["total"]
                with_emb = emb_result[0]["with_embeddings"]
                coverage = (with_emb / total * 100) if total > 0 else 0
                logger.info(f"  • Embeddings coverage: {with_emb}/{total} ({coverage:.1f}%)")
        except Exception as e:
            logger.warning(f"Could not fetch statistics: {e}")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ System startup completed successfully")
        logger.info("=" * 80 + "\n")
        
        return components
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"❌ Startup failed: {str(e)}")
        logger.error("=" * 80)
        
        # Cleanup частично инициализированных компонентов
        await cleanup_components(components)
        
        raise RAGInitializationError(f"System startup failed: {str(e)}")


async def shutdown_event(components: dict) -> None:
    """
    Shutdown event handler
    
    Корректно закрывает все соединения и освобождает ресурсы:
    1. Закрывает соединение с Neo4j
    2. Очищает кэши
    3. Логирует финальную статистику
    
    Args:
        components: Словарь с компонентами системы
    """
    logger.info("\n" + "=" * 80)
    logger.info("🛑 Shutting down Neo4j RAG System")
    logger.info("=" * 80)
    
    await cleanup_components(components)
    
    logger.info("=" * 80)
    logger.info("👋 System shutdown completed")
    logger.info("=" * 80 + "\n")


async def cleanup_components(components: dict) -> None:
    """
    Очистка всех компонентов системы
    
    Args:
        components: Словарь с компонентами для очистки
    """
    # Закрываем Neo4j соединение
    neo4j_connector = components.get("neo4j_connector")
    if neo4j_connector:
        try:
            # Финальная статистика перед закрытием
            logger.info("\n📊 Final Statistics:")
            stats = neo4j_connector.get_statistics()
            for node_type, count in stats.get("nodes", {}).items():
                logger.info(f"  • {node_type}: {count} nodes")
            for rel_type, count in stats.get("relationships", {}).items():
                logger.info(f"  • {rel_type}: {count} relationships")
            
            neo4j_connector.close()
            logger.info("✓ Neo4j connection closed")
        except Exception as e:
            logger.error(f"Error closing Neo4j connection: {e}")

    semantic_cache = components.get("semantic_cache")
    if semantic_cache:
        try:
            cache_stats = semantic_cache.get_stats()
            logger.info(f"\n💾 Final Cache Statistics:")
            logger.info(f"  • Total requests: {cache_stats.get('total_requests', 0)}")
            logger.info(f"  • Cache hits: {cache_stats.get('total_hits', 0)}")
            logger.info(f"  • Cache misses: {cache_stats.get('total_misses', 0)}")
            logger.info(f"  • Hit rate: {cache_stats.get('hit_rate', 0)}%")
            
            semantic_cache.close()
            logger.info("✓ Redis connection closed")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")
    
    # Очищаем остальные компоненты
    components_to_clear = ["rag_pipeline", "vector_store", "graph_builder", "ollama_loader"]
    for comp_name in components_to_clear:
        if comp_name in components:
            components[comp_name] = None
            logger.info(f"✓ Cleared {comp_name}")
    
    components.clear()


async def health_check_event(components: dict) -> dict:
    """
    Периодическая проверка здоровья компонентов
    
    Args:
        components: Словарь с компонентами системы
    
    Returns:
        dict: Статус каждого компонента
    """
    health_status = {
        "overall": "healthy",
        "components": {}
    }
    
    # Neo4j
    try:
        neo4j_connector = components.get("neo4j_connector")
        if neo4j_connector:
            # Простой тестовый запрос
            neo4j_connector.execute_query("RETURN 1")
            health_status["components"]["neo4j"] = "healthy"
        else:
            health_status["components"]["neo4j"] = "not_initialized"
            health_status["overall"] = "degraded"
    except Exception as e:
        health_status["components"]["neo4j"] = f"unhealthy: {str(e)}"
        health_status["overall"] = "unhealthy"
    
    # Ollama
    try:
        ollama_loader = components.get("ollama_loader")
        if ollama_loader:
            # Тестовый эмбеддинг
            test_embedding = ollama_loader.embed_text("health check")
            if len(test_embedding) > 0:
                health_status["components"]["ollama"] = "healthy"
            else:
                health_status["components"]["ollama"] = "unhealthy: empty embedding"
                health_status["overall"] = "unhealthy"
        else:
            health_status["components"]["ollama"] = "not_initialized"
            health_status["overall"] = "degraded"
    except Exception as e:
        health_status["components"]["ollama"] = f"unhealthy: {str(e)}"
        health_status["overall"] = "unhealthy"
    
    # RAG Pipeline
    rag_pipeline = components.get("rag_pipeline")
    if rag_pipeline:
        health_status["components"]["rag_pipeline"] = "healthy"
    else:
        health_status["components"]["rag_pipeline"] = "not_initialized"
        health_status["overall"] = "degraded"
    
    return health_status


async def warm_up_event(components: dict) -> None:
    """
    Прогрев системы после запуска
    
    Выполняет тестовые операции для:
    - Прогрева соединений
    - Загрузки моделей в память
    - Проверки работоспособности
    
    Args:
        components: Словарь с компонентами системы
    """
    logger.info("🔥 Warming up system...")
    
    try:
        # Прогрев Ollama
        ollama_loader = components.get("ollama_loader")
        if ollama_loader:
            logger.info("  • Warming up Ollama models...")
            # Генерируем тестовый эмбеддинг
            test_embedding = ollama_loader.embed_text("test warmup query")
            # Генерируем тестовый текст
            test_response = ollama_loader.generate("Say 'ready' if you are working.")
            logger.info("  ✓ Ollama models warmed up")
        
        # Прогрев Neo4j
        neo4j_connector = components.get("neo4j_connector")
        if neo4j_connector:
            logger.info("  • Warming up Neo4j connection...")
            # Несколько тестовых запросов
            neo4j_connector.execute_query("RETURN 1")
            neo4j_connector.execute_query("MATCH (n) RETURN count(n) LIMIT 1")
            logger.info("  ✓ Neo4j connection warmed up")
        
        # Прогрев векторного поиска
        vector_store = components.get("vector_store")
        if vector_store:
            logger.info("  • Warming up vector search...")
            try:
                # Тестовый векторный поиск
                vector_store.similarity_search("test query", k=1)
                logger.info("  ✓ Vector search warmed up")
            except Exception as e:
                logger.warning(f"  ⚠ Vector search warmup failed: {e}")
        
        logger.info("✓ System warmup completed\n")
        
    except Exception as e:
        logger.warning(f"Warmup encountered issues: {e}")


async def initialize_sample_data(components: dict) -> None:
    """
    Инициализация примеров данных (для разработки)
    
    Args:
        components: Словарь с компонентами системы
    """
    logger.info("📝 Checking for sample data...")
    
    neo4j_connector = components.get("neo4j_connector")
    graph_builder = components.get("graph_builder")
    vector_store = components.get("vector_store")
    
    if not all([neo4j_connector, graph_builder, vector_store]):
        logger.warning("Components not ready for sample data initialization")
        return
    
    try:
        # Проверяем, есть ли уже документы
        count_query = "MATCH (d:Document) RETURN count(d) as count"
        result = neo4j_connector.execute_query(count_query)
        doc_count = result[0]["count"] if result else 0
        
        if doc_count > 0:
            logger.info(f"✓ Found {doc_count} existing documents, skipping sample data")
            return
        
        logger.info("Adding sample documents...")
        
        sample_docs = [
            {
                "id": "sample_ml",
                "title": "Введение в машинное обучение",
                "content": """
                Машинное обучение (Machine Learning, ML) — это раздел искусственного интеллекта, 
                который изучает методы построения алгоритмов, способных обучаться на данных.
                
                Основные типы обучения включают:
                1. Обучение с учителем (Supervised Learning) - обучение на размеченных данных
                2. Обучение без учителя (Unsupervised Learning) - поиск паттернов в неразмеченных данных
                3. Обучение с подкреплением (Reinforcement Learning) - обучение через взаимодействие со средой
                
                Нейронные сети являются ключевым инструментом глубокого обучения. Они состоят 
                из слоёв нейронов, которые обрабатывают информацию и выявляют сложные паттерны.
                """,
                "metadata": {"category": "AI", "difficulty": "beginner"}
            },
            {
                "id": "sample_graphdb",
                "title": "Графовые базы данных и Neo4j",
                "content": """
                Графовые базы данных, такие как Neo4j, оптимизированы для хранения и обработки 
                связанных данных. Они используют узлы для представления сущностей и рёбра для 
                связей между ними.
                
                Основные преимущества графовых БД:
                - Естественное моделирование связей
                - Высокая производительность при обходе графа
                - Гибкая схема данных
                - Мощный язык запросов Cypher
                
                RAG (Retrieval-Augmented Generation) — это подход, который комбинирует поиск 
                релевантной информации с генерацией ответов языковой моделью. GraphRAG расширяет 
                этот подход, используя графовую структуру для более точного поиска контекста.
                """,
                "metadata": {"category": "Databases", "difficulty": "intermediate"}
            }
        ]
        
        for doc in sample_docs:
            graph_builder.add_document(
                doc_id=doc["id"],
                title=doc["title"],
                content=doc["content"],
                metadata=doc["metadata"]
            )
        
        # Генерация эмбеддингов
        vector_store.generate_embeddings()
        
        logger.info(f"✓ Added {len(sample_docs)} sample documents\n")
        
    except Exception as e:
        logger.warning(f"Failed to initialize sample data: {e}")
