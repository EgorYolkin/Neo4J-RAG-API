#!/usr/bin/env python3
"""
Neo4j RAG System - Main Entry Point
"""

from neo4jrag.config import Config
from neo4jrag.services.neo4j.neo4j_connector import Neo4jConnector
from neo4jrag.services.ollama.ollama_loader import OllamaLoader
from neo4jrag.services.neo4j.graph_builder import GraphBuilder
from neo4jrag.services.neo4j.vector_store import VectorStore
from neo4jrag.services.ollama.rag_pipeline import RAGPipeline
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Основная функция"""
    
    print("\n🚀 Запуск Neo4j RAG System\n")
    
    # Загрузка конфигурации
    config = Config.from_env()
    
    # Подключение к Neo4j
    connector = Neo4jConnector(
        uri=config.neo4j.uri,
        username=config.neo4j.username,
        password=config.neo4j.password,
        database=config.neo4j.database
    )
    connector.connect()
    
    # Загрузка Ollama
    ollama = OllamaLoader(
        base_url=config.ollama.base_url,
        model=config.ollama.model,
        embedding_model=config.ollama.embedding_model,
        temperature=config.ollama.temperature
    )
    ollama.load_llm()
    ollama.load_embeddings()
    
    # Построение графа
    print("\n📚 Создание графа знаний...")
    builder = GraphBuilder(
        connector=connector,
        chunk_size=config.rag.chunk_size,
        chunk_overlap=config.rag.chunk_overlap
    )
    builder.setup_schema()
    
    # Добавление примеров документов
    sample_docs = [
        {
            "id": "doc_ml",
            "title": "Машинное обучение",
            "content": """
            Машинное обучение — это раздел искусственного интеллекта, который изучает методы 
            построения алгоритмов, способных обучаться. Основные типы обучения включают: 
            обучение с учителем, обучение без учителя и обучение с подкреплением.
            
            Нейронные сети являются ключевым инструментом глубокого обучения. Они состоят 
            из слоёв нейронов, которые обрабатывают информацию и выявляют сложные паттерны.
            """
        },
        {
            "id": "doc_graphdb",
            "title": "Графовые базы данных",
            "content": """
            Графовые базы данных, такие как Neo4j, оптимизированы для хранения и обработки 
            связанных данных. Они используют узлы для сущностей и рёбра для связей между ними.
            
            RAG (Retrieval-Augmented Generation) — это подход, который комбинирует поиск 
            релевантной информации с генерацией ответов языковой моделью. GraphRAG расширяет 
            этот подход, используя графовую структуру для более точного поиска контекста.
            """
        }
    ]
    
    for doc in sample_docs:
        builder.add_document(
            doc_id=doc["id"],
            title=doc["title"],
            content=doc["content"]
        )
    
    # Настройка векторного хранилища
    print("\n🔍 Настройка векторного поиска...")
    vector_store = VectorStore(
        connector=connector,
        ollama=ollama,
        index_name=config.rag.vector_index_name,
        dimensions=config.rag.embedding_dimension
    )
    vector_store.create_vector_index()
    vector_store.generate_embeddings()
    
    # Создание RAG пайплайна
    print("\n🤖 Создание RAG пайплайна...")
    rag = RAGPipeline(vector_store=vector_store, ollama=ollama)
    
    # Статистика
    print("\n📊 Статистика графа:")
    stats = connector.get_statistics()
    for node_type, count in stats.get("nodes", {}).items():
        print(f"  {node_type}: {count}")
    for rel_type, count in stats.get("relationships", {}).items():
        print(f"  {rel_type}: {count}")
    
    print("\n✅ Система готова!\n")
    
    # Примеры вопросов
    rag.ask("Что такое машинное обучение?")
    rag.ask("Расскажи про GraphRAG")
    rag.ask("Что такое нейронные сети?")
    
    # Закрытие соединения
    connector.close()
    print("\n👋 Работа завершена")


if __name__ == "__main__":
    main()
