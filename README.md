# 🚀 Neo4j GraphRAG System

Production-ready система для работы с RAG (Retrieval-Augmented Generation) на базе графовой базы данных Neo4j с семантическим кэшированием Redis.

## 📋 Возможности

### Основной функционал
- 🧠 **GraphRAG** - Retrieval-Augmented Generation с использованием графовой структуры Neo4j
- 🔍 **Гибридный поиск** - Векторный и графовый поиск для максимальной точности
- 💾 **Семантическое кэширование** - Redis-кэш с поиском по векторному сходству
- 🤖 **Локальные LLM** - Ollama для запуска моделей без облака
- 🔗 **LangGraph workflows** - Сложные RAG-пайплайны с управлением состоянием
- 🌐 **REST API** - FastAPI с автоматической документацией

### Технологический стек
- **Backend**: FastAPI + Uvicorn
- **Graph Database**: Neo4j 5.x с APOC плагинами
- **Vector Search**: Neo4j Vector Index (cosine similarity)
- **Cache**: Redis 7.x с семантическим кэшированием
- **LLM Framework**: LangChain + LangGraph
- **Embeddings & LLM**: Ollama (llama3.1, nomic-embed-text)
- **Language**: Python 3.12+

---

## 🏗️ Архитектура

```
Neo4jRAG/
├── app.py                          # FastAPI entry point
├── main.py                         # CLI tool для тестов
│
├── neo4jrag/
│   ├── config.py                   # Конфигурация (Pydantic)
│   │
│   ├── api/                        # REST API Layer
│   │   ├── deps.py                 # FastAPI Dependencies
│   │   └── v1/
│   │       ├── endpoints/          # API endpoints
│   │       │   ├── health.py       # Health checks
│   │       │   ├── query.py        # RAG queries
│   │       │   ├── documents.py    # Document CRUD
│   │       │   ├── stats.py        # Graph statistics
│   │       │   └── cache.py        # Cache management
│   │       ├── middleware/         # CORS, logging, errors
│   │       └── router.py           # Main router
│   │
│   ├── core/                       # Business Logic
│   │   ├── events.py               # Startup/Shutdown handlers
│   │   └── exceptions.py           # Custom exceptions
│   │
│   ├── services/                   # Service Layer
│   │   ├── neo4j/                  # Neo4j services
│   │   │   ├── neo4j_connector.py  # Connection & queries
│   │   │   ├── graph_builder.py    # Knowledge graph builder
│   │   │   └── vector_store.py     # Vector search
│   │   │
│   │   ├── ollama/                 # Ollama/LLM services
│   │   │   ├── ollama_loader.py    # Model loader
│   │   │   └── rag_pipeline.py     # LangGraph RAG workflow
│   │   │
│   │   └── cache/                  # Caching services
│   │       └── semantic_cache.py   # Redis semantic cache
│   │
│   ├── domain/                     # Domain Layer
│   │   └── schemas/                # Pydantic schemas
│   │       ├── request.py          # Request DTOs
│   │       └── response.py         # Response DTOs
│   │
│   └── utils/                      # Utilities
│       ├── logger.py               # Logging setup
│       └── validators.py           # Custom validators
│
└── tests/                          # Tests
    ├── test_api/                   # API tests
    └── test_services/              # Service tests
```

---

## 🚀 Быстрый старт

### Требования

- **Python** 3.12+
- **Docker** (для Neo4j и Redis)
- **Ollama** (для локальных LLM)
- **uv** (менеджер пакетов Python)

---

## 📦 Установка и запуск

### Вариант 1: Запуск с Docker Compose (рекомендуется)

**1. Клонируйте репозиторий**
```
git clone <repository-url>
cd Neo4jRAG
```

**2. Создайте `.env` файл**
```
cp .env.example .env
```

Содержимое `.env`:
```
# Application
ENVIRONMENT=development
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO

# Neo4j
NEO4J_URI=bolt://neo4j:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password123
NEO4J_DATABASE=neo4j

# Ollama (запущен на хосте)
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=llama3.1
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# Redis
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_CACHE_TTL=3600
REDIS_SEMANTIC_THRESHOLD=0.95
REDIS_MAX_CACHE_SIZE=10000
```

**3. Установите Ollama и модели**
```
# macOS
brew install ollama

# Запустите сервис
ollama serve

# Загрузите модели
ollama pull llama3.1
ollama pull nomic-embed-text
```

**4. Запустите через Docker Compose**
```
cd infra/docker
docker-compose up -d
```

Это запустит:
- Neo4j (http://localhost:7474, bolt://localhost:7687)
- Redis (localhost:6379)
- FastAPI Backend (http://localhost:8000)

**5. Проверьте статус**
```
docker-compose ps
docker-compose logs -f backend
```

**6. Откройте документацию API**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

### Вариант 2: Локальный запуск (без Docker для backend)

**1. Установите зависимости**
```
# Установите uv (если нет)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Установите зависимости проекта
uv sync
```

**2. Запустите Neo4j и Redis в Docker**
```
# Neo4j
docker run -d \
    --name neo4j-local \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password123 \
    -e NEO4J_PLUGINS='["apoc"]' \
    neo4j:latest

# Redis
docker run -d \
    --name redis-cache \
    -p 6379:6379 \
    redis:7-alpine
```

**3. Установите и запустите Ollama**
```
# macOS
brew install ollama
ollama serve

# Загрузите модели
ollama pull llama3.1
ollama pull nomic-embed-text
```

**4. Создайте `.env` файл**
```
# Application
ENVIRONMENT=development
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password123
NEO4J_DATABASE=neo4j

# Ollama (локально)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_CACHE_TTL=3600
REDIS_SEMANTIC_THRESHOLD=0.95
REDIS_MAX_CACHE_SIZE=10000
```

**5. Запустите приложение**
```
# Через uv
uv run uvicorn app:app --reload --port 8000

# Или через Python напрямую
uv run python app.py
```

**6. Откройте документацию**
- Swagger: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## 🧪 Тестирование API

### 1. Health Check
```
curl http://localhost:8000/api/v1/health
```

Ответ:
```
{
  "status": "healthy",
  "components": {
    "neo4j": "healthy",
    "ollama": "healthy",
    "redis": "healthy"
  },
  "version": "1.0.0"
}
```

### 2. Создание документа
```
curl -X POST http://localhost:8000/api/v1/documents/ \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Машинное обучение",
    "content": "Машинное обучение — это раздел искусственного интеллекта...",
    "metadata": {"category": "AI"}
  }'
```

### 3. RAG запрос (первый раз - без кэша)
```
curl -X POST http://localhost:8000/api/v1/query/ \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Что такое машинное обучение?",
    "top_k": 3,
    "search_type": "hybrid"
  }'
```

Ответ:
```
{
  "question": "Что такое машинное обучение?",
  "answer": "Машинное обучение — это раздел искусственного интеллекта...",
  "sources": [
    {
      "text": "Машинное обучение...",
      "score": 0.95,
      "doc_title": "Машинное обучение"
    }
  ],
  "search_type": "hybrid",
  "processing_steps": ["Маршрут: Гибридный поиск", "Найдено 3 результата"],
  "cached": false
}
```

### 4. Повторный запрос (из кэша)
```
curl -X POST http://localhost:8000/api/v1/query/ \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Что такое машинное обучение?"
  }'
```

Ответ:
```
{
  "question": "Что такое машинное обучение?",
  "answer": "Машинное обучение — это раздел искусственного интеллекта...",
  "sources": [...],
  "search_type": "hybrid",
  "processing_steps": ["✓ Retrieved from cache"],
  "cached": true,
  "cache_similarity": 1.0
}
```

### 5. Статистика кэша
```
curl http://localhost:8000/api/v1/cache/stats
```

Ответ:
```
{
  "cache_size": 10,
  "max_cache_size": 10000,
  "total_hits": 5,
  "total_misses": 3,
  "hit_rate": 62.5,
  "similarity_threshold": 0.95
}
```

### 6. Статистика графа
```
curl http://localhost:8000/api/v1/stats/
```

---

## 📚 Примеры использования

### Python SDK
```
import requests

BASE_URL = "http://localhost:8000/api/v1"

# Создание документа
response = requests.post(f"{BASE_URL}/documents/", json={
    "title": "GraphRAG Introduction",
    "content": "GraphRAG combines knowledge graphs with RAG...",
    "metadata": {"author": "AI Research Team"}
})
doc = response.json()
print(f"Created document: {doc['id']}")

# RAG запрос
response = requests.post(f"{BASE_URL}/query/", json={
    "question": "What is GraphRAG?",
    "top_k": 3
})
result = response.json()
print(f"Answer: {result['answer']}")
print(f"Cached: {result['cached']}")

# Получить список всех документов
response = requests.get(f"{BASE_URL}/documents/?skip=0&limit=10")
docs = response.json()
print(f"Total documents: {docs['total']}")
```

---

## 🔧 Конфигурация

### Настройка RAG параметров

Отредактируйте `.env`:

```
# Размер чанков для разбиения текста
RAG_CHUNK_SIZE=500
RAG_CHUNK_OVERLAP=50

# Количество результатов для поиска
RAG_TOP_K=3

# Порог схожести для семантического кэша (0-1)
REDIS_SEMANTIC_THRESHOLD=0.95

# TTL кэша в секундах (1 час = 3600)
REDIS_CACHE_TTL=3600

# Максимальный размер кэша
REDIS_MAX_CACHE_SIZE=10000
```

### Настройка моделей Ollama

```
# LLM модель для генерации ответов
OLLAMA_MODEL=llama3.1

# Модель для эмбеддингов
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

Доступные модели:
```
# Посмотреть установленные модели
ollama list

# Установить другую модель
ollama pull mistral
ollama pull llama2
```

---

## 🛠️ Управление системой

### Остановка и перезапуск

**Docker Compose:**
```
# Остановить все сервисы
docker-compose down

# Остановить с удалением volumes (удалит данные!)
docker-compose down -v

# Перезапустить
docker-compose restart

# Перезапустить только backend
docker-compose restart backend
```

**Локальный запуск:**
```
# Остановить Neo4j
docker stop neo4j-local

# Остановить Redis
docker stop redis-cache

# Запустить снова
docker start neo4j-local
docker start redis-cache
```

### Очистка данных

**Очистить кэш Redis:**
```
curl -X DELETE http://localhost:8000/api/v1/cache/clear
```

**Очистить граф Neo4j:**
```
# Через Neo4j Browser (http://localhost:7474)
MATCH (n) DETACH DELETE n

# Или через API
curl -X DELETE http://localhost:8000/api/v1/documents/{document_id}
```

### Просмотр логов

**Docker Compose:**
```
# Все логи
docker-compose logs -f

# Только backend
docker-compose logs -f backend

# Только Neo4j
docker-compose logs -f neo4j
```

**Локальный запуск:**
```
# Логи приложения
tail -f logs/app.log
```

---

## 📊 Мониторинг

### Neo4j Browser
- URL: http://localhost:7474
- Login: `neo4j`
- Password: `password123`

Примеры запросов:
```
// Посмотреть все узлы
MATCH (n) RETURN n LIMIT 25

// Статистика
MATCH (d:Document) RETURN count(d) as total_documents
MATCH (c:Chunk) RETURN count(c) as total_chunks

// Документы с чанками
MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
RETURN d.title, count(c) as chunks_count
```

### Redis CLI
```
# Подключиться к Redis
docker exec -it redis-cache redis-cli

# Проверить ключи
KEYS semantic_cache:*

# Размер кэша
ZCARD semantic_cache:embeddings

# Статистика
HGETALL semantic_cache:stats
```

---

## 🧪 Тестирование

```
# Установить dev зависимости
uv add --dev pytest pytest-asyncio httpx

# Запустить тесты
uv run pytest tests/ -v

# С coverage
uv run pytest tests/ --cov=neo4jrag --cov-report=html
```

---

## 🐛 Troubleshooting

### Neo4j не подключается
```
# Проверить статус
docker ps | grep neo4j

# Проверить логи
docker logs neo4j-local

# Перезапустить
docker restart neo4j-local
```

### Ollama не отвечает
```
# Проверить статус
ollama list

# Перезапустить
brew services restart ollama

# Проверить порт
lsof -i :11434
```

### Redis недоступен
```
# Проверить статус
docker ps | grep redis

# Тест подключения
docker exec -it redis-cache redis-cli ping
# Должен вернуть: PONG
```

### Кэш не работает
```
# Проверить настройки
curl http://localhost:8000/api/v1/cache/health

# Очистить кэш
curl -X DELETE http://localhost:8000/api/v1/cache/clear
```

---

## 📖 Документация API

Полная интерактивная документация доступна после запуска:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

---

## 🤝 Разработка

### Структура веток
- `main` - production-ready код
- `develop` - разработка новых фич
- `feature/*` - отдельные фичи

### Commit conventions
```
feat: Добавить семантическое кэширование
fix: Исправить ошибку в векторном поиске
docs: Обновить README
refactor: Реорганизовать структуру проекта
test: Добавить тесты для RAG pipeline
```

---

## 📝 Лицензия

MIT License

