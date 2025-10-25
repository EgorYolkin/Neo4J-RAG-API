"""
API Integration Tests
"""
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health_check():
    """Тест health endpoint"""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "components" in data


def test_create_and_query_document():
    """Тест создания документа и запроса"""
    # Создание документа
    doc_data = {
        "title": "Test Document",
        "content": "This is a test document about machine learning and AI.",
        "metadata": {"author": "Test"}
    }
    
    create_response = client.post("/api/v1/documents/", json=doc_data)
    assert create_response.status_code == 201
    doc = create_response.json()
    assert doc["title"] == "Test Document"
    
    # Запрос
    query_data = {
        "question": "What is this document about?",
        "top_k": 3
    }
    
    query_response = client.post("/api/v1/query/", json=query_data)
    assert query_response.status_code == 200
    result = query_response.json()
    assert "answer" in result
    assert len(result["sources"]) > 0


def test_statistics():
    """Тест статистики"""
    response = client.get("/api/v1/stats/")
    assert response.status_code == 200
    stats = response.json()
    assert "nodes" in stats
    assert "relationships" in stats
