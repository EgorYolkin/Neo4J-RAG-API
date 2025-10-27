"""
Knowledge Graph endpoints для работы с сущностями
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query as FastAPIQuery
from typing import List, Dict, Optional

from neo4jrag.services.neo4j.neo4j_connector import Neo4jConnector
from neo4jrag.services.entity_extractor.hybrid_entity_extractor import HybridEntityExtractor
from neo4jrag.api.deps import get_neo4j_connector, get_components

router = APIRouter()


@router.get("/entities", response_model=List[Dict])
async def list_entities(
    user_id: int = None,
    entity_type: Optional[str] = FastAPIQuery(None, description="Фильтр по типу (PERSON, LOCATION, etc.)"),
    limit: int = FastAPIQuery(50, ge=1, le=200),
    neo4j_connector: Neo4jConnector = Depends(get_neo4j_connector)
) -> List[Dict]:
    """Получение всех сущностей пользователя"""
    try:
        if entity_type:
            query = """
            MATCH (e:Entity {user_id: $user_id, type: $type})
            OPTIONAL MATCH (d:Document)-[:MENTIONS]->(e)
            RETURN e.name as name, 
                   e.type as type, 
                   e.description as description,
                   count(DISTINCT d) as mentioned_in_docs
            ORDER BY e.name
            LIMIT $limit
            """
            results = neo4j_connector.execute_query(query, {
                "user_id": user_id,
                "type": entity_type.upper(),
                "limit": limit
            })
        else:
            query = """
            MATCH (e:Entity {user_id: $user_id})
            OPTIONAL MATCH (d:Document)-[:MENTIONS]->(e)
            RETURN e.name as name, 
                   e.type as type, 
                   e.description as description,
                   count(DISTINCT d) as mentioned_in_docs
            ORDER BY e.type, e.name
            LIMIT $limit
            """
            results = neo4j_connector.execute_query(query, {
                "user_id": user_id,
                "limit": limit
            })
        
        return results
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get entities: {str(e)}"
        )


@router.get("/relationships", response_model=List[Dict])
async def list_relationships(
    user_id: int = None,
    entity_name: Optional[str] = FastAPIQuery(None, description="Фильтр по имени сущности"),
    limit: int = FastAPIQuery(50, ge=1, le=200),
    neo4j_connector: Neo4jConnector = Depends(get_neo4j_connector)
) -> List[Dict]:
    """Получение связей между сущностями"""
    try:
        if entity_name:
            query = """
            MATCH (e1:Entity {user_id: $user_id, name: $name})-[r]->(e2:Entity)
            RETURN e1.name as source, 
                   type(r) as relationship, 
                   e2.name as target,
                   e1.type as source_type,
                   e2.type as target_type
            LIMIT $limit
            """
            results = neo4j_connector.execute_query(query, {
                "user_id": user_id,
                "name": entity_name,
                "limit": limit
            })
        else:
            query = """
            MATCH (e1:Entity {user_id: $user_id})-[r]->(e2:Entity)
            RETURN e1.name as source, 
                   type(r) as relationship, 
                   e2.name as target,
                   e1.type as source_type,
                   e2.type as target_type
            LIMIT $limit
            """
            results = neo4j_connector.execute_query(query, {
                "user_id": user_id,
                "limit": limit
            })
        
        return results
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get relationships: {str(e)}"
        )


@router.get("/graph-stats", response_model=Dict)
async def get_graph_statistics(
    user_id: int = None,
    neo4j_connector: Neo4jConnector = Depends(get_neo4j_connector)
) -> Dict:
    """Статистика графа знаний пользователя"""
    try:
        # Общая статистика
        query = """
        MATCH (e:Entity {user_id: $user_id})
        OPTIONAL MATCH (e)-[r]->()
        WITH count(DISTINCT e) as total_entities,
             count(DISTINCT r) as total_relationships,
             collect(DISTINCT e.type) as entity_types
        RETURN total_entities, total_relationships, entity_types
        """
        
        result = neo4j_connector.execute_query(query, {"user_id": user_id})
        
        if result:
            stats = result[0]
            
            # Статистика по типам
            type_stats_query = """
            MATCH (e:Entity {user_id: $user_id})
            RETURN e.type as type, count(e) as count
            ORDER BY count DESC
            """
            type_stats = neo4j_connector.execute_query(type_stats_query, {"user_id": user_id})
            
            return {
                "total_entities": stats["total_entities"],
                "total_relationships": stats["total_relationships"],
                "entity_types": stats["entity_types"],
                "entities_by_type": {item["type"]: item["count"] for item in type_stats}
            }
        
        return {
            "total_entities": 0,
            "total_relationships": 0,
            "entity_types": [],
            "entities_by_type": {}
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get statistics: {str(e)}"
        )


@router.get("/entity/{entity_name}", response_model=Dict)
async def get_entity_details(
    user_id: int,
    entity_name: str,
    neo4j_connector: Neo4jConnector = Depends(get_neo4j_connector)
) -> Dict:
    """Детальная информация о сущности"""
    try:

        query = """
        MATCH (e:Entity {user_id: $user_id, name: $name})
        OPTIONAL MATCH (e)-[r]->(e2:Entity)
        OPTIONAL MATCH (d:Document)-[:MENTIONS]->(e)
        RETURN e.name as name,
               e.type as type,
               e.description as description,
               collect(DISTINCT {target: e2.name, relation: type(r)}) as relationships,
               collect(DISTINCT d.title) as mentioned_in
        LIMIT 1
        """
        
        result = neo4j_connector.execute_query(query, {
            "user_id": user_id,
            "name": entity_name
        })
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Entity '{entity_name}' not found"
            )
        
        return result[0]
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get entity details: {str(e)}"
        )


@router.post("/test-extraction", response_model=Dict)
async def test_entity_extraction(
    user_id: int,
    text: str = FastAPIQuery(..., min_length=10),
    components: Dict = Depends(get_components)
) -> Dict:
    """ТЕСТ: Проверка работы entity extractor"""
    try:
        entity_extractor = components.get("entity_extractor")
        
        if entity_extractor is None:
            return {
                "status": "error",
                "message": "Entity extractor not initialized",
                "extractor_type": None
            }
        
        # Пробуем извлечь сущности
        entities = entity_extractor.extract_entities_fast(text)
        
        return {
            "status": "success",
            "extractor_type": type(entity_extractor).__name__,
            "text_length": len(text),
            "entities_found": len(entities),
            "entities": entities
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "error_type": type(e).__name__
        }