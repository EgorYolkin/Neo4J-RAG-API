"""
Statistics endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict

from neo4jrag.api.deps import get_neo4j_connector
from neo4jrag.domain.schemas.response import StatsResponse, GraphSchemaResponse
from neo4jrag.services.neo4j.neo4j_connector import Neo4jConnector

router = APIRouter()


@router.get("/", response_model=StatsResponse)
async def get_statistics(
    neo4j_connector: Neo4jConnector = Depends(get_neo4j_connector)
) -> StatsResponse:
    """Получение статистики графа"""
    try:
        stats = neo4j_connector.get_statistics()
        
        return StatsResponse(
            nodes=stats.get("nodes", {}),
            relationships=stats.get("relationships", {}),
            total_documents=stats.get("nodes", {}).get("Document", 0),
            total_chunks=stats.get("nodes", {}).get("Chunk", 0)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get statistics: {str(e)}"
        )


@router.get("/schema", response_model=GraphSchemaResponse)
async def get_graph_schema(
    neo4j_connector: Neo4jConnector = Depends(get_neo4j_connector)
) -> GraphSchemaResponse:
    """Получение схемы графа"""
    try:
        # Labels
        labels_query = "CALL db.labels() YIELD label RETURN collect(label) as labels"
        labels_result = neo4j_connector.execute_query(labels_query)
        labels = labels_result[0]["labels"] if labels_result else []
        
        # Relationship types
        rel_types_query = "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as types"
        rel_result = neo4j_connector.execute_query(rel_types_query)
        relationship_types = rel_result[0]["types"] if rel_result else []
        
        # Constraints
        constraints_query = "SHOW CONSTRAINTS YIELD name, type RETURN collect({name: name, type: type}) as constraints"
        constraints_result = neo4j_connector.execute_query(constraints_query)
        constraints = constraints_result[0]["constraints"] if constraints_result else []
        
        # Indexes
        indexes_query = "SHOW INDEXES YIELD name, type RETURN collect({name: name, type: type}) as indexes"
        indexes_result = neo4j_connector.execute_query(indexes_query)
        indexes = indexes_result[0]["indexes"] if indexes_result else []
        
        return GraphSchemaResponse(
            node_labels=labels,
            relationship_types=relationship_types,
            constraints=constraints,
            indexes=indexes
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get schema: {str(e)}"
        )


@router.get("/embeddings", response_model=Dict)
async def get_embeddings_stats(
    neo4j_connector: Neo4jConnector = Depends(get_neo4j_connector)
) -> Dict:
    """Статистика по эмбеддингам"""
    try:
        query = """
        MATCH (c:Chunk)
        RETURN 
            count(c) as total_chunks,
            count(c.embedding) as chunks_with_embeddings,
            count(CASE WHEN c.embedding IS NULL THEN 1 END) as chunks_without_embeddings
        """
        
        result = neo4j_connector.execute_query(query)
        
        if not result:
            return {
                "total_chunks": 0,
                "chunks_with_embeddings": 0,
                "chunks_without_embeddings": 0,
                "coverage_percentage": 0.0
            }
        
        stats = result[0]
        total = stats["total_chunks"]
        with_emb = stats["chunks_with_embeddings"]
        coverage = (with_emb / total * 100) if total > 0 else 0.0
        
        return {
            "total_chunks": total,
            "chunks_with_embeddings": with_emb,
            "chunks_without_embeddings": stats["chunks_without_embeddings"],
            "coverage_percentage": round(coverage, 2)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get embeddings stats: {str(e)}"
        )
