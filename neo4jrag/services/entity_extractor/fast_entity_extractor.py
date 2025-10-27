"""
Fast Entity Extraction using spaCy
"""
from typing import List, Dict, Tuple
import spacy
import logging

from neo4jrag.services.neo4j.neo4j_connector import Neo4jConnector  # ✅ Полный импорт

logger = logging.getLogger(__name__)


class FastEntityExtractor:
    """Быстрое извлечение сущностей с помощью spaCy"""
    
    def __init__(self, neo4j_connector: Neo4jConnector, language: str = "ru"):
        self.connector = neo4j_connector
        
        try:
            if language == "ru":
                # Пробуем загрузить модели по убыванию размера
                try:
                    self.nlp = spacy.load("ru_core_news_lg")  # Большая (лучше)
                    logger.info("✓ Loaded spaCy ru_core_news_lg")
                except OSError:
                    try:
                        self.nlp = spacy.load("ru_core_news_md")  # Средняя
                        logger.info("✓ Loaded spaCy ru_core_news_md")
                    except OSError:
                        self.nlp = spacy.load("ru_core_news_sm")  # Маленькая (хуже)
                        logger.info("✓ Loaded spaCy ru_core_news_sm")
            else:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info(f"✓ Loaded spaCy model for {language}")
        except OSError:
            logger.error(f"No spaCy model found. Install: python -m spacy download ru_core_news_md")
            raise

    
    def extract_entities_fast(self, text: str) -> List[Dict]:
        """Быстрое извлечение сущностей"""
        doc = self.nlp(text)
        
        entities = []
        seen = set()
        
        for ent in doc.ents:
            entity_name = ent.text.strip()
            entity_type = self.entity_mapping.get(ent.label_, "CONCEPT")
            
            if entity_name in seen or len(entity_name) < 2:
                continue
            
            seen.add(entity_name)
            
            entities.append({
                "name": entity_name,
                "type": entity_type,
                "description": f"Тип: {ent.label_}"
            })
        
        logger.info(f"Extracted {len(entities)} entities")
        return entities
    
    def extract_relationships_simple(self, text: str, entities: List[Dict]) -> List[Dict]:
        """Простое извлечение отношений"""
        doc = self.nlp(text)
        relationships = []
        
        entity_positions = []
        for ent in doc.ents:
            entity_name = ent.text.strip()
            entity_positions.append({
                "name": entity_name,
                "start": ent.start_char,
                "end": ent.end_char
            })
        
        proximity_threshold = 100
        
        for i, ent1 in enumerate(entity_positions):
            for ent2 in entity_positions[i+1:]:
                distance = abs(ent1["start"] - ent2["start"])
                
                if distance <= proximity_threshold:
                    relationships.append({
                        "source": ent1["name"],
                        "target": ent2["name"],
                        "type": "RELATED_TO",
                        "description": "Упоминаются вместе"
                    })
        
        logger.info(f"Extracted {len(relationships)} relationships")
        return relationships
    
    def create_knowledge_graph(
        self,
        text: str,
        document_id: str,
        user_id: str
    ) -> Tuple[int, int]:
        """Создание графа знаний"""
        entities = self.extract_entities_fast(text)
        relationships = self.extract_relationships_simple(text, entities)
        
        entities_count = 0
        for entity in entities:
            entity_name = entity["name"]
            entity_type = entity["type"]
            
            query = f"""
            MERGE (e:Entity {{name: $name, user_id: $user_id}})
            ON CREATE SET 
                e.type = $type,
                e.created_at = datetime()
            SET e:{entity_type}
            
            WITH e
            MATCH (d:Document {{id: $doc_id}})
            MERGE (d)-[:MENTIONS]->(e)
            
            RETURN e
            """
            
            try:
                self.connector.execute_write(query, {
                    "name": entity_name,
                    "user_id": user_id,
                    "type": entity_type,
                    "doc_id": document_id
                })
                entities_count += 1
            except Exception as e:
                logger.error(f"Error creating entity: {e}")
        
        relationships_count = 0
        for rel in relationships:
            source = rel["source"]
            target = rel["target"]
            rel_type = rel["type"]
            
            query = f"""
            MATCH (s:Entity {{name: $source, user_id: $user_id}})
            MATCH (t:Entity {{name: $target, user_id: $user_id}})
            MERGE (s)-[r:{rel_type}]->(t)
            ON CREATE SET r.created_at = datetime()
            RETURN r
            """
            
            try:
                self.connector.execute_write(query, {
                    "source": source,
                    "target": target,
                    "user_id": user_id
                })
                relationships_count += 1
            except Exception:
                pass
        
        logger.info(f"Created {entities_count} entities and {relationships_count} relationships")
        return entities_count, relationships_count
