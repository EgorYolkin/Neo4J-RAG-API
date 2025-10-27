"""
LLM-based Entity Extraction с пуленепробиваемым парсингом
"""
from typing import List, Dict, Tuple
import logging
import json
import re

from langchain_ollama import OllamaLLM
from neo4jrag.services.neo4j.neo4j_connector import Neo4jConnector

logger = logging.getLogger(__name__)


class LLMEntityExtractor:
    """Извлечение сущностей с помощью LLM"""
    
    def __init__(
        self, 
        neo4j_connector: Neo4jConnector, 
        model: str = "qwen2:1.5b",
        language: str = "ru"
    ):
        self.connector = neo4j_connector
        self.language = language
        
        try:
            self.llm = OllamaLLM(
                model=model,
                base_url="http://localhost:11434",
                temperature=0.0,  # ✅ Нулевая температура для детерминированности
                num_ctx=2048
            )
            logger.info(f"✓ Loaded LLM for entity extraction: {model}")
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            raise
        
        # Простейший промпт со строгим форматом
        self.extraction_prompt = """Extract person names, locations, and organizations from the Russian text.

Reply ONLY with this format (one per line):
NAME|TYPE

Where TYPE is: PERSON, LOCATION, or ORGANIZATION

Example:
Иван Иванов|PERSON
Москва|LOCATION
Google|ORGANIZATION

Text: {text}

Entities:"""
    
    def extract_entities_fast(self, text: str) -> List[Dict]:
        """Извлечение сущностей с множественными стратегиями парсинга"""
        prompt = self.extraction_prompt.format(text=text[:1000])  # Ограничиваем длину
        
        try:
            response = self.llm.invoke(prompt)
            logger.info(f"LLM raw response:\n{response}")
            
            # Стратегия 1: Парсинг построчного формата (ПРОСТЕЙШИЙ И НАДЁЖНЫЙ)
            entities = self._parse_line_format(response)
            if entities:
                logger.info(f"✓ Extracted {len(entities)} entities via line format")
                return entities
            
            # Стратегия 2: Парсинг JSON массива
            entities = self._parse_json_array(response)
            if entities:
                logger.info(f"✓ Extracted {len(entities)} entities via JSON array")
                return entities
            
            # Стратегия 3: Парсинг JSON объекта
            entities = self._parse_json_object(response)
            if entities:
                logger.info(f"✓ Extracted {len(entities)} entities via JSON object")
                return entities
            
            # Стратегия 4: Regex extraction как последняя надежда
            entities = self._parse_with_regex(text)
            if entities:
                logger.info(f"✓ Extracted {len(entities)} entities via regex fallback")
                return entities
            
            logger.warning("All parsing strategies failed, no entities extracted")
            return []
            
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}", exc_info=True)
            # Fallback на regex
            return self._parse_with_regex(text)
    
    def _parse_line_format(self, response: str) -> List[Dict]:
        """Парсинг формата: Name|TYPE"""
        entities = []
        
        for line in response.strip().split('\n'):
            line = line.strip()
            if '|' not in line:
                continue
            
            parts = line.split('|')
            if len(parts) >= 2:
                name = parts[0].strip()
                entity_type = parts[1].strip().upper()
                
                if name and entity_type in ['PERSON', 'LOCATION', 'ORGANIZATION', 'DATE', 'CONCEPT']:
                    entities.append({
                        "name": name,
                        "type": entity_type,
                        "description": f"Тип: {entity_type}"
                    })
        
        return entities
    
    def _parse_json_array(self, response: str) -> List[Dict]:
        """Парсинг JSON массива"""
        try:
            # Очистка
            cleaned = response.strip()
            cleaned = re.sub(r'```\s*', '', cleaned)
            
            # Ищем массив
            array_match = re.search(r'\[.*\]', cleaned, re.DOTALL)
            if array_match:
                entities = json.loads(array_match.group(0))
                
                # Валидация и добавление description
                valid_entities = []
                for entity in entities:
                    if isinstance(entity, dict) and 'name' in entity and 'type' in entity:
                        entity["description"] = f"Тип: {entity.get('type', 'UNKNOWN')}"
                        valid_entities.append(entity)
                
                return valid_entities
        except Exception as e:
            logger.debug(f"JSON array parsing failed: {e}")
        
        return []
    
    def _parse_json_object(self, response: str) -> List[Dict]:
        """Парсинг JSON объекта с полем entities"""
        try:
            cleaned = response.strip()
            cleaned = re.sub(r'```\s*', '', cleaned)
            
            # Ищем объект
            object_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if object_match:
                result = json.loads(object_match.group(0))
                entities = result.get("entities", [])
                
                for entity in entities:
                    if isinstance(entity, dict):
                        entity["description"] = f"Тип: {entity.get('type', 'UNKNOWN')}"
                
                return entities
        except Exception as e:
            logger.debug(f"JSON object parsing failed: {e}")
        
        return []
    
    def _parse_with_regex(self, text: str) -> List[Dict]:
        """
        Fallback: Простая regex экстракция имён собственных
        (заглавная буква + слова)
        """
        entities = []
        
        # Ищем слова с заглавной буквы (потенциальные имена собственные)
        pattern = r'\b[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)*\b'
        matches = re.findall(pattern, text)
        
        # Дедупликация и фильтрация
        seen = set()
        common_words = {'Привет', 'Доброе', 'Это', 'Все', 'Может', 'Надо', 'Будет'}
        
        for match in matches:
            if match not in seen and match not in common_words and len(match) > 3:
                seen.add(match)
                entities.append({
                    "name": match,
                    "type": "CONCEPT",  # По умолчанию CONCEPT
                    "description": "Извлечено с помощью regex"
                })
        
        return entities[:10]  # Максимум 10 сущностей
    
    def extract_relationships_simple(self, text: str, entities: List[Dict]) -> List[Dict]:
        """Извлечение связей"""
        if len(entities) < 2:
            return []
        
        relationships = []
        entity_names = [e["name"] for e in entities if isinstance(e, dict) and "name" in e]
        
        for i, ent1 in enumerate(entity_names[:5]):  # Только первые 5 для скорости
            for ent2 in entity_names[i+1:i+3]:
                if ent1 in text and ent2 in text:
                    pos1 = text.find(ent1)
                    pos2 = text.find(ent2)
                    
                    if abs(pos1 - pos2) < 200:
                        relationships.append({
                            "source": ent1,
                            "target": ent2,
                            "type": "RELATED_TO",
                            "description": "Упоминаются рядом"
                        })
        
        return relationships
    
    def create_knowledge_graph(
        self,
        text: str,
        document_id: str,
        user_id: str
    ) -> Tuple[int, int]:
        """Создание графа знаний"""
        entities = self.extract_entities_fast(text)
        
        if not entities:
            logger.warning("No entities extracted")
            return 0, 0
        
        relationships = self.extract_relationships_simple(text, entities)
        
        # Создаём узлы
        entities_count = 0
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            
            entity_name = entity.get("name", "").strip()
            if not entity_name or len(entity_name) < 2:
                continue
            
            entity_type = entity.get("type", "CONCEPT").upper()
            
            # Защита от SQL injection через параметризованные запросы
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
                logger.error(f"Error creating entity {entity_name}: {e}")
        
        # Создаём отношения
        relationships_count = 0
        for rel in relationships:
            source = rel.get("source", "").strip()
            target = rel.get("target", "").strip()
            rel_type = rel.get("type", "RELATED_TO")
            
            if not source or not target:
                continue
            
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
        
        logger.info(f"✓ Created {entities_count} entities and {relationships_count} relationships")
        return entities_count, relationships_count
