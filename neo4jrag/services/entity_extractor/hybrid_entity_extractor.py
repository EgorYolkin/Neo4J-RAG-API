from typing import List, Dict
from langchain_ollama import OllamaLLM
from neo4jrag.services.entity_extractor.fast_entity_extractor import FastEntityExtractor


class HybridEntityExtractor:
    """
    Гибридный экстрактор:
    1. spaCy для быстрого извлечения сущностей (миллисекунды)
    2. Tiny LLM для извлечения отношений (секунды)
    """
    
    def __init__(self, neo4j_connector, language="ru"):
        # spaCy для сущностей
        self.fast_extractor = FastEntityExtractor(neo4j_connector, language)
        
        # Tiny LLM для отношений
        self.tiny_llm = OllamaLLM(
            model="qwen2:1.5b",
            base_url="http://localhost:11434",
            temperature=0.1
        )
    
    def extract_all(self, text: str) -> Dict:
        """
        Быстрое извлечение:
        - Сущности: spaCy (мгновенно)
        - Отношения: tiny LLM (3-5 сек)
        """
        # 1. Быстро извлекаем сущности
        entities = self.fast_extractor.extract_entities_fast(text)
        
        # 2. Извлекаем отношения между найденными сущностями
        entity_names = [e["name"] for e in entities]
        
        if len(entity_names) > 1:
            prompt = f"""Find relationships between entities.

Entities: {", ".join(entity_names[:10])}  # Макс 10 для скорости

Text: {text[:500]}

Relationships (format: entity1->RELATION->entity2):"""
            
            response = self.tiny_llm.invoke(prompt)
            relationships = self._parse_relationships(response, entity_names)
        else:
            relationships = []
        
        return {
            "entities": entities,
            "relationships": relationships
        }
    
    def _parse_relationships(self, response: str, valid_entities: List[str]) -> List[Dict]:
        """Парсинг отношений из ответа"""
        relationships = []
        
        for line in response.strip().split("\n"):
            if "->" in line:
                parts = line.split("->")
                if len(parts) == 3:
                    source = parts[0].strip()
                    rel_type = parts[1].strip()
                    target = parts[2].strip()
                    
                    # Проверяем, что сущности валидны
                    if source in valid_entities and target in valid_entities:
                        relationships.append({
                            "source": source,
                            "target": target,
                            "type": rel_type.upper().replace(" ", "_")
                        })
        
        return relationships
