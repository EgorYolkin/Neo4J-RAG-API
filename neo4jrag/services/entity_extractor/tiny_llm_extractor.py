"""
Ultra-fast entity extraction with tiny LLMs
"""
from typing import List, Dict
from langchain_ollama import OllamaLLM
import logging

logger = logging.getLogger(__name__)


class TinyLLMEntityExtractor:
    """
    Извлечение сущностей с помощью крошечных LLM (2-3GB, работают быстро)
    
    Рекомендуемые модели для Ollama:
    - qwen2:1.5b (1.5B параметров, ~1GB)
    - phi3:mini (3.8B параметров, ~2.3GB)
    - gemma:2b (2B параметров, ~1.7GB)
    """
    
    def __init__(self, model: str = "qwen2:1.5b"):
        self.llm = OllamaLLM(
            model=model,
            base_url="http://localhost:11434",
            temperature=0.1,  # Низкая температура для точности
            num_ctx=2048      # Меньший контекст = быстрее
        )
        logger.info(f"✓ Loaded tiny LLM: {model}")
    
    def extract_entities_fast(self, text: str) -> Dict:
        """
        Быстрое извлечение сущностей (3-5 секунд на чанк)
        """
        # Сокращённый промпт для скорости
        prompt = f"""Extract entities from text. Format: name|type

Types: PERSON, LOCATION, ORG, EVENT, CONCEPT

Text: {text[:1000]}

Entities:"""
        
        response = self.llm.invoke(prompt)
        
        # Парсинг простого формата
        entities = []
        for line in response.strip().split("\n"):
            if "|" in line:
                parts = line.split("|")
                if len(parts) >= 2:
                    entities.append({
                        "name": parts[0].strip(),
                        "type": parts[1].strip().upper()
                    })
        
        return {"entities": entities, "relationships": []}
