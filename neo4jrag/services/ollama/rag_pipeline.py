from typing import TypedDict, List, Dict, Annotated
from operator import add
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from neo4jrag.services.neo4j.vector_store import VectorStore
from neo4jrag.services.ollama.ollama_loader import OllamaLoader
import logging

logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    """Состояние графа"""
    question: str
    search_type: str
    context: Annotated[List[Dict], add]
    answer: str
    steps: Annotated[List[str], add]


class RAGPipeline:
    """RAG пайплайн с LangGraph"""
    
    def __init__(self, vector_store: VectorStore, ollama: OllamaLoader):
        self.vector_store = vector_store
        self.ollama = ollama
        self.app = self._build_workflow()
    
    def _route_question(self, state: GraphState) -> GraphState:
        """Маршрутизация запроса"""
        question = state["question"].lower()
        
        if any(kw in question for kw in ["что такое", "объясни", "расскажи"]):
            state["search_type"] = "vector"
            state["steps"].append("Маршрут: Векторный поиск")
        else:
            state["search_type"] = "hybrid"
            state["steps"].append("Маршрут: Гибридный поиск")
        
        return state
    
    def _vector_search(self, state: GraphState) -> GraphState:
        """Векторный поиск"""
        results = self.vector_store.similarity_search(state["question"], k=3)
        state["context"].extend(results)
        state["steps"].append(f"Векторный поиск: {len(results)} результатов")
        return state
    
    def _hybrid_search(self, state: GraphState) -> GraphState:
        """Гибридный поиск"""
        results = self.vector_store.hybrid_search(state["question"], k=3)
        state["context"].extend(results)
        state["steps"].append(f"Гибридный поиск: {len(results)} результатов")
        return state
    
    def _generate_answer(self, state: GraphState) -> GraphState:
        """Генерация ответа"""
        context_text = "\n\n".join([
            f"Источник {idx+1}:\n{ctx['text']}"
            for idx, ctx in enumerate(state["context"])
        ])
        
        prompt = ChatPromptTemplate.from_template("""
        Ответь на вопрос на основе контекста. Будь кратким и точным.

        Контекст:
        {context}

        Вопрос: {question}

        Ответ:
        """)
        
        chain = prompt | self.ollama.llm
        answer = chain.invoke({
            "context": context_text,
            "question": state["question"]
        })
        
        state["answer"] = answer
        state["steps"].append("Ответ сгенерирован")
        return state
    
    def _should_continue(self, state: GraphState) -> str:
        """Определение следующего шага"""
        return state["search_type"]
    
    def _build_workflow(self) -> StateGraph:
        """Построение LangGraph workflow"""
        workflow = StateGraph(GraphState)
        
        workflow.add_node("router", self._route_question)
        workflow.add_node("vector_search", self._vector_search)
        workflow.add_node("hybrid_search", self._hybrid_search)
        workflow.add_node("generate", self._generate_answer)
        
        workflow.set_entry_point("router")
        
        workflow.add_conditional_edges(
            "router",
            self._should_continue,
            {
                "vector": "vector_search",
                "hybrid": "hybrid_search"
            }
        )
        
        workflow.add_edge("vector_search", "generate")
        workflow.add_edge("hybrid_search", "generate")
        workflow.add_edge("generate", END)
        
        return workflow.compile()
    
    def ask(self, question: str, verbose: bool = True) -> Dict:
        """Задать вопрос системе"""
        initial_state = GraphState(
            question=question,
            search_type="",
            context=[],
            answer="",
            steps=[]
        )
        
        result = self.app.invoke(initial_state)
        
        if verbose:
            print("\n" + "="*80)
            print(f"ВОПРОС: {question}")
            print("="*80)
            print("\nШаги:")
            for step in result["steps"]:
                print(f"  • {step}")
            print("\n" + "-"*80)
            print(f"ОТВЕТ:\n{result['answer']}")
            print("="*80 + "\n")
        
        return result
