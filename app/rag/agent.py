from typing import Annotated, List, Dict
from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, START, END, add_messages
from langchain.tools import tool
from neo4j import GraphDatabase
from app.graph.node import get_node_info
from app.config import NEO4J_USER, NEO4J_PASSWORD, NEO4J_URI
from langsmith import traceable

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    graph_payload: dict
    llm_calls: int

def format_graph_to_string(payload: dict) -> str:
    nodes = payload.get("nodes", [])
    if not nodes: return "Граф сейчас пуст."
    
    lines = []
    for n in nodes:
        info = n.get("graph_info", {})
        rels = []
        for out in info.get("outgoing", []):
            rels.append(f"RELATION: {out.get('type')} -> TARGET: {out.get('target')}")
        for inc in info.get("incoming", []):
            rels.append(f"RELATION: {inc.get('type')} <- SOURCE: {inc.get('source')}")
        
        node_block = [
            f"=== УЗЕЛ: {info.get('title')} (ID: {n['id']}) ===",
            f"ОПИСАНИЕ: {info.get('text', '')}"
        ]
        if rels:
            node_block.append("ДОСТУПНЫЕ СВЯЗИ:")
            node_block.extend(rels)
        lines.append("\n".join(node_block))
    return "\n\n" + "\n---\n".join(lines)


@tool
def delete_nodes(node_ids: List[str]):
    """
    Оптимизация контекста: удаляет узлы, которые не содержат ответа на текущий вопрос.
    Используй для очистки графа от побочных веток, не связанных с вопросом.
    """
    return {"action": "delete", "ids": node_ids}

@tool
def expand_nodes_via_relation(source_node_title: str, relation_type: str):
    """
    Получает данные о связанном узле из Neo4j.
    Аргументы: 
    - source_node_title: Название узла (из блока === УЗЕЛ: ... ===)
    - relation_type: Тип из списка "ДОСТУПНЫЕ СВЯЗИ" (например, 'ВРАГ', 'РОДНОЙ_МИР').
    """
    clean_rel = relation_type.strip("()[]'\" ").upper()
    
    with driver.session() as session:
        query = """
        MATCH (n {title: $source_title})-[r]-(m)
        WHERE type(r) = $rel_type OR type(r) = toUpper($rel_type)
        RETURN m.title AS target_title LIMIT 1
        """
        result = session.run(query, source_title=source_node_title, rel_type=clean_rel).single()
        
        if not result:
            return {"action": "expand", "new_nodes": [], "status": "Связь не найдена"}
            
        target_title = result["target_title"]
        node_data = get_node_info(target_title, detailed=True)
        
        new_node = {
            "id": f"node_{node_data['title'].replace(' ', '_')}",
            "graph_info": node_data
        }
        return {"action": "expand", "new_nodes": [new_node]}
    
class GraphContextOptimizer:
    def __init__(self, model, max_iterations: int = 5):
        self.tools = [delete_nodes, expand_nodes_via_relation]
        self.model_with_tools = model.bind_tools(self.tools)
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.max_iterations = max_iterations
        self.graph = self._build_graph()

    def _llm_node(self, state: GraphState):
        """Логика принятия решения: удалить, расширить или закончить."""
        user_query = state["messages"][0].content
        graph_text = format_graph_to_string(state["graph_payload"])
        
        system_msg = SystemMessage(content=f"""
ТЫ — ГРАФОВЫЙ РЕДАКТОР. 
Твоя задача: оставить в контексте только то, что касается темы "{user_query}", и найти недостающие факты.

ТЕКУЩИЙ СПИСОК УЗЛОВ:
{graph_text}

АЛГОРИТМ РАБОТЫ:
1. ПРОВЕРКА НА МУСОР: Просмотри список. Если узел не упоминается в контексте "{user_query}", вызови delete_nodes.
   Пример: Если ищем Тразина, а видим узел "Дэн Абнетт" или "Мортис (роман)" — УДАЛЯЙ ИХ.
2. ПРОВЕРКА НА ПОЛНОТУ: Если в оставшихся узлах нет четкого ответа на вопрос, выбери самую перспективную связь из "ДОСТУПНЫЕ СВЯЗИ" и используй expand_nodes_via_relation.
3. ИТОГ: Если граф содержит ТОЛЬКО исчерпывающую информацию для ответа, просто напиши слово "ГОТОВО" (без вызова инструментов).

ВАЖНО: Оставляй только те узлы, где в поле НАЗВАНИЕ или ОПИСАНИЕ встречается "{user_query}". Все остальное — лишний шум.
""")
        if state.get("llm_calls", 0) >= self.max_iterations:
            return {"messages": [AIMessage(content="ГОТОВО")]}

        response = self.model_with_tools.invoke([system_msg] + state["messages"])
        return {"messages": [response], "llm_calls": state.get("llm_calls", 0) + 1}

    def _tools_node(self, state: GraphState):
        """Выполнение инструментов и обновление payload."""
        last_msg = state["messages"][-1]
        payload = state["graph_payload"].copy()
        new_messages = []

        for call in last_msg.tool_calls:
            tool_obj = self.tools_by_name[call["name"]]
            result = tool_obj.invoke(call["args"])
            
            if result["action"] == "delete":
                target_ids = result["ids"]
                payload["nodes"] = [n for n in payload["nodes"] if n["id"] not in target_ids]
                obs = f"Удалено узлов: {len(target_ids)}"
            
            elif result["action"] == "expand":
                existing_titles = {n["graph_info"]["title"] for n in payload["nodes"]}
                added = 0
                for nn in result["new_nodes"]:
                    if nn["graph_info"]["title"] not in existing_titles:
                        payload["nodes"].append(nn)
                        added += 1
                obs = f"Добавлено узлов: {added}"

            new_messages.append(ToolMessage(content=obs, tool_call_id=call["id"]))

        return {"messages": new_messages, "graph_payload": payload}

    def _router(self, state: GraphState):
        """Определяет, нужно ли идти в инструменты."""
        if state["messages"][-1].tool_calls:
            return "tools"
        return END

    def _build_graph(self):
        builder = StateGraph(GraphState)
        builder.add_node("llm", self._llm_node)
        builder.add_node("tools", self._tools_node)
        
        builder.add_edge(START, "llm")
        builder.add_conditional_edges("llm", self._router, {"tools": "tools", END: END})
        builder.add_edge("tools", "llm")
        return builder.compile()

    def optimize(self, query: str, initial_payload: dict) -> dict:
        state = {
            "messages": [HumanMessage(content=query)],
            "graph_payload": initial_payload,
            "llm_calls": 0
        }
        result = self.graph.invoke(state)
        return result["graph_payload"]