from typing import Literal
from langgraph.graph import END, START, StateGraph
from src.config import settings
from src.nodes import evaluate_node, planner_node, search_node, writer_node
from src.state import ResearchState


def route_after_evaluate(state: ResearchState) -> Literal["search", "writer"]:
    """
    Conditional Edge после узла evaluate:
    - Если информации недостаточно, есть уточняющие запросы в `pending_queries`,
      и количество итераций рефлексии не превысило лимит (<= max_iterations, т.е. максимум 2 доп. цикла),
      возвращаемся в узел `search`.
    - Иначе переходим к формированию финального отчета в узел `writer`.
    """
    is_sufficient = state.get("is_sufficient", False)
    iteration = state.get("iteration", 0)
    pending_queries = state.get("pending_queries", [])

    if not is_sufficient and pending_queries and iteration <= settings.max_iterations:
        return "search"
    return "writer"


def build_research_graph():
    """
    Собирает и компилирует LangGraph агента исследования:
    START -> planner -> search -> evaluate --(conditional)--> [search | writer] -> END
    """
    workflow = StateGraph(ResearchState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("search", search_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("writer", writer_node)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "search")
    workflow.add_edge("search", "evaluate")

    workflow.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {
            "search": "search",
            "writer": "writer",
        },
    )

    workflow.add_edge("writer", END)

    return workflow.compile()


research_graph = build_research_graph()
