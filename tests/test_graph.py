import pytest
from unittest.mock import AsyncMock, patch
from src.evaluation import evaluate_report_deterministic
from src.graph import build_research_graph, route_after_evaluate
from src.state import EvaluationOutput, FactItem, PlanOutput


def test_conditional_routing_logic():
    """Проверяет работу условного ребра (Conditional Edge) для цикла рефлексии."""
    # Случай 1: Информации недостаточно, итерация 1 (<=2), есть follow-up вопросы -> идем в search
    state_retry = {
        "user_query": "Test",
        "questions": ["q1", "q2", "q3", "q4_followup"],
        "pending_queries": ["q4_followup"],
        "facts": [],
        "iteration": 1,
        "is_sufficient": False,
        "coverage_score": 0.55,
        "evaluation_notes": "Need benchmarks",
        "final_report": "",
        "metrics": {},
    }
    assert route_after_evaluate(state_retry) == "search"

    # Случай 2: Лимит итераций превышен или информации достаточно -> идем в writer
    state_done = {
        **state_retry,
        "is_sufficient": True,
        "pending_queries": [],
    }
    assert route_after_evaluate(state_done) == "writer"


@pytest.mark.asyncio
async def test_full_research_graph_with_reflection_cycle():
    """
    Интеграционный тест полного цикла LangGraph:
    planner (4 вопроса) -> search -> evaluate (недостаточно, итерация 1)
    -> search (доп. запрос) -> evaluate (достаточно) -> writer (Markdown со ссылками).
    """
    fake_plan = PlanOutput(
        subquestions=[
            "Архитектура LangGraph для multi-agent RAG систем",
            "Метрики оценки качества Agentic RAG в Arize Phoenix",
            "Сравнение задержек и стоимости Agentic RAG vs Naive RAG",
            "Кейсы внедрения циклов рефлексии Self-RAG и Corrective RAG",
        ],
        reasoning="Декомпозиция охватывает архитектуру, метрики, latency/cost и индустриальные паттерны.",
    )

    # На первой проверке evaluate возвращает is_sufficient=False, на второй — True
    eval_responses = [
        EvaluationOutput(
            is_sufficient=False,
            coverage_score=0.62,
            missing_aspects=["Не хватает точных цифр по снижению галлюцинаций в продакшене"],
            follow_up_queries=["Self-RAG Corrective RAG hallucination reduction benchmark numbers"],
        ),
        EvaluationOutput(
            is_sufficient=True,
            coverage_score=0.94,
            missing_aspects=[],
            follow_up_queries=[],
        ),
    ]

    async def fake_execute_batch(queries, existing_facts):
        start_id = len(existing_facts) + 1
        return [
            FactItem(
                source_id=start_id + idx,
                query=q,
                title=f"Research Source {start_id + idx}",
                url=f"https://example.org/research/{start_id + idx}",
                snippet=f"Краткий сниппет по запросу: {q}",
                scraped_content=f"Полный скрапленный текст исследования по теме '{q}' с цифрами: точность 91.4%, снижение галлюцинаций на 38%.",
                relevance_score=0.92,
            )
            for idx, q in enumerate(queries)
        ]

    class DummyResponse:
        content = (
            "# Аналитический отчет: Agentic RAG на базе LangGraph\n\n"
            "## Executive Summary\n"
            "Использование графовых агентов с циклом рефлексии повышает полноту ответа (Recall) до 91.4% [1, 2] "
            "и снижает долю галлюцинаций на 38% по сравнению с линейным Naive RAG [4, 5].\n\n"
            "## Сравнение метрик\n"
            "- Декомпозиция запроса на 3–5 подвопросов закрывает multi-hop зависимости [1, 3].\n"
            "- Мониторинг через Arize Phoenix (OpenTelemetry) позволяет отслеживать стоимость и задержки каждого узла [2].\n\n"
            "## Источники (References)\n"
            "[1] Research Source 1 — https://example.org/research/1\n"
            "[2] Research Source 2 — https://example.org/research/2\n"
            "[3] Research Source 3 — https://example.org/research/3\n"
            "[4] Research Source 4 — https://example.org/research/4\n"
            "[5] Research Source 5 — https://example.org/research/5\n"
        )

    mock_planner_llm = AsyncMock()
    mock_planner_llm.ainvoke = AsyncMock(return_value=fake_plan)

    mock_eval_llm = AsyncMock()
    mock_eval_llm.ainvoke = AsyncMock(side_effect=eval_responses)

    mock_writer_llm = AsyncMock()
    mock_writer_llm.ainvoke = AsyncMock(return_value=DummyResponse())

    class FakeLLMRouter:
        def __init__(self, is_writer: bool = False):
            self.is_writer = is_writer

        def with_structured_output(self, schema):
            if schema is PlanOutput:
                return mock_planner_llm
            return mock_eval_llm

        async def ainvoke(self, messages):
            return await mock_writer_llm.ainvoke(messages)

    with (
        patch("src.nodes.get_llm", side_effect=lambda model=None, temperature=None: FakeLLMRouter()),
        patch("src.tools.TavilySearchAndScrapeTool.execute_batch", side_effect=fake_execute_batch),
    ):
        graph = build_research_graph()
        final_state = await graph.ainvoke(
            {
                "user_query": "Как устроен Agentic RAG с рефлексией и какие метрики он дает в продакшене?",
                "questions": [],
                "pending_queries": [],
                "facts": [],
                "iteration": 0,
                "is_sufficient": False,
                "coverage_score": 0.0,
                "evaluation_notes": "",
                "final_report": "",
                "metrics": {},
            }
        )

    # Проверяем, что было сгенерировано 4 стартовых вопроса + 1 доп. вопрос на цикле рефлексии = 5 вопросов
    assert len(final_state["questions"]) == 5
    # Проверяем, что отработал ровно 1 цикл рефлексии
    assert final_state["iteration"] == 1
    # Проверяем, что собрано 5 уникальных фактов со скрапингом
    assert len(final_state["facts"]) == 5
    # Проверяем финальный Markdown отчет и метрики цитирования
    assert "https://example.org/research/1" in final_state["final_report"]
    assert final_state["metrics"]["citation_count"] >= 5

    eval_scores = evaluate_report_deterministic(
        report_md=final_state["final_report"],
        facts=final_state["facts"],
        coverage_score=final_state["coverage_score"],
    )
    assert eval_scores.citation_precision == 1.0
    assert eval_scores.hallucinated_urls_count == 0
    assert eval_scores.groundedness_heuristic > 0.85
