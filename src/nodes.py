import logging
import time
from typing import Any, Dict, List
from langchain_core.messages import HumanMessage, SystemMessage
from src.config import settings
from src.prompts import EVALUATOR_SYSTEM_PROMPT, PLANNER_SYSTEM_PROMPT, WRITER_SYSTEM_PROMPT
from src.state import (
    EvaluationOutput,
    FactItem,
    PlanOutput,
    ResearchState,
    compute_citation_metrics,
)
from src.tools import TavilySearchAndScrapeTool

logger = logging.getLogger(__name__)


def get_llm(model_name: str | None = None, temperature: float | None = None):
    """Фабрика LLM с поддержкой OpenAI и Anthropic."""
    target_model = model_name or settings.llm_model
    target_temp = settings.temperature if temperature is None else temperature

    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=target_model,
            temperature=target_temp,
            api_key=settings.anthropic_api_key or None,
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=target_model,
        temperature=target_temp,
        api_key=settings.openai_api_key or None,
    )


def _format_facts_for_prompt(facts: List[FactItem], max_chars_per_source: int = 1800) -> str:
    """Форматирует собранные факты с номерами [source_id] для подачи в evaluate и writer."""
    if not facts:
        return "Факты пока не найдены."

    blocks = []
    for fact in facts:
        body = (fact.scraped_content or fact.snippet)[:max_chars_per_source]
        blocks.append(
            f"[{fact.source_id}] Заголовок: {fact.title}\n"
            f"URL: {fact.url}\n"
            f"Поисковый подвопрос: {fact.query}\n"
            f"Извлеченный контент:\n{body}\n"
        )
    return "\n---\n".join(blocks)


async def planner_node(state: ResearchState) -> Dict[str, Any]:
    """
    Узел 1: Planner.
    Разбивает сложный запрос пользователя на 3–5 поисковых подвопросов (Structured Output).
    """
    t0 = time.perf_counter()
    user_query = state["user_query"]
    llm = get_llm(settings.llm_model)
    structured_planner = llm.with_structured_output(PlanOutput)

    sys_prompt = PLANNER_SYSTEM_PROMPT.format(
        min_q=settings.min_subquestions,
        max_q=settings.max_subquestions,
    )
    plan: PlanOutput = await structured_planner.ainvoke(
        [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=f"Исходный запрос пользователя:\n{user_query}"),
        ]
    )

    subquestions = plan.subquestions[
        : settings.max_subquestions
    ]
    elapsed = round(time.perf_counter() - t0, 3)

    metrics = dict(state.get("metrics") or {})
    metrics["planner_latency_sec"] = elapsed
    metrics["subquestions_count"] = len(subquestions)

    logger.info("Planner generated %d subquestions in %.2fs", len(subquestions), elapsed)

    return {
        "questions": subquestions,
        "pending_queries": subquestions,
        "iteration": 0,
        "metrics": metrics,
    }


async def search_node(state: ResearchState) -> Dict[str, Any]:
    """
    Узел 2: Search & Scrape.
    Выполняет поиск через Tavily API по текущим подвопросам и параллельно скрапит веб-страницы.
    """
    t0 = time.perf_counter()
    queries = state.get("pending_queries") or state.get("questions", [])
    existing_facts = list(state.get("facts") or [])

    tool = TavilySearchAndScrapeTool()
    new_facts = await tool.execute_batch(queries=queries, existing_facts=existing_facts)
    all_facts = existing_facts + new_facts

    elapsed = round(time.perf_counter() - t0, 3)
    metrics = dict(state.get("metrics") or {})
    metrics["search_latency_sec"] = round(metrics.get("search_latency_sec", 0.0) + elapsed, 3)
    metrics["unique_sources_count"] = len(all_facts)
    metrics["scraped_pages_success"] = sum(
        1 for f in all_facts if f.scraped_content and len(f.scraped_content) > len(f.snippet)
    )

    logger.info(
        "Search node collected %d new facts (total=%d) in %.2fs",
        len(new_facts),
        len(all_facts),
        elapsed,
    )

    return {
        "facts": all_facts,
        "pending_queries": [],
        "metrics": metrics,
    }


async def evaluate_node(state: ResearchState) -> Dict[str, Any]:
    """
    Узел 3: Evaluate (Reflection).
    Проверяет через LLM достаточность собранных фактов. Если данных мало и лимит итераций (< 2)
    не исчерпан — генерирует уточняющие поисковые запросы и отправляет граф на новый цикл.
    """
    t0 = time.perf_counter()
    user_query = state["user_query"]
    questions = state.get("questions", [])
    facts = state.get("facts", [])
    current_iteration = state.get("iteration", 0)

    llm = get_llm(settings.llm_model)
    structured_evaluator = llm.with_structured_output(EvaluationOutput)

    formatted_facts = _format_facts_for_prompt(facts, max_chars_per_source=1200)
    eval_prompt = (
        f"Исходный запрос пользователя: {user_query}\n\n"
        f"Уже заданные вопросы ({len(questions)} шт.):\n"
        + "\n".join(f"- {q}" for q in questions)
        + f"\n\nСобранные факты ({len(facts)} источников):\n{formatted_facts}"
    )

    evaluation: EvaluationOutput = await structured_evaluator.ainvoke(
        [
            SystemMessage(content=EVALUATOR_SYSTEM_PROMPT),
            HumanMessage(content=eval_prompt),
        ]
    )

    elapsed = round(time.perf_counter() - t0, 3)
    metrics = dict(state.get("metrics") or {})
    metrics["evaluate_latency_sec"] = round(metrics.get("evaluate_latency_sec", 0.0) + elapsed, 3)
    metrics["coverage_score"] = evaluation.coverage_score

    # Если фактов недостаточно и мы еще можем сделать дополнительную итерацию поиска (максимум 2 ретрая)
    can_retry = current_iteration < settings.max_iterations
    needs_more_search = (not evaluation.is_sufficient) and can_retry and bool(evaluation.follow_up_queries)

    if needs_more_search:
        follow_ups = evaluation.follow_up_queries[:3]
        next_iteration = current_iteration + 1
        metrics["iterations_used"] = next_iteration
        metrics["subquestions_count"] = len(questions) + len(follow_ups)
        logger.info(
            "Evaluate node triggered reflection retry #%d (coverage=%.2f). Follow-up queries: %s",
            next_iteration,
            evaluation.coverage_score,
            follow_ups,
        )
        return {
            "is_sufficient": False,
            "coverage_score": evaluation.coverage_score,
            "evaluation_notes": "; ".join(evaluation.missing_aspects),
            "iteration": next_iteration,
            "questions": follow_ups,
            "pending_queries": follow_ups,
            "metrics": metrics,
        }

    logger.info(
        "Evaluate node passed to writer (is_sufficient=%s, coverage=%.2f, iteration=%d)",
        evaluation.is_sufficient,
        evaluation.coverage_score,
        current_iteration,
    )
    return {
        "is_sufficient": True,
        "coverage_score": evaluation.coverage_score,
        "evaluation_notes": "; ".join(evaluation.missing_aspects),
        "pending_queries": [],
        "metrics": metrics,
    }


async def writer_node(state: ResearchState) -> Dict[str, Any]:
    """
    Узел 4: Writer.
    Формирует финальный аналитический отчет в Markdown со строгими inline-цитатами [1], [2] и списком ссылок.
    """
    t0 = time.perf_counter()
    user_query = state["user_query"]
    questions = state.get("questions", [])
    facts = state.get("facts", [])

    llm = get_llm(settings.writer_model)
    formatted_facts = _format_facts_for_prompt(facts, max_chars_per_source=2200)

    writer_input = (
        f"Исходный запрос пользователя:\n{user_query}\n\n"
        f"Исследованные подвопросы:\n"
        + "\n".join(f"- {q}" for q in questions)
        + f"\n\nПроверенные источники и факты (используй номера [1]..[{len(facts)}] для цитирования):\n"
        f"{formatted_facts}"
    )

    response = await llm.ainvoke(
        [
            SystemMessage(content=WRITER_SYSTEM_PROMPT),
            HumanMessage(content=writer_input),
        ]
    )
    report_md = response.content if isinstance(response.content, str) else str(response.content)

    elapsed = round(time.perf_counter() - t0, 3)
    citation_count, citation_density = compute_citation_metrics(report_md, len(facts))

    metrics = dict(state.get("metrics") or {})
    metrics["writer_latency_sec"] = elapsed
    metrics["citation_count"] = citation_count
    metrics["citation_density"] = citation_density
    metrics["total_latency_sec"] = round(
        metrics.get("planner_latency_sec", 0.0)
        + metrics.get("search_latency_sec", 0.0)
        + metrics.get("evaluate_latency_sec", 0.0)
        + elapsed,
        3,
    )

    logger.info(
        "Writer generated Markdown report (%d chars, %d citations, density=%.2f) in %.2fs",
        len(report_md),
        citation_count,
        citation_density,
        elapsed,
    )

    return {
        "final_report": report_md,
        "metrics": metrics,
    }
