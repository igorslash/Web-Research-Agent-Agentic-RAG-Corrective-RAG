import logging
import operator
import re
from typing import Annotated, List, Optional, TypedDict
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FactItem(BaseModel):
    """Структурированный факт/источник, добытый из поиска и веб-скрапинга."""

    source_id: int = Field(description="Порядковый номер источника для цитирования [1], [2]...")
    query: str = Field(description="Поисковый подвопрос, по которому найден источник")
    title: str = Field(description="Заголовок веб-страницы")
    url: str = Field(description="URL источника")
    snippet: str = Field(description="Краткий сниппет из поисковой выдачи")
    scraped_content: str = Field(description="Очищенный полный текст со страницы (скрапер)")
    relevance_score: float = Field(default=0.0, description="Оценка релевантности от поискового движка")


class PlanOutput(BaseModel):
    """Structured Output для узла planner."""

    subquestions: List[str] = Field(
        description="Список из 3-5 конкретных, атомарных поисковых подвопросов для глубокого раскрытия темы",
        min_length=3,
        max_length=5,
    )
    reasoning: str = Field(description="Краткое обоснование стратегии декомпозиции запроса")


class EvaluationOutput(BaseModel):
    """Structured Output для узла evaluate (LLM-as-a-Judge / Reflection)."""

    is_sufficient: bool = Field(
        description="True, если собранных фактов достаточно для полного, объективного и проверяемого ответа на исходный запрос"
    )
    coverage_score: float = Field(
        description="Оценка полноты покрытия подвопросов от 0.0 до 1.0",
        ge=0.0,
        le=1.0,
    )
    missing_aspects: List[str] = Field(
        default_factory=list,
        description="Какие аспекты исходного вопроса остались нераскрытыми или требуют верификации",
    )
    follow_up_queries: List[str] = Field(
        default_factory=list,
        description="Переформулированные уточняющие поисковые запросы (1-3 шт.), если информации недостаточно",
    )


class TelemetryMetrics(BaseModel):
    """Production-метрики выполнения графа для логирования и Phoenix."""

    total_latency_sec: float = 0.0
    planner_latency_sec: float = 0.0
    search_latency_sec: float = 0.0
    evaluate_latency_sec: float = 0.0
    writer_latency_sec: float = 0.0
    iterations_used: int = 0
    subquestions_count: int = 0
    unique_sources_count: int = 0
    scraped_pages_success: int = 0
    coverage_score: float = 0.0
    citation_count: int = 0
    citation_density: float = 0.0


class ResearchState(TypedDict):
    """Глобальное состояние графа LangGraph."""

    # Исходный запрос пользователя
    user_query: str

    # Полный список всех заданных подвопросов (история запросов)
    questions: Annotated[List[str], operator.add]

    # Текущая очередь подвопросов для следующей итерации узла search
    pending_queries: List[str]

    # Собранные факты (сниппеты + скрапленный контент страниц)
    facts: List[FactItem]

    # Счетчик выполненных циклов рефлексии (максимум 2 дополнительных цикла)
    iteration: int

    # Результат оценки полноты на узле evaluate
    is_sufficient: bool
    coverage_score: float
    evaluation_notes: str

    # Финальный Markdown-отчет со ссылками
    final_report: str

    # Метрики исполнения для продакшн-мониторинга
    metrics: dict


def compute_citation_metrics(markdown_report: str, total_sources: int) -> tuple[int, float]:
    """Подсчитывает число уникальных цитирований [N] и долю задействованных источников."""
    if not markdown_report:
        return 0, 0.0
    matches = re.findall(r"\[(\d+)\]", markdown_report)
    unique_cited = {int(m) for m in matches if int(m) <= max(total_sources, 1)}
    density = round(len(unique_cited) / total_sources, 3) if total_sources > 0 else 0.0
    return len(matches), density
