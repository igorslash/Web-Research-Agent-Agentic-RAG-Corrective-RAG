import re
from typing import Dict, List
from pydantic import BaseModel, Field
from src.state import FactItem


class ReportQualityScores(BaseModel):
    """Метрики качества финального отчета (RAGAS / Phoenix Eval совместимые)."""

    citation_precision: float = Field(description="Доля валидных ссылок [i] в тексте, которые реально существуют в пуле источников")
    source_utilization_rate: float = Field(description="Доля найденных источников, процитированных в финальном отчете")
    subquestion_coverage: float = Field(description="Оценка покрытия подвопросов (из узла evaluate)")
    hallucinated_urls_count: int = Field(description="Количество URL в отчете, отсутствующих в исходных фактах")
    groundedness_heuristic: float = Field(description="Итоговый индекс проверяемости и приземленности фактов (0..1)")


def evaluate_report_deterministic(
    report_md: str,
    facts: List[FactItem],
    coverage_score: float,
) -> ReportQualityScores:
    """
    Быстрый детерминированный расчет production-метрик качества цитирования и отсутствия галлюцинаций URL.
    Используется в CI/CD пайплайне и как кастомный эвалюатор для Arize Phoenix.
    """
    valid_ids = {f.source_id for f in facts}
    valid_urls = {f.url.rstrip("/") for f in facts}

    # 1. Проверяем все inline-цитаты вида [1], [2, 5]
    raw_brackets = re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", report_md)
    all_cited_nums: List[int] = []
    for group in raw_brackets:
        for num_str in group.split(","):
            num_str = num_str.strip()
            if num_str.isdigit():
                all_cited_nums.append(int(num_str))

    if all_cited_nums:
        valid_citations = [n for n in all_cited_nums if n in valid_ids]
        citation_precision = round(len(valid_citations) / len(all_cited_nums), 4)
        unique_valid_cited = set(valid_citations)
    else:
        citation_precision = 0.0
        unique_valid_cited = set()

    source_utilization = round(len(unique_valid_cited) / len(valid_ids), 4) if valid_ids else 0.0

    # 2. Проверяем наличие сгенерированных/галлюцинированных URL
    urls_in_report = re.findall(r"https?://[^\s\)\]>]+", report_md)
    hallucinated_urls = [
        u for u in urls_in_report if u.rstrip(".,;/)") not in valid_urls
    ]

    url_penalty = 0.0 if not hallucinated_urls else min(0.5, 0.15 * len(hallucinated_urls))
    groundedness = max(
        0.0,
        round(0.45 * citation_precision + 0.35 * coverage_score + 0.20 * min(1.0, source_utilization * 1.3) - url_penalty, 4),
    )

    return ReportQualityScores(
        citation_precision=citation_precision,
        source_utilization_rate=source_utilization,
        subquestion_coverage=round(coverage_score, 4),
        hallucinated_urls_count=len(hallucinated_urls),
        groundedness_heuristic=groundedness,
    )
