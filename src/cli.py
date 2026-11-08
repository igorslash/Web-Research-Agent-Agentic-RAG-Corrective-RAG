import argparse
import asyncio
import json
import logging
from src.evaluation import evaluate_report_deterministic
from src.graph import research_graph
from src.telemetry import setup_phoenix_telemetry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


async def run_live(query: str) -> None:
    setup_phoenix_telemetry()
    initial_state = {
        "user_query": query,
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
    result = await research_graph.ainvoke(initial_state)
    scores = evaluate_report_deterministic(
        report_md=result["final_report"],
        facts=result["facts"],
        coverage_score=result.get("coverage_score", 0.0),
    )

    print("\n" + "=" * 80)
    print("SUB-QUESTIONS ASKED:")
    for i, q in enumerate(result["questions"], 1):
        print(f"  {i}. {q}")
    print("\nTELEMETRY & QUALITY METRICS:")
    print(json.dumps({**result["metrics"], **scores.model_dump()}, indent=2, ensure_ascii=False))
    print("=" * 80 + "\n")
    print(result["final_report"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep Research Agentic RAG CLI")
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default="Сравни подходы Agentic RAG на базе LangGraph и классический Naive RAG для корпоративного поиска",
        help="Исследовательский запрос пользователя",
    )
    args = parser.parse_args()
    asyncio.run(run_live(args.query))


if __name__ == "__main__":
    main()
