from contextlib import asynccontextmanager
import logging
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.graph import research_graph
from src.state import FactItem, TelemetryMetrics
from src.telemetry import setup_phoenix_telemetry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_phoenix_telemetry()
    yield


app = FastAPI(
    title="Deep Research Agentic RAG API",
    description="LangGraph multi-step web research agent with reflection loop and Arize Phoenix observability.",
    version="1.0.0",
    lifespan=lifespan,
)


class ResearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=5,
        description="Сложный исследовательский запрос пользователя",
        examples=[
            "Сравни архитектуры GraphRAG и классического гибридного RAG (BM25 + Dense) по задержкам, стоимости индексации и качеству на multi-hop вопросах"
        ],
    )


class ResearchResponse(BaseModel):
    query: str
    questions: List[str]
    iterations: int
    facts_count: int
    facts: List[FactItem]
    final_report: str
    metrics: TelemetryMetrics


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "deep-research-rag-agent"}


@app.post("/research", response_model=ResearchResponse)
async def run_research(payload: ResearchRequest):
    try:
        initial_state = {
            "user_query": payload.query,
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
        facts = result.get("facts", [])
        metrics_data = result.get("metrics", {})

        return ResearchResponse(
            query=payload.query,
            questions=result.get("questions", []),
            iterations=result.get("iteration", 0),
            facts_count=len(facts),
            facts=facts,
            final_report=result.get("final_report", ""),
            metrics=TelemetryMetrics(**metrics_data),
        )
    except Exception as exc:
        logger.exception("Error while executing research graph")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
