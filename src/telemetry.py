import logging
from typing import Optional
from src.config import settings

logger = logging.getLogger(__name__)

_tracer_initialized = False


def setup_phoenix_telemetry() -> Optional[object]:
    """
    Инициализирует OpenTelemetry + Arize Phoenix для автоматического трейсинга
    всех узлов LangGraph, вызовов LLM, задержек, токенов и стоимости.
    """
    global _tracer_initialized
    if _tracer_initialized or not settings.phoenix_enabled:
        return None

    try:
        from phoenix.otel import register
        from openinference.instrumentation.langchain import LangChainInstrumentor

        tracer_provider = register(
            project_name=settings.phoenix_project_name,
            endpoint=settings.phoenix_collector_endpoint,
        )
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
        _tracer_initialized = True
        logger.info(
            "Arize Phoenix OpenTelemetry instrumentation initialized: project='%s', endpoint='%s'",
            settings.phoenix_project_name,
            settings.phoenix_collector_endpoint,
        )
        return tracer_provider
    except Exception as exc:
        logger.warning(
            "Could not initialize Arize Phoenix telemetry (continuing without OTel export): %s",
            exc,
        )
        return None
