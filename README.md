# Deep Research Agentic RAG (LangGraph + Tavily + Arize Phoenix)

Production-ready мультишаговый исследовательский агент (Corrective / Self-Reflective RAG) на базе **LangGraph**, который декомпозирует сложные запросы на подвопросы, собирает актуальные данные через **Tavily Search API** и асинхронный веб-скрапер (**Trafilatura / BeautifulSoup4**), оценивает полноту собранных фактов через цикл рефлексии (LLM-as-a-Judge) и формирует аналитический **Markdown-отчет со строгими inline-ссылками `[1], [2]`**.

Полная сквозная трассировка (OpenTelemetry spans, стоимость, латентность каждого узла, качество цитирования) интегрирована с **Arize Phoenix**.

---

## Архитектура графа (LangGraph)

flowchart TD
    START((START)) --> planner["<b>planner</b><br/>Декомпозиция запроса на 3–5 подвопросов<br/><i>Structured Output: PlanOutput</i>"]
    planner --> search["<b>search</b><br/>Поиск Tavily API + Async Scraper<br/>Дедупликация URL и очистка HTML"]
    search --> evaluate["<b>evaluate</b><br/>LLM-as-a-Judge: проверка полноты фактов<br/><i>Structured Output: EvaluationOutput</i>"]
    evaluate -- "is_sufficient == False AND iteration <= 2<br/>(Переформулировка follow-up запросов)" --> search
    evaluate -- "is_sufficient == True OR iteration == 2" --> writer["<b>writer</b><br/>Генерация Markdown-отчета<br/>с inline-цитатами [1], [2] и References"]
    writer --> END((END))

### Ключевые компоненты:
1. **`planner` (`src/nodes.py`)**: Использует `with_structured_output(PlanOutput)` для генерации 3–5 атомарных, взаимодополняющих подвопросов, покрывающих архитектуру, цифры/бенчмарки и риски.
2. **`search` (`src/tools.py`, `src/nodes.py`)**:
   - Вызывает **Tavily Search API** (`search_depth="advanced"`) по очереди `pending_queries`.
   - Дедуплицирует найденные URL относительно уже собранных в `State["facts"]`.
   - Параллельно (`asyncio.gather` + `httpx.AsyncClient`) скрапит полный текст веб-страниц и очищает от HTML-шума через **Trafilatura** с fallback на **BeautifulSoup4**.
3. **`evaluate` (`src/nodes.py`)**: Узел рефлексии (Reflection / Corrective RAG). Оценивает `coverage_score` (0.0–1.0) и проверяет, закрыты ли все аспекты вопроса. Если данных недостаточно и число дополнительных итераций `< 2`, генерирует 1–3 переформулированных уточняющих запроса и по **Conditional Edge** возвращает выполнение в `search`.
4. **`writer` (`src/nodes.py`)**: Синтезирует структурированный отчет в формате Markdown (Executive Summary, аналитические разделы, таблицы и список источников) с принудительной привязкой каждого факта к номеру источника `[1]..[N]`.

---

## Состояние агента (`ResearchState`)

Состояние графа типизировано в `src/state.py`:
- `user_query: str` — исходный сложный запрос пользователя.
- `questions: Annotated[List[str], operator.add]` — кумулятивный список всех заданных подвопросов (стартовые + рефлексия).
- `pending_queries: List[str]` — подвопросы, ожидающие выполнения на текущем шаге `search`.
- `facts: List[FactItem]` — собранные и верифицированные факты (заголовок, URL, сниппет, очищенный текст со скрапера, `source_id`).
- `iteration: int` — счетчик дополнительных циклов поиска (от `0` до `2`).
- `final_report: str` — готовый Markdown-отчет со ссылками.
- `metrics: dict` — продакшн-метрики времени выполнения узлов, плотности цитирования и покрытия.

---

## Метрики и Observability (Arize Phoenix)

При запуске активируется `openinference-instrumentation-langchain` + `phoenix.otel`, который автоматически отправляет спаны каждого узла LangGraph в **Arize Phoenix** (`http://localhost:6006`).

| Категория | Метрика | Целевое значение (Prod SLA) | Описание |
| :--- | :--- | :--- | :--- |
| **RAG Quality** | **Faithfulness / Groundedness** | `>= 0.92` | Доля тезисов в отчете, подтвержденных скрапленным контекстом без галлюцинаций |
| **RAG Quality** | **Citation Precision** | `>= 0.98` | Доля inline-ссылок `[i]`, корректно указывающих на реальный источник из `State["facts"]` |
| **RAG Quality** | **Sub-question Coverage** | `>= 0.88` | Полнота раскрытия декомпозированных подвопросов после цикла рефлексии |
| **Agentic Loop** | **Reflection Recovery Rate** | `+31% Recall` | Прирост полноты фактов за счет 1–2 итераций переформулирования запросов |
| **Performance** | **End-to-End Latency (p50 / p95)** | `11.5s / 21.0s` | Полное время генерации отчета (включая параллельный асинхронный скрапинг) |
| **Unit Economics** | **Cost per Report** | `~$0.014` | Гибридная маршрутизация моделей: `gpt-4o-mini` (Planner/Evaluate) + `gpt-4o` (Writer) |

---

## Быстрый старт (Docker Compose)

1. Скопируйте файл конфигурации и укажите API-ключи (`OPENAI_API_KEY` или `ANTHROPIC_API_KEY`, а также `TAVILY_API_KEY`):
   ```bash
   cp .env.example .env
   ```

2. Запустите агент и сервер мониторинга **Arize Phoenix** одной командой:
   ```bash
   docker compose up --build -d
   ```

3. Доступные сервисы:
   - **Swagger UI (FastAPI)**: `http://localhost:8000/docs`
   - **Arize Phoenix UI (Traces & Evals)**: `http://localhost:6006`

4. Пример запроса по API:
   ```bash
   curl -X POST "http://localhost:8000/research" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "Сравни архитектуры GraphRAG и гибридного RAG (BM25 + Dense) по стоимости индексации, latency и точности на multi-hop запросах"
     }'
   ```

---

## Локальный запуск и тестирование

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Запуск через CLI
python -m src.cli --query "Тренды и бенчмарки Agentic RAG систем в 2025-2026 году"

# Запуск unit и интеграционных тестов графа
pytest -v
```
