import asyncio
import logging
from typing import Dict, List, Optional, Set
import httpx
from bs4 import BeautifulSoup
from src.config import settings
from src.state import FactItem

logger = logging.getLogger(__name__)

try:
    import trafilatura
except ImportError:
    trafilatura = None


class WebScraper:
    """Асинхронный скрапер веб-страниц с очисткой HTML через Trafilatura + fallback на BeautifulSoup4."""

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/128.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    def __init__(self, timeout: float = settings.scraper_timeout_sec, max_chars: int = settings.max_scraped_chars):
        self.timeout = timeout
        self.max_chars = max_chars

    @staticmethod
    def _clean_html_bs4(html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside", "form", "svg"]):
            tag.decompose()
        lines = [line.strip() for line in soup.get_text(separator="\n").splitlines() if line.strip()]
        return "\n".join(lines)

    def extract_main_text(self, html: str) -> str:
        if not html:
            return ""
        text = ""
        if trafilatura is not None:
            try:
                extracted = trafilatura.extract(
                    html,
                    include_comments=False,
                    include_tables=True,
                    no_fallback=False,
                )
                if extracted:
                    text = extracted.strip()
            except Exception as exc:
                logger.debug("Trafilatura extraction error: %s", exc)

        if not text:
            text = self._clean_html_bs4(html)

        return text[: self.max_chars]

    async def scrape_url(self, client: httpx.AsyncClient, url: str) -> str:
        if not url or not url.startswith(("http://", "https://")):
            return ""
        try:
            response = await client.get(url, headers=self.HEADERS, follow_redirects=True, timeout=self.timeout)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                return ""
            return self.extract_main_text(response.text)
        except Exception as exc:
            logger.debug("Failed to scrape URL %s: %s", url, exc)
            return ""


class TavilySearchAndScrapeTool:
    """Инструмент гибридного поиска: Tavily Search API + параллельный асинхронный до-скрапинг страниц."""

    def __init__(self, api_key: Optional[str] = None, max_results: int = settings.search_results_per_query):
        self.api_key = api_key or settings.tavily_api_key
        self.max_results = max_results
        self.scraper = WebScraper()

    def _search_tavily_sync(self, query: str) -> List[Dict]:
        from tavily import TavilyClient

        if not self.api_key:
            raise ValueError("TAVILY_API_KEY is not configured. Set it in .env or environment variables.")

        client = TavilyClient(api_key=self.api_key)
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=self.max_results,
            include_raw_content=False,
        )
        return response.get("results", [])

    async def execute_batch(
        self,
        queries: List[str],
        existing_facts: List[FactItem],
    ) -> List[FactItem]:
        """
        Выполняет поиск по списку подвопросов, дедуплицирует URL и параллельно скрапит полный текст страниц.
        """
        seen_urls: Set[str] = {fact.url for fact in existing_facts}
        next_source_id = len(existing_facts) + 1
        raw_candidates: List[Dict] = []

        # 1. Выполняем поисковые запросы через Tavily
        for query in queries:
            try:
                results = await asyncio.to_thread(self._search_tavily_sync, query)
                for item in results:
                    url = item.get("url", "").strip()
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    raw_candidates.append(
                        {
                            "source_id": next_source_id,
                            "query": query,
                            "title": item.get("title") or url,
                            "url": url,
                            "snippet": item.get("content") or "",
                            "relevance_score": float(item.get("score", 0.0)),
                        }
                    )
                    next_source_id += 1
            except Exception as exc:
                logger.error("Tavily search error for query '%s': %s", query, exc)

        if not raw_candidates:
            return []

        # 2. Параллельно скрапим страницы для обогащения сниппетов полным контекстом
        async with httpx.AsyncClient() as http_client:
            scrape_tasks = [
                self.scraper.scrape_url(http_client, candidate["url"])
                for candidate in raw_candidates
            ]
            scraped_texts = await asyncio.gather(*scrape_tasks, return_exceptions=True)

        new_facts: List[FactItem] = []
        for candidate, scraped in zip(raw_candidates, scraped_texts):
            clean_scraped = scraped if isinstance(scraped, str) and scraped.strip() else candidate["snippet"]
            new_facts.append(
                FactItem(
                    source_id=candidate["source_id"],
                    query=candidate["query"],
                    title=candidate["title"],
                    url=candidate["url"],
                    snippet=candidate["snippet"],
                    scraped_content=clean_scraped,
                    relevance_score=candidate["relevance_score"],
                )
            )

        return new_facts
