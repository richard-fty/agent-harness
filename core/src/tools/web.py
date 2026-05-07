"""Built-in web research tool."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import httpx

from agent.core.models import ToolGroup, ToolParameter
from services.web_search import search_web
from tools.base import BuiltinTool


class WebResearchTool(BuiltinTool):
    name = "web_research"
    description = (
        "Search the web and fetch the top pages in one call. Returns each result's "
        "url, title, snippet, and fetched text when available. Prefer this over "
        "separate search and fetch calls."
    )
    tool_group = ToolGroup.RUNTIME
    is_read_only = True
    is_concurrency_safe = True
    requires_confirmation = False
    is_networked = True
    mutates_state = False
    parameters = [
        ToolParameter(name="query", type="string", description="Search query"),
        ToolParameter(
            name="num_results",
            type="integer",
            description="Number of search results to return (default: 5, max: 10)",
            required=False,
            default=5,
        ),
        ToolParameter(
            name="fetch_top",
            type="integer",
            description="How many of the top results to also fetch in a second network pass (default: 0, max: 5)",
            required=False,
            default=0,
        ),
        ToolParameter(
            name="max_chars",
            type="integer",
            description="Per-page character cap after fetch (default: 4000)",
            required=False,
            default=4000,
        ),
        ToolParameter(
            name="topic",
            type="string",
            description='Search topic for Tavily: "general" or "news" (default: "general")',
            required=False,
            default="general",
            enum=["general", "news"],
        ),
        ToolParameter(
            name="time_range",
            type="string",
            description='Tavily recency filter such as "day", "week", "month", or "year"',
            required=False,
        ),
    ]

    def __init__(self) -> None:
        self._page_cache: dict[str, str] = {}

    async def execute(self, **kwargs: Any) -> str:
        query = kwargs["query"]
        raw_num_results = kwargs.get("num_results", 5)
        raw_fetch_top = kwargs.get("fetch_top", 0)
        raw_max_chars = kwargs.get("max_chars", 4000)
        topic = str(kwargs.get("topic") or "general").strip().lower()
        if topic not in {"general", "news"}:
            topic = "general"
        time_range = kwargs.get("time_range")
        time_range = str(time_range).strip().lower() if time_range else None
        if topic == "news":
            query = _normalize_news_query(query)
            time_range = time_range or "week"
        num_results = max(1, min(int(5 if raw_num_results is None else raw_num_results), 10))
        fetch_top = max(0, min(int(0 if raw_fetch_top is None else raw_fetch_top), 5))
        max_chars = max(250, int(4000 if raw_max_chars is None else raw_max_chars))

        queries_used = _plan_research_queries(query)
        per_query_limit = num_results if len(queries_used) == 1 else max(2, min(num_results, 4))
        raw_results: list[dict[str, str]] = []
        for planned_query in queries_used:
            raw_results.extend(await search_web(
                planned_query,
                max_results=per_query_limit,
                include_raw_content=fetch_top > 0,
                topic=topic,
                time_range=time_range,
            ))
        deduped_results: list[dict[str, str]] = []
        seen_urls: set[str] = set()
        for item in raw_results:
            url = (item.get("url") or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            result = {
                "title": (item.get("title") or url).strip(),
                "url": url,
                "snippet": (item.get("snippet") or "").strip(),
            }
            if item.get("text"):
                result["text"] = item["text"][:max_chars]
            deduped_results.append(result)
            if len(deduped_results) >= num_results:
                break

        async def fetch_result(result: dict[str, str]) -> dict[str, Any]:
            enriched: dict[str, Any] = dict(result)
            if enriched.get("text"):
                return enriched
            url = result["url"]
            try:
                text = self._page_cache.get(url)
                if text is None:
                    text = await _fetch_page_text(url, max_chars)
                    self._page_cache[url] = text
                enriched["text"] = text[:max_chars]
            except httpx.TimeoutException:
                enriched["fetch_error"] = "timeout"
            except httpx.HTTPStatusError as exc:
                enriched["fetch_error"] = f"http_{exc.response.status_code}"
            except Exception as exc:  # pragma: no cover - defensive
                enriched["fetch_error"] = str(exc)
            return enriched

        fetched: list[dict[str, Any]] = []
        if fetch_top > 0 and deduped_results:
            fetched = await asyncio.gather(
                *(fetch_result(item) for item in deduped_results[:fetch_top])
            )

        enriched_results: list[dict[str, Any]] = fetched + [
            dict(item) for item in deduped_results[len(fetched):]
        ]
        payload = {
            "query": query,
            "queries_used": queries_used,
            "topic": topic,
            "time_range": time_range,
            "results": enriched_results,
        }
        return json.dumps(payload, indent=2)


def _plan_research_queries(query: str) -> list[str]:
    cleaned = _normalize_query(query)
    return [cleaned]


_SOURCE_HINTS = {
    "bloomberg",
    "cnbc",
    "cnn",
    "coindesk",
    "reuters",
    "wsj",
}

_NEWS_QUERY_NOISE = {
    "analysis",
    "company",
    "current",
    "earnings",
    "financial",
    "latest",
    "market",
    "news",
    "overview",
    "outlook",
    "performance",
    "price",
    "recent",
    "stock",
    "stocks",
}


def _normalize_news_query(query: str) -> str:
    cleaned = _normalize_query(query)
    lowered = cleaned.lower()
    if any(source in lowered for source in _SOURCE_HINTS):
        return cleaned

    duplicate_latest_news = lowered.count("latest news") > 1
    tokens = re.findall(r"[A-Za-z0-9&.-]+", cleaned)
    has_noise = duplicate_latest_news or any(token.lower() in _NEWS_QUERY_NOISE for token in tokens)
    has_year = any(re.fullmatch(r"20\d{2}", token) for token in tokens)
    if not (has_noise or has_year):
        return cleaned

    subject_tokens: list[str] = []
    for token in tokens:
        lowered_token = token.lower()
        if re.fullmatch(r"20\d{2}", token):
            break
        if lowered_token in _NEWS_QUERY_NOISE:
            break
        if subject_tokens and re.fullmatch(r"[A-Z]{1,6}(?:-[A-Z]{1,4})?", token):
            continue
        subject_tokens.append(token)
        if len(subject_tokens) >= 4:
            break
    subject = " ".join(subject_tokens).strip()
    if not subject:
        return cleaned
    return f"{subject} latest news"


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query).strip()


def _strip_html(html: str) -> str:
    """Simple HTML tag stripping. Not perfect but good enough for agent consumption."""
    import re
    # Remove script and style blocks
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # Remove tags
    html = re.sub(r"<[^>]+>", " ", html)
    # Collapse whitespace
    html = re.sub(r"\s+", " ", html).strip()
    return html


async def _fetch_page_text(url: str, max_chars: int) -> str:
    async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
        resp = await client.get(url, headers={"User-Agent": "ApexAgent/0.1"})
        resp.raise_for_status()

    content_type = resp.headers.get("content-type", "")
    text = resp.text

    if "html" in content_type:
        text = _strip_html(text)

    if len(text) > max_chars:
        return text[:max_chars] + f"\n\n[... truncated, {len(resp.text)} total chars]"
    return text
