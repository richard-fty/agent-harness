from __future__ import annotations

import json

import httpx
import pytest

from tools.web import WebResearchTool, _normalize_news_query, _plan_research_queries


@pytest.mark.asyncio
async def test_web_research_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_search(
        query: str,
        *,
        max_results: int = 5,
        include_raw_content: bool = False,
        topic: str = "general",
        time_range: str | None = None,
    ) -> list[dict[str, str]]:
        assert query == "nvda earnings"
        assert max_results == 4
        return [
            {"title": "Reuters", "url": "https://example.com/1", "snippet": "r1"},
            {"title": "SEC", "url": "https://example.com/2", "snippet": "r2"},
            {"title": "FT", "url": "https://example.com/3", "snippet": "r3"},
            {"title": "Bloomberg", "url": "https://example.com/4", "snippet": "r4"},
        ]

    async def fake_fetch(url: str, max_chars: int) -> str:
        return f"body:{url}:{max_chars}"

    monkeypatch.setattr("tools.web.search_web", fake_search)
    monkeypatch.setattr("tools.web._fetch_page_text", fake_fetch)

    tool = WebResearchTool()
    content = await tool.execute(query="nvda earnings", num_results=4, fetch_top=2, max_chars=1234)
    payload = json.loads(content)

    assert payload["query"] == "nvda earnings"
    assert len(payload["results"]) == 4
    assert payload["results"][0]["text"] == "body:https://example.com/1:1234"
    assert payload["results"][1]["text"] == "body:https://example.com/2:1234"
    assert "text" not in payload["results"][2]
    assert "text" not in payload["results"][3]


@pytest.mark.asyncio
async def test_web_research_marks_partial_fetch_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_search(
        query: str,
        *,
        max_results: int = 5,
        include_raw_content: bool = False,
        topic: str = "general",
        time_range: str | None = None,
    ) -> list[dict[str, str]]:
        return [
            {"title": "One", "url": "https://example.com/1", "snippet": "a"},
            {"title": "Two", "url": "https://example.com/2", "snippet": "b"},
        ]

    async def fake_fetch(url: str, max_chars: int) -> str:
        if url.endswith("/2"):
            raise httpx.TimeoutException("timed out")
        return "ok"

    monkeypatch.setattr("tools.web.search_web", fake_search)
    monkeypatch.setattr("tools.web._fetch_page_text", fake_fetch)

    tool = WebResearchTool()
    payload = json.loads(await tool.execute(query="nvda", fetch_top=2))

    assert payload["results"][0]["text"] == "ok"
    assert payload["results"][1]["fetch_error"] == "timeout"


@pytest.mark.asyncio
async def test_web_research_search_only_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"fetch": 0}

    async def fake_search(
        query: str,
        *,
        max_results: int = 5,
        include_raw_content: bool = False,
        topic: str = "general",
        time_range: str | None = None,
    ) -> list[dict[str, str]]:
        return [{"title": "Only", "url": "https://example.com/1", "snippet": "x"}]

    async def fake_fetch(url: str, max_chars: int) -> str:
        calls["fetch"] += 1
        return "unexpected"

    monkeypatch.setattr("tools.web.search_web", fake_search)
    monkeypatch.setattr("tools.web._fetch_page_text", fake_fetch)

    tool = WebResearchTool()
    payload = json.loads(await tool.execute(query="nvda", fetch_top=0))

    assert calls["fetch"] == 0
    assert "text" not in payload["results"][0]


@pytest.mark.asyncio
async def test_web_research_reuses_cached_fetches(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"search": 0, "fetch": 0}

    async def fake_search(
        query: str,
        *,
        max_results: int = 5,
        include_raw_content: bool = False,
        topic: str = "general",
        time_range: str | None = None,
    ) -> list[dict[str, str]]:
        calls["search"] += 1
        return [{"title": "Only", "url": "https://example.com/1", "snippet": query}]

    async def fake_fetch(url: str, max_chars: int) -> str:
        calls["fetch"] += 1
        return "cached body"

    monkeypatch.setattr("tools.web.search_web", fake_search)
    monkeypatch.setattr("tools.web._fetch_page_text", fake_fetch)

    tool = WebResearchTool()
    first = json.loads(await tool.execute(query="nvda earnings", fetch_top=1))
    second = json.loads(await tool.execute(query="nvda filings", fetch_top=1))

    assert calls["search"] == 2
    assert calls["fetch"] == 1
    assert first["results"][0]["text"] == "cached body"
    assert second["results"][0]["text"] == "cached body"


def test_plan_research_queries_keeps_vague_stock_query_unchanged() -> None:
    queries = _plan_research_queries("Adobe ADBE stock analysis earnings financial performance 2026")
    assert queries == ["Adobe ADBE stock analysis earnings financial performance 2026"]


def test_plan_research_queries_keeps_prompt_words_unchanged() -> None:
    queries = _plan_research_queries("brief me on Tesla stock")
    assert queries == ["brief me on Tesla stock"]


def test_plan_research_queries_keeps_focused_company_news_query() -> None:
    queries = _plan_research_queries("Adobe layoffs latest news 2026")
    assert queries == ["Adobe layoffs latest news 2026"]


def test_plan_research_queries_keeps_allowed_stock_harness_queries() -> None:
    assert _plan_research_queries("Datadog stock ticker") == ["Datadog stock ticker"]
    assert _plan_research_queries("Datadog latest news") == ["Datadog latest news"]


def test_normalize_news_query_simplifies_noisy_stock_news_query() -> None:
    assert _normalize_news_query("Tesla TSLA company overview recent news 2025") == "Tesla latest news"
    assert _normalize_news_query("Bitcoin BTC latest news latest news 2025") == "Bitcoin latest news"
    assert _normalize_news_query("Super Micro Computer latest news") == "Super Micro Computer latest news"
    assert _normalize_news_query("Tesla CNBC") == "Tesla CNBC"


@pytest.mark.asyncio
async def test_web_research_uses_exact_normalized_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_queries: list[str] = []

    async def fake_search(
        query: str,
        *,
        max_results: int = 5,
        include_raw_content: bool = False,
        topic: str = "general",
        time_range: str | None = None,
    ) -> list[dict[str, str]]:
        seen_queries.append(query)
        slug = len(seen_queries)
        return [{"title": f"Result {slug}", "url": f"https://example.com/{slug}", "snippet": query}]

    monkeypatch.setattr("tools.web.search_web", fake_search)

    tool = WebResearchTool()
    payload = json.loads(await tool.execute(
        query="Adobe ADBE stock analysis earnings financial performance 2026",
        num_results=5,
        fetch_top=0,
    ))

    assert payload["queries_used"] == ["Adobe ADBE stock analysis earnings financial performance 2026"]
    assert seen_queries == payload["queries_used"]
    assert len(payload["results"]) == 1


@pytest.mark.asyncio
async def test_web_research_requests_top_five_for_single_planned_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_limits: list[int] = []

    async def fake_search(
        query: str,
        *,
        max_results: int = 5,
        include_raw_content: bool = False,
        topic: str = "general",
        time_range: str | None = None,
    ) -> list[dict[str, str]]:
        seen_limits.append(max_results)
        return [
            {"title": f"Result {idx}", "url": f"https://example.com/{idx}", "snippet": query}
            for idx in range(max_results)
        ]

    monkeypatch.setattr("tools.web.search_web", fake_search)

    tool = WebResearchTool()
    payload = json.loads(await tool.execute(
        query="Adobe ADBE stock analysis earnings financial performance 2026",
        num_results=5,
        fetch_top=0,
    ))

    assert payload["queries_used"] == ["Adobe ADBE stock analysis earnings financial performance 2026"]
    assert seen_limits == [5]
    assert len(payload["results"]) == 5


@pytest.mark.asyncio
async def test_web_research_passes_tavily_topic_and_time_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    async def fake_search(
        query: str,
        *,
        max_results: int = 5,
        include_raw_content: bool = False,
        topic: str = "general",
        time_range: str | None = None,
    ) -> list[dict[str, str]]:
        seen.update({
            "query": query,
            "max_results": max_results,
            "include_raw_content": include_raw_content,
            "topic": topic,
            "time_range": time_range,
        })
        return [{"title": "Only", "url": "https://example.com/1", "snippet": "x"}]

    monkeypatch.setattr("tools.web.search_web", fake_search)

    tool = WebResearchTool()
    payload = json.loads(await tool.execute(
        query="Tesla latest news",
        num_results=5,
        fetch_top=0,
        topic="news",
        time_range="week",
    ))

    assert seen == {
        "query": "Tesla latest news",
        "max_results": 5,
        "include_raw_content": False,
        "topic": "news",
        "time_range": "week",
    }
    assert payload["topic"] == "news"
    assert payload["time_range"] == "week"


@pytest.mark.asyncio
async def test_web_research_news_topic_cleans_query_and_defaults_to_week(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    async def fake_search(
        query: str,
        *,
        max_results: int = 5,
        include_raw_content: bool = False,
        topic: str = "general",
        time_range: str | None = None,
    ) -> list[dict[str, str]]:
        seen.update({"query": query, "topic": topic, "time_range": time_range})
        return [{"title": "Only", "url": "https://example.com/1", "snippet": "x"}]

    monkeypatch.setattr("tools.web.search_web", fake_search)

    tool = WebResearchTool()
    payload = json.loads(await tool.execute(
        query="Tesla TSLA company overview recent news 2025",
        num_results=5,
        fetch_top=0,
        topic="news",
    ))

    assert seen == {"query": "Tesla latest news", "topic": "news", "time_range": "week"}
    assert payload["query"] == "Tesla latest news"
    assert payload["queries_used"] == ["Tesla latest news"]
