from __future__ import annotations

from config import settings
from tools.base import assemble_tool_pool


def test_rag_tools_are_not_assembled_by_default() -> None:
    original = settings.enable_rag_tools
    settings.enable_rag_tools = False
    try:
        names = {tool.name for tool in assemble_tool_pool(include_runtime_injected=True)}
    finally:
        settings.enable_rag_tools = original

    assert "rag_query" not in names
    assert "rag_index" not in names
    assert "rag_list_collections" not in names
