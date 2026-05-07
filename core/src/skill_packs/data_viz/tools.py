"""Tools for the data-viz skill pack.

The first version reuses the coding skill's app-editing tools so data-viz
tasks get the same patch, plan, and app-preview surface.
"""

from __future__ import annotations

from typing import Any

from agent.core.models import ToolDef
from skill_packs.coding.tools import get_tools as get_coding_tools


def get_tools() -> list[tuple[ToolDef, Any]]:
    return get_coding_tools()
