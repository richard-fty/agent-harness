"""Data visualization skill metadata."""

from __future__ import annotations

from typing import Any

from agent.core.models import ToolDef
from skill_packs.base import SkillPack, ToolHandler
from skill_packs.data_viz.tools import get_tools


class DataVizSkill(SkillPack):
    @property
    def name(self) -> str:
        return "data_viz"

    @property
    def description(self) -> str:
        return "Build, validate, and preview interactive data visualization story apps."

    @property
    def keywords(self) -> list[str]:
        return [
            "data",
            "dataset",
            "csv",
            "chart",
            "charts",
            "dashboard",
            "visualization",
            "visualisation",
            "story",
            "report",
            "kpi",
            "metric",
            "metrics",
            "filter",
            "insight",
            "trend",
            "breakdown",
        ]

    def get_tools(self) -> list[tuple[ToolDef, ToolHandler]]:
        return get_tools()
