from __future__ import annotations

from agent.context.assembler import ContextAssembler, PLAN_TOOL_NAMES
from agent.runtime.tool_dispatch import ToolDispatch
from agent.skills.loader import SkillLoader
from tools.planner import PlanManager, register_plan_tools


def test_preload_by_intent_falls_back_for_obvious_stock_request() -> None:
    loader = SkillLoader(ToolDispatch())
    loader.discover()

    loaded = loader.pre_load_by_intent("analyze adobe stock and give me a briefing")

    assert "stock_strategy" in loaded
    assert len(loaded) == 1


def test_preload_by_intent_loads_only_top_match_for_overlapping_request() -> None:
    loader = SkillLoader(ToolDispatch())
    loader.discover()

    loaded = loader.pre_load_by_intent(
        "Build an interactive data story report from the uploaded sales.csv dataset"
    )

    assert loaded == ["data_viz"]
    assert set(loader.loaded) == {"data_viz"}


def test_load_skill_for_tool_recovers_unloaded_skill_tool() -> None:
    loader = SkillLoader(ToolDispatch())
    loader.discover()
    assert loader.load_skill("stock_strategy") is True
    assert loader.unload_skill("stock_strategy") is True

    loaded_name = loader.load_skill_for_tool("fetch_market_data")

    assert loaded_name == "stock_strategy"
    assert "stock_strategy" in loader.loaded


def test_skill_planning_mode_is_declared_by_skill_md() -> None:
    loader = SkillLoader(ToolDispatch())
    loader.discover()

    assert loader.analyzed["stock_strategy"].planning_mode == "off"
    assert loader.analyzed["coding"].planning_mode == "on"


def test_plan_tools_are_hidden_unless_loaded_skill_enables_planning() -> None:
    dispatch = ToolDispatch()
    plan_manager = PlanManager()
    for tool_def, handler in register_plan_tools(plan_manager):
        dispatch.register(tool_def, handler)

    loader = SkillLoader(dispatch)
    loader.discover()
    assembler = ContextAssembler(None, None, plan_manager, loader)  # type: ignore[arg-type]

    assert assembler._excluded_tool_names() == PLAN_TOOL_NAMES
    assert "todo_update" not in {
        schema["function"]["name"]
        for schema in dispatch.to_openai_tools(exclude_names=assembler._excluded_tool_names())
    }

    loader.load_skill("stock_strategy")
    assert assembler._excluded_tool_names() == PLAN_TOOL_NAMES

    loader.load_skill("coding")
    assert assembler._excluded_tool_names() == set()
