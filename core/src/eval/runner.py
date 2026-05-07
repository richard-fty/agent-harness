"""Benchmark runner — run model x scenario x strategy matrix.

Usage:
    uv run python -m harness.runner
    uv run python -m harness.runner --models deepseek/deepseek-chat,gpt-4o
    uv run python -m harness.runner --cases single_stock_analysis,chart_generation
    uv run python -m harness.runner --strategies truncate,summary,tiered
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.ERROR)
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

import litellm
litellm.suppress_debug_info = True

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from agent.runtime.loop import run_agent
from agent.runtime.guards import RuntimeConfig
from eval.comparator import (
    compare_against_baseline,
    compare_results,
    format_regression_gate,
    load_baseline,
    save_baseline,
)
from eval.report import generate_report
from scenarios.lt1_equity_briefing.docx_utils import create_briefing_docx, write_placeholder_png
from scenarios.registry import get_scenario, list_scenarios
from config import settings
from agent.runtime.sandbox import DockerSandbox, LocalSandbox, SandboxMount
from agent.runtime.trace import Trace

console = Console()


def _lt1_mock_news() -> list[dict[str, str]]:
    return [
        {
            "title": "NVIDIA quarterly results",
            "url": "https://example.com/nvda/earnings",
            "snippet": "Revenue and margin update",
        },
        {
            "title": "NVIDIA SEC filing",
            "url": "https://example.com/nvda/filing",
            "snippet": "10-Q filing",
        },
        {
            "title": "Reuters on NVIDIA",
            "url": "https://example.com/nvda/reuters",
            "snippet": "Analyst and market reaction",
        },
        {
            "title": "FT on AI spending",
            "url": "https://example.com/nvda/ft",
            "snippet": "Datacenter demand context",
        },
        {
            "title": "Bloomberg on semis",
            "url": "https://example.com/nvda/bloomberg",
            "snippet": "Sector positioning",
        },
    ]


def _make_mock_tool_execute(original_execute: Any, test_case: dict[str, Any]):
    from agent.core.models import ToolResult

    async def mock_execute(tool_call):
        if tool_call.name == "write_file":
            return await original_execute(tool_call)

        if test_case.get("id") == "lt1_brief_nvda":
            if tool_call.name == "web_research":
                payload = {
                    "query": tool_call.arguments.get("query", "NVDA earnings filings analyst news"),
                    "results": [
                        {**item, "text": f"Fetched text for {item['title']}."}
                        if idx < 3 else item
                        for idx, item in enumerate(_lt1_mock_news())
                    ],
                }
                return ToolResult(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    content=json.dumps(payload, indent=2),
                    success=True,
                    summary="Mock web research completed.",
                )

            if tool_call.name == "fetch_market_data":
                payload = {
                    "symbol": "NVDA",
                    "period": "6mo",
                    "interval": "1d",
                    "data_points": 126,
                    "latest": {"date": "2026-04-20", "close": 127.84, "volume": 48200000},
                    "stats": {"period_high": 153.11, "period_low": 95.42, "price_change_pct": 18.7},
                }
                return ToolResult(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    content=json.dumps(payload, indent=2),
                    success=True,
                    summary="Mock market data fetched.",
                )

            if tool_call.name == "compute_indicator":
                indicator = str(tool_call.arguments.get("indicator", "RSI")).upper()
                payload = {"symbol": "NVDA", "indicator": indicator}
                if indicator == "RSI":
                    payload |= {"latest_value": 61.4, "signal": "neutral"}
                elif indicator == "SMA":
                    payload |= {"latest_value": 121.3, "signal": "price above SMA"}
                return ToolResult(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    content=json.dumps(payload, indent=2),
                    success=True,
                    summary=f"Mock {indicator} computed.",
                )

            if tool_call.name == "generate_chart":
                chart_path = Path("results/lt1_briefing/NVDA_chart.png")
                write_placeholder_png(chart_path)
                payload = {
                    "chart_saved": str(chart_path),
                    "symbol": "NVDA",
                    "period": "6mo",
                    "indicators": ["sma_50", "sma_200", "rsi"],
                }
                return ToolResult(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    content=json.dumps(payload, indent=2),
                    success=True,
                    summary="Mock chart generated.",
                )

            if tool_call.name == "run_command":
                chart_path = Path("results/lt1_briefing/NVDA_chart.png")
                write_placeholder_png(chart_path)
                create_briefing_docx(
                    "results/lt1_briefing/NVDA_briefing.docx",
                    title="NVDA - Equity Research Briefing",
                    summary="NVIDIA remains levered to AI infrastructure demand, with recent results supporting continued revenue momentum.",
                    interpretation="Price is above the medium-term trend, RSI is neutral, and the attached chart captures the six-month move.",
                    news_items=_lt1_mock_news(),
                    risks=[
                        "Customer concentration in hyperscale AI demand can amplify cyclical downside.",
                        "Export controls and supply-chain constraints remain material operational risks.",
                    ],
                    sources=[{"url": item["url"]} for item in _lt1_mock_news()],
                    chart_path=chart_path,
                )
                return ToolResult(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    content="Rendered results/lt1_briefing/NVDA_briefing.docx\n[exit code: 0]",
                    success=True,
                    summary="Mock render command completed.",
                )

        from eval.mock_brain import get_mock_tool_response

        response = get_mock_tool_response(tool_call.name, tool_call.arguments)
        return ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            content=response,
            success=True,
            summary=response[:240],
        )

    return mock_execute


def load_test_cases(
    scenario_name: str,
    case_ids: list[str] | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    """Load test cases from a named scenario."""
    scenario = get_scenario(scenario_name)
    cases = scenario.get_test_cases()
    if case_ids:
        cases = [c for c in cases if c.get("id") in case_ids]
    return scenario, cases


async def run_single(
    scenario: Any,
    model: str,
    test_case: dict[str, Any],
    strategy: str,
    timeout: int = 300,
    use_mock: bool = False,
) -> tuple[Trace, dict[str, Any] | None]:
    """Run a single model x test_case x strategy combination.
    
    Supports LT1 tier: kill at midpoint and wake() for continuation.
    Supports mock mode for CI testing without API costs.
    """
    from agent.runtime.managed_runtime import ManagedAgentRuntime
    from agent.runtime.wake import wake
    from agent.session.archive import SessionArchive
    
    runtime = RuntimeConfig(
        max_steps=test_case.get("max_steps", 20),
        timeout_seconds=timeout,
    )
    
    # Check for LT1 tier (long-task checkpoint/resume)
    tier = test_case.get("tier")
    if tier == "LT1":
        # LT1: Run until midpoint, kill, wake(), continue without duplicate work
        return await _run_lt1(scenario, model, test_case, strategy, runtime, use_mock=use_mock)

    if use_mock:
        # Mock mode: use deterministic responses without real API calls
        return await _run_mock(scenario, model, test_case, strategy, runtime)

    trace = await run_agent(
        user_input=test_case["input"],
        model=model,
        context_strategy=strategy,
        runtime_config=runtime,
    )
    trace.scenario = scenario.name

    score: dict[str, Any] | None = None
    if not scenario.is_app_artifact():
        score = scenario.evaluate(trace, test_case)
        score["model"] = model
        score["context_strategy"] = strategy
        score["scenario"] = scenario.name

    # Save trace
    trace.save("results")

    return trace, score


async def _run_mock(
    scenario: Any,
    model: str,
    test_case: dict[str, Any],
    strategy: str,
    runtime: RuntimeConfig,
) -> tuple[Trace, dict[str, Any] | None]:
    """Run a test case with mock LLM and mock tool responses (no API calls)."""
    from agent.session.archive import SessionArchive
    from agent.session.engine import SessionEngine
    from agent.runtime.orchestrator import SessionOrchestrator
    from agent.runtime.guards import RuntimeGuard
    from agent.runtime.trace import Trace
    from eval.mock_brain import inject_mock_brain
    
    console.print("  [dim]MOCK MODE: Using deterministic responses[/dim]")
    
    # Create archive and session
    archive = SessionArchive()
    session = SessionEngine(model=model, context_strategy=strategy)
    original_execute = session.dispatch.execute
    session.dispatch.execute = _make_mock_tool_execute(original_execute, test_case)
    
    # Create orchestrator and runtime
    orchestrator = SessionOrchestrator(archive=archive)
    handle = orchestrator.create_runtime(
        session_engine=session,
        model=model,
        runtime_config=runtime,
    )
    
    # Inject mock brain
    inject_mock_brain(handle.runtime, test_case)
    
    # Create trace
    trace = Trace(
        run_id=f"mock_{test_case.get('id', 'unknown')}",
        model=model,
        scenario=scenario.name,
        prompt=test_case["input"],
        context_strategy=strategy,
    )
    
    # Run to completion
    guard = RuntimeGuard(runtime)
    await handle.runtime.run_to_completion(
        user_input=test_case["input"],
        guard=guard,
        trace=trace,
        callback=lambda e: None,
    )
    
    score: dict[str, Any] | None = None
    if not scenario.is_app_artifact():
        score = scenario.evaluate(trace, test_case)
        score["model"] = model
        score["context_strategy"] = strategy
        score["scenario"] = scenario.name
        score["mock_mode"] = True
    
    # Save trace
    trace.save("results")
    
    return trace, score


async def _run_lt1(
    scenario: Any,
    model: str,
    test_case: dict[str, Any],
    strategy: str,
    runtime: RuntimeConfig,
    use_mock: bool = False,
) -> tuple[Trace, dict[str, Any] | None]:
    """LT1 tier: checkpoint at midpoint, wake(), continue without duplicate work."""
    from agent.runtime.orchestrator import SessionOrchestrator
    from agent.runtime.guards import RuntimeGuard
    from agent.runtime.trace import Trace
    from agent.runtime.wake import wake
    from agent.session.archive import SessionArchive
    from agent.session.engine import SessionEngine
    
    mode_str = "MOCK " if use_mock else ""
    console.print(f"  [dim]LT1: {mode_str}Running with checkpoint/resume...[/dim]")

    archive = SessionArchive()

    session = SessionEngine(model=model, context_strategy=strategy, archive=archive)
    orchestrator = SessionOrchestrator(archive=archive)
    handle = orchestrator.create_runtime(
        session_engine=session,
        model=model,
        runtime_config=runtime,
    )
    rt = handle.runtime

    if use_mock:
        from eval.mock_brain import inject_mock_brain

        inject_mock_brain(rt, test_case)
        original_execute = rt.session_engine.dispatch.execute
        rt.session_engine.dispatch.execute = _make_mock_tool_execute(original_execute, test_case)

    expected_tools = test_case.get("expected_tools", [])
    kill_after_tools = max(1, len(expected_tools) // 2) if expected_tools else max(1, test_case.get("max_steps", 20) // 2)
    trace = Trace(
        run_id=f"lt1_{test_case.get('id', 'unknown')}",
        model=model,
        scenario=scenario.name,
        prompt=test_case["input"],
        context_strategy=strategy,
    )

    final_content = ""
    tool_finishes = 0
    async for event in rt.start_turn(test_case["input"], guard=RuntimeGuard(runtime), trace=trace):
        final_content = rt._map_event_to_trace(event, trace, lambda e: None, final_content)
        if event.type == "tool_finished":
            tool_finishes += 1
            if tool_finishes >= kill_after_tools:
                console.print(f"  [dim]LT1: Killing after tool {tool_finishes}...[/dim]")
                break

    rt._persist_session()
    session_id = rt.session.session_id

    events_before_wake = archive.get_events(session_id)
    tool_calls_before = [
        e for e in events_before_wake
        if e.get("type") == "tool_finished"
    ]
    console.print(f"  [dim]LT1: {len(tool_calls_before)} tool calls before wake[/dim]")

    rt2 = wake(archive, session_id, runtime_config=runtime)

    if use_mock:
        from eval.mock_brain import inject_mock_brain

        inject_mock_brain(rt2, test_case)
        original_execute = rt2.session_engine.dispatch.execute
        rt2.session_engine.dispatch.execute = _make_mock_tool_execute(original_execute, test_case)

    final_content = ""
    async for event in rt2.start_turn("continue", guard=RuntimeGuard(runtime), trace=trace):
        final_content = rt2._map_event_to_trace(event, trace, lambda e: None, final_content)

    events_after_wake = archive.get_events(session_id)
    tool_calls_after = [
        e for e in events_after_wake
        if e.get("type") == "tool_finished"
    ]

    unique_tools = set()
    duplicates = []
    for tc in tool_calls_after:
        payload = tc.get("payload", {})
        signature = (
            payload.get("tool_name") or payload.get("name"),
            json.dumps(payload.get("arguments", {}), sort_keys=True),
        )
        if signature in unique_tools:
            duplicates.append(signature[0])
        unique_tools.add(signature)

    score: dict[str, Any] | None = None
    if not scenario.is_app_artifact():
        score = scenario.evaluate(trace, test_case)
        score["model"] = model
        score["context_strategy"] = strategy
        score["scenario"] = scenario.name
        score["lt1_checkpoint_step"] = kill_after_tools
        score["lt1_tool_calls_before_wake"] = len(tool_calls_before)
        score["lt1_tool_calls_total"] = len(tool_calls_after)
        score["lt1_duplicate_calls"] = len(duplicates)
        score["lt1_success"] = len(duplicates) == 0 and rt2.session.state.value == "completed"

    if duplicates:
        console.print(f"  [bold red]LT1 FAILED: Duplicate tool calls: {duplicates}[/bold red]")
    else:
        console.print(f"  [dim]LT1: Success - {len(tool_calls_after)} total tool calls, no duplicates[/dim]")

    trace.save("results")
    return trace, score


async def run_benchmark(
    scenario: Any,
    models: list[str],
    test_cases: list[dict[str, Any]],
    strategies: list[str],
    timeout: int = 300,
    use_mock: bool = False,
) -> list[dict[str, Any]]:
    """Run the full benchmark matrix."""
    total = len(models) * len(test_cases) * len(strategies)
    results: list[dict[str, Any]] = []

    console.print(Panel(
        f"Models: {', '.join(models)}\n"
        f"Scenario: {scenario.name}\n"
        f"Test cases: {len(test_cases)}\n"
        f"Strategies: {', '.join(strategies)}\n"
        f"Total runs: {total}",
        title="[bold cyan]Benchmark Runner[/bold cyan]",
        border_style="cyan",
    ))

    run_num = 0
    for model in models:
        for strategy in strategies:
            for case in test_cases:
                run_num += 1
                case_id = case.get("id", "?")
                console.print(
                    f"\n[bold]Run {run_num}/{total}:[/bold] "
                    f"{model} | {strategy} | {case_id}"
                )
                console.print(f"[dim]  Prompt: {case['input'][:80]}...[/dim]")

                start = time.time()
                try:
                    _, score = await run_single(scenario, model, case, strategy, timeout, use_mock=use_mock)
                    if score is None:
                        raise RuntimeError(f"Scenario {scenario.name} did not produce an inline score")
                    elapsed = time.time() - start

                    # Print quick result
                    total_score = score["total_score"]
                    color = "green" if total_score >= 0.7 else "yellow" if total_score >= 0.4 else "red"
                    console.print(
                        f"  [{color}]Score: {total_score:.3f}[/{color}] | "
                        f"Steps: {score['steps']} | "
                        f"Tokens: {score['tokens']} | "
                        f"Cost: ${score['cost_usd']:.4f} | "
                        f"Time: {elapsed:.1f}s"
                    )

                    results.append(score)

                except Exception as e:
                    console.print(f"  [bold red]Error: {e}[/bold red]")
                    results.append({
                        "test_case_id": case_id,
                        "model": model,
                        "context_strategy": strategy,
                        "total_score": 0.0,
                        "error": str(e),
                    })

    return results


async def run_app_artifact_benchmark(
    scenario: Any,
    test_cases: list[dict[str, Any]],
    *,
    model: str,
    strategy: str,
    timeout: int,
    replay: str | None,
    use_mock: bool,
    output_dir: str,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    scenario_dir = Path(__file__).resolve().parents[1] / "scenarios" / scenario.name
    out_root = Path(output_dir).resolve()
    sandbox_root = out_root / "sandbox"
    trace_root = out_root / "traces"
    sandbox_root.mkdir(parents=True, exist_ok=True)
    trace_root.mkdir(parents=True, exist_ok=True)

    selected = test_cases
    if replay and replay != "all":
        selected = [case for case in test_cases if case.get("id") == replay]
    if replay and not selected:
        raise ValueError(f"Replay case not found: {replay}")

    for case in selected:
        case_id = case["id"]
        console.print(f"\n[bold]App artifact case:[/bold] {scenario.name}/{case_id}")
        start = time.time()
        workspace = _prepare_app_artifact_workspace(case, scenario_dir, sandbox_root)
        trace = Trace(
            run_id=f"{scenario.name}_{case_id}_{int(start)}",
            model=model,
            scenario=scenario.name,
            prompt=case["input"],
            context_strategy=strategy,
        )
        try:
            if replay or use_mock:
                _apply_patch_file(workspace, _case_path(scenario_dir, case["golden_patch"]))
            else:
                old_cwd = Path.cwd()
                try:
                    os.chdir(workspace)
                    trace, _ = await run_single(
                        scenario,
                        model,
                        case,
                        strategy,
                        timeout=timeout,
                        use_mock=False,
                    )
                finally:
                    os.chdir(old_cwd)

            trace.gate_results = await _run_app_artifact_gates(
                case,
                scenario_dir,
                workspace,
                timeout=timeout,
            )
            trace.artifacts.extend(
                _capture_app_artifact_outputs(case, scenario_dir, workspace, out_root)
            )
            preview_url = _resolve_preview_url(trace)
            if preview_url:
                trace.artifacts.append({"kind": "app_preview", "url": preview_url})
            trace.finish(output="coding gates completed")
            score = scenario.evaluate(trace, case)
            score["model"] = model
            score["context_strategy"] = strategy
            score["scenario"] = scenario.name
            score["mock_mode"] = use_mock
            score["replay_mode"] = bool(replay)
            score["duration_seconds"] = time.time() - start
            results.append(score)
        except Exception as exc:
            trace.finish(error=str(exc))
            results.append({
                "test_case_id": case_id,
                "model": model,
                "context_strategy": strategy,
                "scenario": scenario.name,
                "total_score": 0.0,
                "install_passed": False,
                "build_passed": False,
                "test_passed": False,
                "duration_seconds": time.time() - start,
                "cost_usd": trace.total_usage.cost_usd,
                "error": str(exc),
            })
        finally:
            trace.save(str(trace_root))

    csv_path = _write_coding_csv(results, out_root)
    console.print(f"[dim]App artifact CSV saved to {csv_path}[/dim]")
    return results


def _prepare_app_artifact_workspace(case: dict[str, Any], scenario_dir: Path, sandbox_root: Path) -> Path:
    template = scenario_dir / "templates" / case["template"]
    workspace = sandbox_root / case["id"]
    if workspace.exists():
        shutil.rmtree(workspace)
    shutil.copytree(template, workspace)
    case_src = _case_path(scenario_dir, case["golden_patch"]).parents[1]
    public_src = case_src / "public"
    if public_src.exists():
        shutil.copytree(public_src, workspace, dirs_exist_ok=True)
    cases_dst = workspace / "cases"
    cases_dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        case_src,
        cases_dst / case_src.name,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("hidden"),
    )
    _git(workspace, "init")
    _git(workspace, "config", "user.email", "apex@example.invalid")
    _git(workspace, "config", "user.name", "Apex Eval")
    _git(workspace, "add", ".")
    _git(workspace, "commit", "-m", "baseline")
    return workspace


def _case_path(scenario_dir: Path, rel: str) -> Path:
    return scenario_dir / rel


def _apply_patch_file(workspace: Path, patch_path: Path) -> None:
    result = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", str(patch_path)],
        cwd=workspace,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout or f"git apply failed: {patch_path}")


async def _run_app_artifact_gates(
    case: dict[str, Any],
    scenario_dir: Path,
    workspace: Path,
    *,
    timeout: int,
) -> dict[str, bool]:
    manifest = case["manifest"]
    sandbox = _make_gate_sandbox(workspace)
    log_dir = workspace / ".gate_logs"
    log_dir.mkdir(exist_ok=True)
    results: dict[str, bool] = {}
    case_root = _case_path(scenario_dir, case["golden_patch"]).parents[1]
    hidden_src = case_root / "hidden"
    hidden_dst = workspace / "cases" / case_root.name / "hidden"
    for stage in ("install", "build", "test"):
        cmd_parts = manifest.get(stage)
        if not cmd_parts:
            results[stage] = True
            continue
        injected = False
        if stage == "test" and hidden_src.exists():
            shutil.copytree(hidden_src, hidden_dst, dirs_exist_ok=True)
            injected = True
        try:
            cmd = _stage_command(cmd_parts, case)
            network = "bridge" if stage == "install" else "none"
            result = await sandbox.run_oneshot(cmd, timeout=timeout, network=network)
            (log_dir / f"{stage}.stdout").write_text(result.stdout or "", encoding="utf-8")
            (log_dir / f"{stage}.stderr").write_text(result.stderr or "", encoding="utf-8")
            results[stage] = result.exit_code == 0 and not result.timed_out
            if not results[stage]:
                break
        finally:
            if injected and hidden_dst.exists():
                shutil.rmtree(hidden_dst, ignore_errors=True)
    return results


def _make_gate_sandbox(workspace: Path):
    gate_sandbox = os.environ.get("APEX_GATE_SANDBOX", "").strip().lower()
    if gate_sandbox in {"local", "host"}:
        return _make_local_gate_sandbox(workspace)
    if gate_sandbox in {"docker", "container"} and not _docker_daemon_available():
        raise RuntimeError("APEX_GATE_SANDBOX=docker was requested, but Docker is not available")
    if _docker_daemon_available():
        image = os.environ.get("APEX_SANDBOX_IMAGE", "apex-sandbox:latest")
        pnpm_store = workspace.parent.parent / ".pnpm-store"
        pnpm_store.mkdir(parents=True, exist_ok=True)
        return DockerSandbox(
            image=image,
            work_dir="/workspace",
            network="none",
            mounts=[
                SandboxMount(source=str(workspace), target="/workspace", read_only=False),
                SandboxMount(source=str(pnpm_store), target="/pnpm-store", read_only=False),
            ],
        )
    return _make_local_gate_sandbox(workspace)


def _make_local_gate_sandbox(workspace: Path) -> LocalSandbox:
    return LocalSandbox(
        workspace_root=str(workspace),
        env_allowlist={
            "PATH",
            "LANG",
            "LC_ALL",
            "TERM",
            "PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH",
        },
    )


def _docker_daemon_available() -> bool:
    if not shutil.which("docker"):
        return False
    result = subprocess.run(
        ["docker", "version", "--format", "{{.Server.Version}}"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _stage_command(parts: list[str], case: dict[str, Any]) -> str:
    return shlex.join(parts)


def _capture_app_artifact_outputs(
    case: dict[str, Any],
    scenario_dir: Path,
    workspace: Path,
    out_root: Path,
) -> list[dict[str, Any]]:
    del scenario_dir
    out_dir = out_root / "artifacts" / case["id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    captured: list[dict[str, Any]] = []
    patch = subprocess.run(["git", "diff"], cwd=workspace, capture_output=True, text=True).stdout
    patch_path = out_dir / "patch.diff"
    patch_path.write_text(patch, encoding="utf-8")
    captured.append({"kind": "code", "language": "diff", "path": str(patch_path)})

    for stage in ("install", "build", "test"):
        for stream in ("stdout", "stderr"):
            log = workspace / f".gate_logs/{stage}.{stream}"
            if log.exists():
                dst = out_dir / f"{stage}.{stream}.log"
                shutil.copy2(log, dst)
                captured.append({
                    "kind": "log",
                    "stage": stage,
                    "stream": stream,
                    "path": str(dst),
                })

    test_stdout = workspace / ".gate_logs" / "test.stdout"
    if test_stdout.exists():
        json_blob = _extract_playwright_json(
            test_stdout.read_text(encoding="utf-8", errors="replace")
        )
        if json_blob is not None:
            dst = out_dir / "playwright.json"
            dst.write_text(json_blob, encoding="utf-8")
            captured.append({"kind": "test_report", "format": "playwright_json", "path": str(dst)})

    screenshots = workspace / "test-results"
    if screenshots.exists():
        for png in screenshots.rglob("*.png"):
            dst = out_dir / "screenshots" / png.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(png, dst)
            captured.append({"kind": "screenshot", "path": str(dst)})

    return captured


def _extract_playwright_json(text: str) -> str | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            _, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        return text[index:index + end]
    return None


def _resolve_preview_url(trace: Trace) -> str | None:
    preview_id: str | None = None
    for event in trace.artifact_events:
        data = event.get("data", {})
        if event.get("event_type") == "artifact_created" and data.get("kind") == "app_preview":
            preview_id = data.get("artifact_id")
        elif (
            event.get("event_type") == "artifact_patch"
            and preview_id is not None
            and data.get("artifact_id") == preview_id
            and data.get("op") == "replace"
        ):
            return data.get("content")
    return None


def _write_coding_csv(results: list[dict[str, Any]], out_root: Path) -> Path:
    path = out_root / f"coding_eval_{int(time.time())}.csv"
    fields = [
        "case_id",
        "install_passed",
        "build_passed",
        "test_passed",
        "score",
        "duration_sec",
        "cost_usd",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({
                "case_id": row.get("test_case_id"),
                "install_passed": row.get("install_passed", False),
                "build_passed": row.get("build_passed", False),
                "test_passed": row.get("test_passed", False),
                "score": row.get("total_score", 0.0),
                "duration_sec": round(float(row.get("duration_seconds", 0.0)), 3),
                "cost_usd": row.get("cost_usd", 0.0),
            })
    return path


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def print_results_table(results: list[dict[str, Any]]) -> None:
    """Print results as a Rich table."""
    table = Table(title="Benchmark Results", border_style="blue")
    table.add_column("Model", style="bold")
    table.add_column("Strategy")
    table.add_column("Test Case")
    table.add_column("Score", justify="right")
    table.add_column("Steps", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Time", justify="right")

    for r in results:
        score = r.get("total_score", 0)
        color = "green" if score >= 0.7 else "yellow" if score >= 0.4 else "red"
        table.add_row(
            r.get("model", "?"),
            r.get("context_strategy", "?"),
            r.get("test_case_id", "?"),
            f"[{color}]{score:.3f}[/{color}]",
            str(r.get("steps", "?")),
            str(r.get("tokens", "?")),
            f"${r.get('cost_usd', 0):.4f}",
            f"{r.get('duration_seconds', 0):.1f}s",
        )

    console.print(table)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Agent Benchmark Runner")
    parser.add_argument(
        "--scenario", default="stock_strategy",
        help=f"Scenario name: {', '.join(list_scenarios())}",
    )
    parser.add_argument(
        "--models", default=settings.default_model,
        help="Comma-separated model IDs (e.g. deepseek/deepseek-chat,gpt-4o)",
    )
    parser.add_argument(
        "--cases", default=None,
        help="Comma-separated test case IDs (default: all)",
    )
    parser.add_argument(
        "--strategies", default="truncate",
        help="Comma-separated context strategies (e.g. truncate,summary,tiered)",
    )
    parser.add_argument(
        "--timeout", type=int, default=300,
        help="Timeout per run in seconds",
    )
    parser.add_argument(
        "--output", default="results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--baseline", default=None,
        help="Path to a saved baseline JSON for regression gating",
    )
    parser.add_argument(
        "--update-baseline", action="store_true",
        help="Write the current results to --baseline after the run completes",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use mock LLM and tool responses (no API calls, for CI testing)",
    )
    parser.add_argument(
        "--replay",
        nargs="?",
        const="all",
        default=None,
        help="For coding scenario: skip agent, apply golden patch, and run gates. Optional case id.",
    )
    args = parser.parse_args()
    
    # Check for mock mode via environment variable or CLI flag
    use_mock = args.mock or os.environ.get("APEX_MOCK_LLM", "").lower() in ("1", "true", "yes")
    if use_mock:
        console.print("[bold yellow]MOCK MODE: Using deterministic responses (no API calls)[/bold yellow]")

    models = [m.strip() for m in args.models.split(",")]
    strategies = [s.strip() for s in args.strategies.split(",")]
    case_ids = [c.strip() for c in args.cases.split(",")] if args.cases else None
    scenario, test_cases = load_test_cases(args.scenario, case_ids)

    if not test_cases:
        console.print("[bold red]No test cases found.[/bold red]")
        return

    if scenario.is_app_artifact():
        if len(models) > 1 or len(strategies) > 1:
            console.print("[yellow]App-artifact runner uses the first model and strategy only.[/yellow]")
        results = await run_app_artifact_benchmark(
            scenario,
            test_cases,
            model=models[0],
            strategy=strategies[0],
            timeout=args.timeout,
            replay=args.replay,
            use_mock=use_mock,
            output_dir=args.output,
        )
        console.print()
        print_results_table(results)
        if args.replay and any(r.get("total_score", 0.0) < 1.0 for r in results):
            raise SystemExit(1)
        return

    # Run benchmark
    results = await run_benchmark(scenario, models, test_cases, strategies, args.timeout, use_mock=use_mock)

    # Print results table
    console.print()
    print_results_table(results)

    # Print comparison if multiple models/strategies
    if len(models) > 1 or len(strategies) > 1:
        console.print()
        comparison = compare_results(results)
        for line in comparison:
            console.print(line)

    if args.baseline:
        console.print()
        baseline = load_baseline(args.baseline) if Path(args.baseline).exists() else []
        gate = compare_against_baseline(results, baseline)
        for line in format_regression_gate(gate):
            console.print(line)
        if args.update_baseline:
            baseline_path = save_baseline(
                results, 
                args.baseline,
                scenario=scenario.name,
                model=models[0] if models else "unknown",
                strategy=strategies[0] if strategies else "unknown"
            )
            console.print(f"[dim]Baseline updated at {baseline_path}[/dim]")
        elif not gate["passed"]:
            raise SystemExit(1)
    elif args.update_baseline:
        raise SystemExit("--update-baseline requires --baseline")

    # Generate and save report
    report_path = generate_report(results, args.output)
    console.print(f"\n[dim]Report saved to {report_path}[/dim]")


if __name__ == "__main__":
    asyncio.run(main())
