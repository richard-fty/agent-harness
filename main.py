"""Agent Harness — CLI entry point with interactive REPL.

Usage:
    uv run python main.py                          # Interactive mode (like Claude Code)
    uv run python main.py "Analyze AAPL"            # Single-shot mode
    uv run python main.py --model deepseek/deepseek-chat "Analyze AAPL"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
import uuid
from typing import Any, Callable

# Suppress noisy logs from libraries
logging.basicConfig(level=logging.ERROR)
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

import litellm
litellm.suppress_debug_info = True
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agent.context.manager import ContextManager
from agent.models import AgentEvent, EventType, TokenUsage
from agent.prompts import build_system_prompt
from agent.skill_loader import SkillLoader
from agent.tool_dispatch import ToolDispatch
from harness.access_control import AccessController, AccessPolicy, get_policy, PRESET_POLICIES
from harness.cost_tracker import CostTracker
from harness.runtime import RuntimeConfig, RuntimeGuard
from harness.token_tracker import extract_usage
from harness.trace import Trace
from tools.base import get_all_builtin_tools
from tools.skill_meta import SkillMetaTools
from config import settings

console = Console()

# Skills that modify the tool registry
_SKILL_MUTATING_TOOLS = {"load_skill", "unload_skill"}


class AgentSession:
    """Persistent agent session — maintains state across conversation turns."""

    def __init__(
        self,
        model: str,
        context_strategy: str,
        runtime_config: RuntimeConfig,
        access_policy: AccessPolicy | None = None,
        cost_budget: float | None = None,
    ) -> None:
        self.model = model
        self.context_strategy = context_strategy
        self.runtime_config = runtime_config

        # Tool dispatch + skills
        self.dispatch = ToolDispatch()
        for tool in get_all_builtin_tools():
            self.dispatch.register(tool.to_tool_def(), tool.execute)

        self.skill_loader = SkillLoader(self.dispatch)
        self.skill_loader.discover()

        meta_tools = SkillMetaTools(self.skill_loader)
        for tool_def, handler in meta_tools.get_tool_pairs():
            self.dispatch.register(tool_def, handler)

        # Context manager
        self.context_mgr = ContextManager(strategy_name=context_strategy, model=model)

        # Harness: access control
        self.access_controller = AccessController(
            policy=access_policy or AccessPolicy()
        )

        # Harness: cost tracker
        self.cost_tracker = CostTracker(model=model, budget_usd=cost_budget)

        # Conversation history (persists across turns)
        system_prompt = build_system_prompt(self.skill_loader)
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]

        # Cumulative metrics
        self.total_usage = TokenUsage()
        self.turn_count = 0

    async def run_turn(self, user_input: str) -> str | None:
        """Process one user turn — may involve multiple LLM calls + tool calls."""
        self.turn_count += 1

        # Pre-load skills by intent before first LLM call
        pre_loaded = self.skill_loader.pre_load_by_intent(user_input)
        if pre_loaded:
            new_prompt = build_system_prompt(self.skill_loader)
            self.messages[0] = {"role": "system", "content": new_prompt}
            for name in pre_loaded:
                console.print(f"  [dim]Skill auto-loaded: {name}[/dim]")

        self.messages.append({"role": "user", "content": user_input})

        guard = RuntimeGuard(self.runtime_config)
        step = 0
        assistant_content = None

        while True:
            # Check limits
            limit_error = guard.check()
            if limit_error:
                console.print(f"\n[bold red]Limit reached:[/bold red] {limit_error}")
                break

            # Context management
            tools_schema = self.dispatch.to_openai_tools()
            fitted_messages = await self.context_mgr.prepare(self.messages, tools_schema)

            console.print(
                f"[dim]  LLM call (step {step}, "
                f"{len(fitted_messages)} msgs, "
                f"{len(tools_schema)} tools)...[/dim]"
            )

            llm_start = time.time()

            try:
                response = await litellm.acompletion(
                    model=self.model,
                    messages=fitted_messages,
                    tools=tools_schema if tools_schema else None,
                )
            except Exception as e:
                console.print(f"\n[bold red]LLM Error:[/bold red] {e}")
                break

            llm_ms = (time.time() - llm_start) * 1000
            usage = extract_usage(response)
            self.total_usage.prompt_tokens += usage.prompt_tokens
            self.total_usage.completion_tokens += usage.completion_tokens
            self.total_usage.total_tokens += usage.total_tokens
            self.total_usage.cost_usd += usage.cost_usd

            # Cost tracking
            self.cost_tracker.add_step(step, usage)
            budget_error = self.cost_tracker.check_budget()
            if budget_error:
                console.print(f"\n[bold red]{budget_error}[/bold red]")
                break

            console.print(
                f"[dim]  Response in {llm_ms:.0f}ms "
                f"(in:{usage.prompt_tokens} out:{usage.completion_tokens} "
                f"${usage.cost_usd:.4f})[/dim]"
            )

            choice = response.choices[0]
            assistant_msg = choice.message

            # Handle tool calls
            if assistant_msg.tool_calls:
                # Show what the assistant said (if anything) before tool calls
                if assistant_msg.content:
                    console.print(f"\n{assistant_msg.content}")

                # Add assistant message with tool calls to history
                assistant_dict: dict[str, Any] = {"role": "assistant"}
                if assistant_msg.content:
                    assistant_dict["content"] = assistant_msg.content
                assistant_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in assistant_msg.tool_calls
                ]
                self.messages.append(assistant_dict)

                # Execute each tool call
                raw_calls = [
                    {
                        "id": tc.id,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in assistant_msg.tool_calls
                ]
                parsed_calls = self.dispatch.parse_tool_calls(raw_calls)
                skill_changed = False

                for tool_call in parsed_calls:
                    args_short = ", ".join(f"{k}={repr(v)[:40]}" for k, v in tool_call.arguments.items())
                    console.print(f"  [yellow]● {tool_call.name}[/yellow]({args_short})")

                    tool_start = time.time()

                    # Access control check
                    access_denied = self.access_controller.check(tool_call)
                    if access_denied:
                        result_content = f"Access denied: {access_denied}"
                        result_success = False
                    # Validate
                    elif (validation_error := self.dispatch.validate_call(tool_call)):
                        result_content = self.dispatch.retry_prompt(tool_call, validation_error)
                        result_success = False
                    else:
                        result = await self.dispatch.execute(tool_call)
                        result_content = result.content
                        result_success = result.success

                    tool_ms = (time.time() - tool_start) * 1000
                    result_content = self.context_mgr.compact_tool_result(result_content)

                    icon = "[green]✓[/green]" if result_success else "[red]✗[/red]"
                    preview = result_content[:100].replace("\n", " ")
                    console.print(f"  {icon} ({tool_ms:.0f}ms) {preview}")

                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.name,
                        "content": result_content,
                    })

                    if tool_call.name in _SKILL_MUTATING_TOOLS:
                        skill_changed = True

                # Rebuild system prompt if skills changed
                if skill_changed:
                    new_prompt = build_system_prompt(self.skill_loader)
                    self.messages[0] = {"role": "system", "content": new_prompt}

                guard.increment_step()
                step += 1

            else:
                # No tool calls — agent is done with this turn
                assistant_content = assistant_msg.content
                if assistant_content:
                    self.messages.append({"role": "assistant", "content": assistant_content})
                break

        return assistant_content

    def print_status(self) -> None:
        """Print session status bar."""
        loaded = self.skill_loader.get_loaded_skill_names()
        skills_str = ", ".join(loaded) if loaded else "none"
        denied = len(self.access_controller.denied_calls)
        denied_str = f" | [red]Denied: {denied}[/red]" if denied else ""
        budget_str = ""
        if self.cost_tracker.budget_usd is not None:
            remaining = self.cost_tracker.budget_usd - self.cost_tracker.total_cost_usd
            budget_str = f" | Budget: ${remaining:.4f} left"
        console.print(
            f"[dim]Model: {self.model} | "
            f"Skills: {skills_str} | "
            f"Turns: {self.turn_count} | "
            f"Tokens: {self.total_usage.total_tokens} | "
            f"Cost: ${self.cost_tracker.total_cost_usd:.4f}"
            f"{budget_str}{denied_str}[/dim]"
        )


async def interactive_mode(
    model: str,
    strategy: str,
    runtime: RuntimeConfig,
    access_policy: AccessPolicy | None = None,
    cost_budget: float | None = None,
) -> None:
    """Interactive REPL — chat with the agent like Claude Code."""
    policy_name = "custom" if access_policy and access_policy.blocked_tools else "unrestricted"
    budget_str = f"${cost_budget:.2f}" if cost_budget else "unlimited"
    console.print(Panel(
        f"[bold]Agent Harness[/bold] — Interactive Mode\n"
        f"Model: {model} | Strategy: {strategy}\n"
        f"Policy: {policy_name} | Budget: {budget_str}\n"
        f"Commands: /status, /skills, /costs, /access, /quit",
        border_style="magenta",
    ))

    session = AgentSession(model, strategy, runtime, access_policy, cost_budget)

    while True:
        try:
            console.print()
            user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.lower() in ("/quit", "/exit", "/q"):
            console.print("[dim]Goodbye.[/dim]")
            break
        elif user_input.lower() == "/status":
            session.print_status()
            continue
        elif user_input.lower() == "/skills":
            loaded = session.skill_loader.get_loaded_skill_names()
            available = session.skill_loader.get_available_skill_names()
            console.print(f"Available: {', '.join(available)}")
            console.print(f"Loaded: {', '.join(loaded) if loaded else 'none'}")
            continue
        elif user_input.lower() == "/costs":
            summary = session.cost_tracker.summary()
            table = Table(title="Cost Summary", border_style="blue")
            table.add_column("Metric", style="bold")
            table.add_column("Value")
            for k, v in summary.items():
                table.add_row(k, str(v))
            console.print(table)
            continue
        elif user_input.lower() == "/access":
            summary = session.access_controller.summary()
            console.print(f"Total calls: {summary['total_calls']}")
            console.print(f"Call counts: {summary['call_counts']}")
            if summary['denied_calls']:
                console.print(f"[red]Denied: {summary['denied_calls']}[/red]")
            else:
                console.print("No denied calls")
            continue

        # Run agent turn
        console.print()
        output = await session.run_turn(user_input)

        if output:
            console.print()
            console.print(Panel(
                Markdown(output),
                title="[bold green]Agent[/bold green]",
                border_style="green",
            ))

        session.print_status()


async def single_shot_mode(
    prompt: str,
    model: str,
    strategy: str,
    runtime: RuntimeConfig,
    access_policy: AccessPolicy | None = None,
    cost_budget: float | None = None,
) -> None:
    """Single-shot mode — one prompt, one response."""
    console.print(Panel(f"[bold]{prompt}[/bold]", title="[bold magenta]Agent Harness[/bold magenta]", border_style="magenta"))

    session = AgentSession(model, strategy, runtime, access_policy, cost_budget)
    output = await session.run_turn(prompt)

    if output:
        console.print()
        console.print(Panel(
            Markdown(output),
            title="[bold green]Agent Output[/bold green]",
            border_style="green",
        ))

    session.print_status()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Agent Harness CLI")
    parser.add_argument("prompt", nargs="?", default=None, help="User prompt (omit for interactive mode)")
    parser.add_argument("--model", "-m", default=settings.default_model, help="LiteLLM model ID")
    parser.add_argument("--strategy", "-s", default=settings.context_strategy, help="Context strategy")
    parser.add_argument("--max-steps", type=int, default=settings.max_steps, help="Max steps per turn")
    parser.add_argument("--timeout", type=int, default=settings.timeout_seconds, help="Timeout per turn")
    parser.add_argument("--policy", "-p", default="unrestricted",
                        help=f"Access policy: {', '.join(PRESET_POLICIES.keys())}")
    parser.add_argument("--budget", type=float, default=None,
                        help="Cost budget in USD (e.g. 0.05)")
    args = parser.parse_args()

    runtime = RuntimeConfig(max_steps=args.max_steps, timeout_seconds=args.timeout)
    access_policy = get_policy(args.policy)

    if args.prompt:
        await single_shot_mode(args.prompt, args.model, args.strategy, runtime, access_policy, args.budget)
    else:
        await interactive_mode(args.model, args.strategy, runtime, access_policy, args.budget)


if __name__ == "__main__":
    asyncio.run(main())
