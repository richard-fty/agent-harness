"""Agent Harness TUI — clean, minimal design inspired by Claude Code.

No color borders. No labels. Clean whitespace. Spinner above prompt while thinking.

Usage:
    uv run python -m tui.app
    uv run python -m tui.app --model gpt-4o --budget 0.05
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from typing import Any

logging.basicConfig(level=logging.ERROR)
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

import litellm
litellm.suppress_debug_info = True

from rich.markdown import Markdown
from rich.text import Text
from rich.table import Table

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Footer, Input, Static
from textual.reactive import reactive
from textual.timer import Timer

from agent.context.manager import ContextManager
from agent.models import TokenUsage
from agent.prompts import build_system_prompt
from agent.skill_loader import SkillLoader
from agent.tool_dispatch import ToolDispatch
from harness.access_control import AccessController, AccessPolicy, get_policy
from harness.cost_tracker import CostTracker
from harness.runtime import RuntimeConfig, RuntimeGuard
from harness.token_tracker import extract_usage
from tools.base import get_all_builtin_tools
from tools.skill_meta import SkillMetaTools
from config import settings

_SKILL_MUTATING_TOOLS = {"load_skill", "unload_skill"}

# Spinner frames for the thinking indicator
_SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


class ThinkingIndicator(Static):
    """Animated spinner shown above the prompt while the agent is working."""

    DEFAULT_CSS = """
    ThinkingIndicator {
        height: 1;
        padding: 0 2;
        color: $text-muted;
    }
    """

    _frame = reactive(0)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._timer: Timer | None = None
        self._message = ""
        self._visible = False

    def show(self, message: str = "Thinking") -> None:
        self._message = message
        self._visible = True
        self._frame = 0
        self._timer = self.set_interval(0.08, self._tick)
        self._render_frame()

    def hide(self) -> None:
        self._visible = False
        if self._timer:
            self._timer.stop()
            self._timer = None
        self.update("")

    def _tick(self) -> None:
        self._frame = (self._frame + 1) % len(_SPINNER_FRAMES)
        self._render_frame()

    def _render_frame(self) -> None:
        if self._visible:
            frame = _SPINNER_FRAMES[self._frame]
            self.update(f"  {frame} {self._message}")


class StreamingMarkdown(Static):
    """Re-renders markdown at throttled rate (~12fps) as tokens accumulate."""

    DEFAULT_CSS = """
    StreamingMarkdown {
        padding: 0 2;
        margin: 0;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._buffer = ""
        self._dirty = False
        self._timer: Timer | None = None

    def on_mount(self) -> None:
        self._timer = self.set_interval(0.08, self._flush)

    def _flush(self) -> None:
        if self._dirty and self._buffer.strip():
            try:
                self.update(Markdown(self._buffer))
            except Exception:
                self.update(self._buffer)
            self._dirty = False

    def append_token(self, token: str) -> None:
        self._buffer += token
        self._dirty = True

    def finalize(self) -> None:
        if self._timer:
            self._timer.stop()
        if self._buffer.strip():
            try:
                self.update(Markdown(self._buffer))
            except Exception:
                self.update(self._buffer)

    @property
    def full_text(self) -> str:
        return self._buffer


class MainOutput(VerticalScroll):
    """Main output — clean scrollable area. No borders, no chrome."""

    DEFAULT_CSS = """
    MainOutput {
        height: 1fr;
        padding: 0;
        scrollbar-gutter: stable;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._streaming_widget: StreamingMarkdown | None = None

    def _append(self, widget: Static) -> None:
        self.mount(widget)
        self.scroll_end(animate=False)

    def user_msg(self, text: str) -> None:
        self._append(Static(
            Text.from_markup(f"\n[bold]❯[/bold] {text}\n"),
            classes="user-msg",
        ))

    def agent_thinking(self, text: str) -> None:
        if text:
            self._append(Static(
                Text.from_markup(f"[dim]{text}[/dim]"),
                classes="thinking",
            ))

    def stream_start(self) -> None:
        self._streaming_widget = StreamingMarkdown()
        self._append(self._streaming_widget)

    def stream_token(self, token: str) -> None:
        if self._streaming_widget is not None:
            self._streaming_widget.append_token(token)
            self.scroll_end(animate=False)

    def stream_end(self, full_text: str) -> None:
        if self._streaming_widget is not None:
            self._streaming_widget.finalize()
            self._streaming_widget = None
        self._append(Static(Text("")))

    def tool_call(self, name: str, args: dict, success: bool, duration_ms: float, preview: str) -> None:
        icon = "✓" if success else "✗"
        color = "green" if success else "red"
        args_short = ", ".join(f"{k}={repr(v)[:30]}" for k, v in args.items())
        self._append(Static(Text.from_markup(
            f"  [{color}]{icon}[/{color}] [dim]{name}({args_short}) · {duration_ms:.0f}ms[/dim]"
        ), classes="tool-call"))

    def tool_denied(self, name: str, reason: str) -> None:
        self._append(Static(Text.from_markup(
            f"  [red]⊘[/red] [dim]{name} — {reason}[/dim]"
        ), classes="tool-call"))

    def info(self, text: str) -> None:
        self._append(Static(Text.from_markup(f"[dim]{text}[/dim]"), classes="info"))

    def system_msg(self, text: str) -> None:
        self._append(Static(Text.from_markup(f"\n[dim]{text}[/dim]\n"), classes="system"))

    def error_msg(self, text: str) -> None:
        self._append(Static(Text.from_markup(f"\n[red]{text}[/red]\n"), classes="error"))

    def show_metrics(self, model: str, strategy: str, turns: int, usage: TokenUsage,
                     cost: float, budget: float | None, skills: list[str], denied: int) -> None:
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("", style="dim", width=14)
        table.add_column("")

        model_short = model.split("/")[-1] if "/" in model else model
        table.add_row("Model", model_short)
        table.add_row("Strategy", strategy)
        table.add_row("Turns", str(turns))
        table.add_row("Tokens", f"{usage.total_tokens:,}")
        table.add_row("Cost", f"${cost:.4f}")
        if budget is not None:
            table.add_row("Budget left", f"${budget - cost:.4f}")
        table.add_row("Skills", ", ".join(skills) if skills else "none")
        if denied:
            table.add_row("Denied", str(denied))

        self._append(Static(table))
        self._append(Static(Text("")))


class StatusBar(Static):
    """Minimal status bar — model + cost + tokens."""

    DEFAULT_CSS = """
    StatusBar {
        height: 1;
        dock: bottom;
        background: $surface;
        color: $text-muted;
        padding: 0 2;
    }
    """

    def set(self, model: str, cost: float, tokens: int, skills: list[str]) -> None:
        model_short = model.split("/")[-1] if "/" in model else model
        parts = [model_short, f"${cost:.4f}", f"{tokens:,} tokens"]
        if skills:
            parts.append(f"skills: {', '.join(skills)}")
        self.update(" · ".join(parts))


class AgentHarnessApp(App):
    """Clean, minimal TUI — inspired by Claude Code."""

    TITLE = "Agent Harness"
    CSS = """
    Screen {
        layout: vertical;
        background: $background;
    }

    .user-msg {
        padding: 0 2;
    }

    .thinking {
        padding: 0 2;
    }

    .tool-call {
        padding: 0 2;
    }

    .info {
        padding: 0 2;
    }

    .system {
        padding: 0 2;
    }

    .error {
        padding: 0 2;
    }

    #prompt-input {
        dock: bottom;
        margin: 0 2 0 2;
        border: none;
        background: $surface;
        padding: 0 1;
    }

    #prompt-input:focus {
        border: none;
    }

    Footer {
        display: none;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+l", "clear", "Clear"),
    ]

    def __init__(
        self,
        model: str = "",
        strategy: str = "truncate",
        policy: str = "unrestricted",
        budget: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._model = model or settings.default_model
        self._strategy = strategy
        self._policy_name = policy
        self._budget = budget
        self._init_session()

    def _init_session(self) -> None:
        self._dispatch = ToolDispatch()
        for tool in get_all_builtin_tools():
            self._dispatch.register(tool.to_tool_def(), tool.execute)

        self._skill_loader = SkillLoader(self._dispatch)
        self._skill_loader.discover()

        meta_tools = SkillMetaTools(self._skill_loader)
        for td, handler in meta_tools.get_tool_pairs():
            self._dispatch.register(td, handler)

        self._context_mgr = ContextManager(strategy_name=self._strategy, model=self._model)
        self._access = AccessController(policy=get_policy(self._policy_name))
        self._cost_tracker = CostTracker(model=self._model, budget_usd=self._budget)
        self._total_usage = TokenUsage()
        self._turn_count = 0

        system_prompt = build_system_prompt(self._skill_loader)
        self._messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]

    def compose(self) -> ComposeResult:
        yield MainOutput(id="main-output")
        yield ThinkingIndicator(id="thinking")
        yield StatusBar(id="status-bar")
        yield Input(placeholder="Message...", id="prompt-input")

    def on_mount(self) -> None:
        self._update_status()
        self.query_one("#prompt-input", Input).focus()
        self._show_welcome()

    def _show_welcome(self) -> None:
        output = self.query_one("#main-output", MainOutput)
        model_short = self._model.split("/")[-1] if "/" in self._model else self._model
        skills = self._skill_loader.get_available_skill_names()
        skill_list = ", ".join(skills) if skills else "none"

        output._append(Static(Text.from_markup(
            "\n"
            "  [bold]Agent Harness[/bold]\n"
            "\n"
            f"  General-purpose autonomous agent with benchmark harness.\n"
            f"  Built-in tools for files, shell, and web.\n"
            f"  Skill packs loaded on demand.\n"
            "\n"
            f"  [dim]Model:[/dim]  {model_short}\n"
            f"  [dim]Skills:[/dim] {skill_list}\n"
            f"  [dim]Policy:[/dim] {self._policy_name}\n"
            "\n"
            "  [dim]Tips:[/dim]\n"
            "  [dim]  Ask anything — the agent loads skills as needed[/dim]\n"
            "  [dim]  /model <name> to switch models mid-session[/dim]\n"
            "  [dim]  /help for all commands[/dim]\n"
        )))

    def _update_status(self) -> None:
        self.query_one("#status-bar", StatusBar).set(
            self._model,
            self._cost_tracker.total_cost_usd,
            self._total_usage.total_tokens,
            self._skill_loader.get_loaded_skill_names(),
        )

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.value = ""

        if text.startswith("/"):
            await self._handle_command(text)
            return

        self.query_one("#main-output", MainOutput).user_msg(text)
        self._run_agent_turn(text)

    @work(exclusive=True)
    async def _run_agent_turn(self, user_input: str) -> None:
        output = self.query_one("#main-output", MainOutput)
        thinking = self.query_one("#thinking", ThinkingIndicator)
        input_widget = self.query_one("#prompt-input", Input)

        input_widget.disabled = True
        thinking.show("Thinking")

        self._turn_count += 1

        # Pre-load skills by intent before first LLM call
        pre_loaded = self._skill_loader.pre_load_by_intent(user_input)
        if pre_loaded:
            # Rebuild system prompt with newly loaded skills
            new_prompt = build_system_prompt(self._skill_loader)
            self._messages[0] = {"role": "system", "content": new_prompt}
            for skill_name in pre_loaded:
                output.info(f"  Skill auto-loaded: {skill_name}")
            self._update_status()

        self._messages.append({"role": "user", "content": user_input})

        guard = RuntimeGuard(RuntimeConfig(
            max_steps=settings.max_steps,
            timeout_seconds=settings.timeout_seconds,
        ))
        step = 0

        try:
            while True:
                limit_error = guard.check()
                if limit_error:
                    output.error_msg(limit_error)
                    break

                tools_schema = self._dispatch.to_openai_tools()
                fitted = await self._context_mgr.prepare(self._messages, tools_schema)

                thinking.show(f"Calling model (step {step})")

                llm_start = time.time()
                try:
                    response = await litellm.acompletion(
                        model=self._model,
                        messages=fitted,
                        tools=tools_schema if tools_schema else None,
                        stream=True,
                    )

                    full_content = ""
                    tool_calls_raw: list[dict] = []
                    usage_data = None
                    is_streaming_text = False

                    async for chunk in response:
                        delta = chunk.choices[0].delta if chunk.choices else None
                        if delta is None:
                            continue

                        if delta.content:
                            if not is_streaming_text:
                                thinking.hide()
                                output.stream_start()
                                is_streaming_text = True
                            full_content += delta.content
                            output.stream_token(delta.content)

                        if delta.tool_calls:
                            for tc_delta in delta.tool_calls:
                                idx = tc_delta.index
                                while len(tool_calls_raw) <= idx:
                                    tool_calls_raw.append({"id": "", "function": {"name": "", "arguments": ""}})
                                if tc_delta.id:
                                    tool_calls_raw[idx]["id"] = tc_delta.id
                                if tc_delta.function:
                                    if tc_delta.function.name:
                                        tool_calls_raw[idx]["function"]["name"] = tc_delta.function.name
                                    if tc_delta.function.arguments:
                                        tool_calls_raw[idx]["function"]["arguments"] += tc_delta.function.arguments

                        if hasattr(chunk, "usage") and chunk.usage:
                            usage_data = chunk.usage

                except Exception as e:
                    thinking.hide()
                    output.error_msg(f"LLM: {e}")
                    break

                llm_ms = (time.time() - llm_start) * 1000

                if is_streaming_text:
                    output.stream_end(full_content)

                # Usage
                usage = TokenUsage()
                if usage_data:
                    usage.prompt_tokens = getattr(usage_data, "prompt_tokens", 0) or 0
                    usage.completion_tokens = getattr(usage_data, "completion_tokens", 0) or 0
                    usage.total_tokens = usage.prompt_tokens + usage.completion_tokens

                self._total_usage.prompt_tokens += usage.prompt_tokens
                self._total_usage.completion_tokens += usage.completion_tokens
                self._total_usage.total_tokens += usage.total_tokens
                self._cost_tracker.add_step(step, usage)
                self._update_status()

                budget_error = self._cost_tracker.check_budget()
                if budget_error:
                    thinking.hide()
                    output.error_msg(budget_error)
                    break

                # Tool calls
                if tool_calls_raw and tool_calls_raw[0]["function"]["name"]:
                    thinking.hide()

                    if full_content:
                        output.agent_thinking(full_content)

                    assistant_dict: dict[str, Any] = {"role": "assistant"}
                    if full_content:
                        assistant_dict["content"] = full_content
                    assistant_dict["tool_calls"] = [
                        {"id": tc["id"], "type": "function", "function": tc["function"]}
                        for tc in tool_calls_raw
                    ]
                    self._messages.append(assistant_dict)

                    parsed_calls = self._dispatch.parse_tool_calls(tool_calls_raw)
                    skill_changed = False

                    for tool_call in parsed_calls:
                        thinking.show(f"Running {tool_call.name}")

                        tool_start = time.time()

                        access_denied = self._access.check(tool_call)
                        if access_denied:
                            result_content = f"Access denied: {access_denied}"
                            result_success = False
                            thinking.hide()
                            output.tool_denied(tool_call.name, access_denied)
                        elif (val_err := self._dispatch.validate_call(tool_call)):
                            result_content = self._dispatch.retry_prompt(tool_call, val_err)
                            result_success = False
                        else:
                            result = await self._dispatch.execute(tool_call)
                            result_content = result.content
                            result_success = result.success

                        tool_ms = (time.time() - tool_start) * 1000
                        result_content = self._context_mgr.compact_tool_result(result_content)

                        thinking.hide()
                        output.tool_call(
                            tool_call.name, tool_call.arguments,
                            result_success, tool_ms, result_content[:100],
                        )

                        self._messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.name,
                            "content": result_content,
                        })

                        if tool_call.name in _SKILL_MUTATING_TOOLS:
                            skill_changed = True

                    if skill_changed:
                        new_prompt = build_system_prompt(self._skill_loader)
                        self._messages[0] = {"role": "system", "content": new_prompt}

                    guard.increment_step()
                    step += 1

                else:
                    # No tool calls — done
                    if full_content and not is_streaming_text:
                        output.stream_start()
                        output.stream_token(full_content)
                        output.stream_end(full_content)
                        self._messages.append({"role": "assistant", "content": full_content})
                    elif full_content:
                        self._messages.append({"role": "assistant", "content": full_content})
                    break

        except Exception as e:
            output.error_msg(str(e))

        thinking.hide()
        self._update_status()
        input_widget.disabled = False
        input_widget.focus()

    async def _handle_command(self, command: str) -> None:
        output = self.query_one("#main-output", MainOutput)
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd == "/help":
            output.system_msg(
                "/model <name>      Switch model\n"
                "/strategy <name>   Context strategy (truncate, summary, tiered)\n"
                "/policy <name>     Access policy (unrestricted, readonly, no_shell)\n"
                "/budget <amount>   Set cost budget in USD\n"
                "/skills            List skills\n"
                "/metrics           Show session metrics\n"
                "/clear             Clear output\n"
                "/reset             Reset session\n"
                "/quit              Exit\n"
                "\n"
                "Ctrl+C quit · Ctrl+L clear"
            )
        elif cmd == "/model":
            if not arg:
                output.system_msg(f"Current: {self._model}")
            else:
                self._model = arg
                self._cost_tracker = CostTracker(model=self._model, budget_usd=self._budget)
                output.system_msg(f"Model → {self._model}")
                self._update_status()
        elif cmd == "/strategy":
            if not arg:
                output.system_msg(f"Current: {self._strategy}")
            elif arg in ("truncate", "summary", "tiered"):
                self._strategy = arg
                self._context_mgr = ContextManager(strategy_name=arg, model=self._model)
                output.system_msg(f"Strategy → {arg}")
            else:
                output.error_msg(f"Unknown: {arg}. Use truncate, summary, tiered")
        elif cmd == "/policy":
            if not arg:
                output.system_msg(f"Current: {self._policy_name}")
            else:
                try:
                    self._access = AccessController(policy=get_policy(arg))
                    self._policy_name = arg
                    output.system_msg(f"Policy → {arg}")
                except ValueError as e:
                    output.error_msg(str(e))
        elif cmd == "/budget":
            if not arg:
                if self._budget:
                    output.system_msg(f"Budget: ${self._budget:.4f} (${self._budget - self._cost_tracker.total_cost_usd:.4f} left)")
                else:
                    output.system_msg("No budget set")
            else:
                try:
                    self._budget = float(arg)
                    self._cost_tracker.budget_usd = self._budget
                    output.system_msg(f"Budget → ${self._budget:.4f}")
                except ValueError:
                    output.error_msg(f"Invalid: {arg}")
        elif cmd == "/skills":
            loaded = self._skill_loader.get_loaded_skill_names()
            available = self._skill_loader.get_available_skill_names()
            output.system_msg(
                f"Available: {', '.join(available) or 'none'}\n"
                f"Loaded: {', '.join(loaded) or 'none'}"
            )
        elif cmd == "/metrics":
            output.show_metrics(
                self._model, self._strategy, self._turn_count,
                self._total_usage, self._cost_tracker.total_cost_usd,
                self._budget, self._skill_loader.get_loaded_skill_names(),
                len(self._access.denied_calls),
            )
        elif cmd == "/costs":
            summary = self._cost_tracker.summary()
            lines = "\n".join(f"  {k}: {v}" for k, v in summary.items())
            output.system_msg(f"Cost Summary\n{lines}")
        elif cmd == "/clear":
            self.query_one("#main-output", MainOutput).remove_children()
        elif cmd == "/reset":
            self._init_session()
            self.query_one("#main-output", MainOutput).remove_children()
            output = self.query_one("#main-output", MainOutput)
            output.system_msg("Session reset")
            self._update_status()
        elif cmd in ("/quit", "/exit", "/q"):
            self.exit()
        else:
            output.error_msg(f"Unknown: {cmd}. Type /help")

    def action_clear(self) -> None:
        self.query_one("#main-output", MainOutput).remove_children()

    def action_quit(self) -> None:
        self.exit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent Harness TUI")
    parser.add_argument("--model", "-m", default=settings.default_model)
    parser.add_argument("--strategy", "-s", default=settings.context_strategy)
    parser.add_argument("--policy", "-p", default="unrestricted")
    parser.add_argument("--budget", type=float, default=None)
    args = parser.parse_args()

    app = AgentHarnessApp(
        model=args.model, strategy=args.strategy,
        policy=args.policy, budget=args.budget,
    )
    app.run()


if __name__ == "__main__":
    main()
