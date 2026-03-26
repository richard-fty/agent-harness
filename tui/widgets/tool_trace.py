"""Tool trace widget — real-time display of tool execution."""

from __future__ import annotations

from rich.text import Text
from textual.widgets import RichLog


class ToolTrace(RichLog):
    """Displays tool calls in real time with status indicators."""

    def __init__(self, **kwargs) -> None:
        super().__init__(
            highlight=True,
            markup=True,
            wrap=True,
            **kwargs,
        )
        self._step = 0

    def set_step(self, step: int) -> None:
        self._step = step

    def add_llm_call(self, step: int, msg_count: int, tool_count: int) -> None:
        """Show LLM call starting."""
        self.write(Text.from_markup(
            f"\n[dim]─── Step {step} ───[/dim]"
        ))
        self.write(Text.from_markup(
            f"[dim]LLM call ({msg_count} msgs, {tool_count} tools)...[/dim]"
        ))

    def add_llm_response(self, duration_ms: float, prompt_tokens: int, completion_tokens: int, cost: float) -> None:
        """Show LLM response stats."""
        self.write(Text.from_markup(
            f"[dim]  {duration_ms:.0f}ms | "
            f"in:{prompt_tokens} out:{completion_tokens} | "
            f"${cost:.4f}[/dim]"
        ))

    def add_tool_start(self, name: str, args: dict) -> None:
        """Show tool call starting."""
        args_short = ", ".join(f"{k}={repr(v)[:30]}" for k, v in args.items())
        self.write(Text.from_markup(
            f"[yellow]● {name}[/yellow]({args_short})"
        ))

    def add_tool_end(self, name: str, success: bool, duration_ms: float, preview: str) -> None:
        """Show tool call result."""
        icon = "[green]✓[/green]" if success else "[red]✗[/red]"
        preview_clean = preview[:60].replace("\n", " ")
        self.write(Text.from_markup(
            f"  {icon} {duration_ms:.0f}ms — {preview_clean}"
        ))

    def add_access_denied(self, name: str, reason: str) -> None:
        """Show access denied."""
        self.write(Text.from_markup(
            f"  [red]⊘ {name}[/red]: {reason}"
        ))

    def add_skill_loaded(self, name: str) -> None:
        """Show skill loaded."""
        self.write(Text.from_markup(
            f"  [cyan]↳ Skill loaded:[/cyan] {name}"
        ))
