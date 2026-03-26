"""Agent output widget — renders streaming markdown responses."""

from __future__ import annotations

from rich.markdown import Markdown
from rich.text import Text
from textual.widgets import RichLog


class AgentOutput(RichLog):
    """Displays agent responses with markdown rendering."""

    def __init__(self, **kwargs) -> None:
        super().__init__(
            highlight=True,
            markup=True,
            wrap=True,
            **kwargs,
        )

    def append_user_message(self, text: str) -> None:
        """Add a user message."""
        self.write(Text.from_markup(f"\n[bold cyan]You:[/bold cyan] {text}"))

    def append_agent_text(self, text: str) -> None:
        """Add agent thinking/reasoning text (non-markdown)."""
        self.write(Text.from_markup(f"[dim]{text}[/dim]"))

    def append_agent_response(self, markdown_text: str) -> None:
        """Add the agent's final response as rendered markdown."""
        self.write(Text(""))  # spacing
        self.write(Text.from_markup("[bold green]Agent:[/bold green]"))
        try:
            self.write(Markdown(markdown_text))
        except Exception:
            self.write(Text(markdown_text))
        self.write(Text(""))  # spacing

    def append_system(self, text: str) -> None:
        """Add a system message (info, warnings)."""
        self.write(Text.from_markup(f"[bold yellow]{text}[/bold yellow]"))

    def append_error(self, text: str) -> None:
        """Add an error message."""
        self.write(Text.from_markup(f"[bold red]Error:[/bold red] {text}"))
