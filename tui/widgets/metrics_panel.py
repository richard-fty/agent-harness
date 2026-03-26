"""Metrics panel widget — live display of session metrics."""

from __future__ import annotations

from textual.widgets import Static
from rich.table import Table
from rich.text import Text


class MetricsPanel(Static):
    """Live metrics display: tokens, cost, steps, skills."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._model = ""
        self._strategy = ""
        self._turns = 0
        self._steps = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._cost = 0.0
        self._budget = None
        self._skills: list[str] = []
        self._denied = 0

    def set_model(self, model: str, strategy: str) -> None:
        self._model = model
        self._strategy = strategy
        self._refresh_display()

    def set_budget(self, budget: float | None) -> None:
        self._budget = budget
        self._refresh_display()

    def update_metrics(
        self,
        turns: int = 0,
        steps: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost: float = 0.0,
        skills: list[str] | None = None,
        denied: int = 0,
    ) -> None:
        self._turns = turns
        self._steps = steps
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens
        self._cost = cost
        if skills is not None:
            self._skills = skills
        self._denied = denied
        self._refresh_display()

    def _refresh_display(self) -> None:
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Label", style="dim", width=10)
        table.add_column("Value", style="bold")

        model_short = self._model.split("/")[-1] if "/" in self._model else self._model
        table.add_row("Model", model_short)
        table.add_row("Strategy", self._strategy)
        table.add_row("Turns", str(self._turns))
        table.add_row("Steps", str(self._steps))
        table.add_row("Tokens In", f"{self._prompt_tokens:,}")
        table.add_row("Tokens Out", f"{self._completion_tokens:,}")
        table.add_row("Cost", f"${self._cost:.4f}")

        if self._budget is not None:
            remaining = self._budget - self._cost
            color = "green" if remaining > 0 else "red"
            table.add_row("Budget", f"[{color}]${remaining:.4f} left[/{color}]")

        skills_str = ", ".join(self._skills) if self._skills else "none"
        table.add_row("Skills", skills_str)

        if self._denied > 0:
            table.add_row("Denied", f"[red]{self._denied}[/red]")

        self.update(table)
