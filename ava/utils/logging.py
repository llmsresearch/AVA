from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

from rich.console import Console
from rich.table import Table


console = Console()


@dataclass
class LogRow:
    event: str
    budget_tokens: int
    used_tokens: int
    tool_calls: int
    verify_calls: int
    accuracy: Optional[float] = None
    info: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def print_table(rows: list[LogRow]) -> None:
    table = Table(show_header=True, header_style="bold magenta")
    for col in [
        "event",
        "budget_tokens",
        "used_tokens",
        "tool_calls",
        "verify_calls",
        "accuracy",
        "info",
    ]:
        table.add_column(col)
    for r in rows:
        d = r.as_dict()
        table.add_row(
            str(d["event"]),
            str(d["budget_tokens"]),
            str(d["used_tokens"]),
            str(d["tool_calls"]),
            str(d["verify_calls"]),
            "" if d["accuracy"] is None else f"{d['accuracy']:.3f}",
            d.get("info") or "",
        )
    console.print(table)



