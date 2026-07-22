"""Minimal plain-text table rendering (no third-party dependencies)."""

from __future__ import annotations

from collections.abc import Sequence


def format_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Render ``rows`` as an aligned monospace table with a header rule.

    Values are stringified with ``str``; ``None`` renders as ``-``.
    """
    str_rows = [["-" if v is None else str(v) for v in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def line(cells: Sequence[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)).rstrip()

    rule = "  ".join("-" * w for w in widths)
    return "\n".join([line(list(headers)), rule, *(line(row) for row in str_rows)])


def human_size(n_bytes: int) -> str:
    """Format a byte count as a short human-readable string (e.g. ``1.4 MB``)."""
    size = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")
