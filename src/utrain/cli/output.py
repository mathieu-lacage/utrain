import datetime


def format_time(ts: float | None) -> str:
    if ts is None:
        return "--"
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def format_time_long(ts: float | None) -> str:
    if ts is None:
        return "--"
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def human_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes //= 1024
    return f"{n_bytes:.1f} PB"


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    all_rows: list[list[str]] = [headers] + rows
    widths = [max(len(r[i]) for r in all_rows) for i in range(len(headers))]
    lines: list[str] = []
    for row in all_rows:
        parts: list[str] = []
        for i, cell in enumerate(row):
            w = widths[i]
            if i == len(headers) - 1:
                parts.append(cell)
            else:
                parts.append(cell.ljust(w))
        lines.append("  ".join(parts))
    return "\n".join(lines)
