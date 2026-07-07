import pathlib
import sqlite3

import pydantic


class PhaseEvent(pydantic.BaseModel):
    phase: str
    event: str  # started | completed | failed
    timestamp: float


class Metric(pydantic.BaseModel):
    rowid: int
    step: int
    timestamp: float
    phase: str
    name: str
    value: float


def _connect(metrics_db: pathlib.Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{metrics_db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def read_phase_events(metrics_db: pathlib.Path) -> list[PhaseEvent]:
    if not metrics_db.exists():
        return []
    try:
        with _connect(metrics_db) as conn:
            rows = conn.execute(
                "SELECT phase, event, timestamp FROM phase_events ORDER BY timestamp"
            ).fetchall()
        return [
            PhaseEvent(phase=r["phase"], event=r["event"], timestamp=r["timestamp"]) for r in rows
        ]
    except sqlite3.OperationalError:
        return []


def read_metrics(
    metrics_db: pathlib.Path,
    phase: str | None = None,
    name: str | None = None,
    since_rowid: int = 0,
) -> list[Metric]:
    if not metrics_db.exists():
        return []
    try:
        with _connect(metrics_db) as conn:
            query = "SELECT rowid, step, timestamp, phase, name, value FROM metrics WHERE rowid > ?"
            params: list[object] = [since_rowid]
            if phase is not None:
                query += " AND phase = ?"
                params.append(phase)
            if name is not None:
                query += " AND name = ?"
                params.append(name)
            query += " ORDER BY rowid"
            rows = conn.execute(query, params).fetchall()
        return [
            Metric(
                rowid=r["rowid"],
                step=r["step"],
                timestamp=r["timestamp"],
                phase=r["phase"],
                name=r["name"],
                value=r["value"],
            )
            for r in rows
        ]
    except sqlite3.OperationalError:
        return []
