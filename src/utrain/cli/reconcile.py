import os
import pathlib
import time

import sqlalchemy
import sqlalchemy.orm

from .. import container
from . import db as dbmod


def _phase_status_from_events(
    events: list[container.run_data.PhaseEvent],
    finalize: bool,
) -> str:
    if not events:
        return "stopped" if finalize else "pending"
    last = events[-1]
    if last.event == "completed":
        return "done"
    if last.event == "failed":
        return "failed"
    return "running"


def _phase_boundaries(
    events: list[container.run_data.PhaseEvent],
) -> tuple[float | None, float | None]:
    started_at: float | None = None
    ended_at: float | None = None
    for e in events:
        if e.event == "started":
            started_at = e.timestamp
        elif e.event in ("completed", "failed"):
            ended_at = e.timestamp
    return started_at, ended_at


def _compute_run_status(statuses: dict[str, str]) -> str:
    if not statuses:
        return "failed"
    if all(s == "done" for s in statuses.values()):
        return "done"
    if any(s == "failed" for s in statuses.values()):
        return "failed"
    return "stopped"


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def reconcile_attempt(
    run_id: str,
    attempt: int,
    session: sqlalchemy.orm.Session,
) -> None:
    """Reconcile one attempt's phase statuses from rtsdb events on disk."""
    run_row = (
        session.execute(sqlalchemy.select(dbmod.runs).where(dbmod.runs.c.id == run_id))
        .mappings()
        .fetchone()
    )
    if run_row is None:
        return

    attempt_row = (
        session.execute(
            sqlalchemy.select(dbmod.run_attempts).where(
                (dbmod.run_attempts.c.run_id == run_id) & (dbmod.run_attempts.c.attempt == attempt)
            )
        )
        .mappings()
        .fetchone()
    )
    if attempt_row is None:
        return

    # Skip reconciliation if attempt is already terminal
    if attempt_row["status"] in ("done", "failed", "stopped"):
        return

    run_dir = pathlib.Path(str(run_row["run_dir"]))
    attempt_dir = run_dir / "attempt" / str(attempt)

    phase_rows = (
        session.execute(
            sqlalchemy.select(dbmod.run_phases)
            .where((dbmod.run_phases.c.run_id == run_id) & (dbmod.run_phases.c.attempt == attempt))
            .order_by(dbmod.run_phases.c.phase_order)
        )
        .mappings()
        .fetchall()
    )

    pid = attempt_row["pid"]
    alive = pid is not None and _is_pid_alive(int(pid))
    finalize = not alive

    statuses: dict[str, str] = {}
    for phase_row in phase_rows:
        phase = str(phase_row["phase"])
        events = container.run_data.read_phase_events(attempt_dir, phase)
        if not events and not finalize:
            statuses[phase] = str(phase_row["status"])
            continue
        status = _phase_status_from_events(events, finalize)
        started_at, ended_at = _phase_boundaries(events)
        statuses[phase] = status
        session.execute(
            sqlalchemy.update(dbmod.run_phases)
            .where(
                (dbmod.run_phases.c.run_id == run_id)
                & (dbmod.run_phases.c.attempt == attempt)
                & (dbmod.run_phases.c.phase == phase)
            )
            .values(status=status, started_at=started_at, ended_at=ended_at)
        )

    if finalize:
        all_terminal = all(s in ("done", "failed", "stopped") for s in statuses.values())
        if all_terminal:
            final_status = _compute_run_status(statuses)
            now = time.time()
            session.execute(
                sqlalchemy.update(dbmod.run_attempts)
                .where(
                    (dbmod.run_attempts.c.run_id == run_id)
                    & (dbmod.run_attempts.c.attempt == attempt)
                )
                .values(
                    status=final_status,
                    ended_at=attempt_row["ended_at"] if attempt_row["ended_at"] else now,
                )
            )
            session.execute(
                sqlalchemy.update(dbmod.runs)
                .where(dbmod.runs.c.id == run_id)
                .values(status=final_status)
            )
