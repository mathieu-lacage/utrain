"""Repair for an attempt whose orchestrator died without recording it.

Nothing here runs on the happy path. The orchestrator writes every status
transition itself, in-process, at the moment it happens: a phase going
`running`, a phase finishing `done` or `failed`, and the attempt and run going
terminal once the last phase lands. Even a requested stop is its own write --
`runs._stop_attempt` sends SIGTERM and the orchestrator records `stopped` on
its way out.

The one thing it cannot do is record its own death. SIGKILL, an OOM kill or a
reboot leaves the database saying `running` with nobody left to correct it.
That is the whole job of this module, and why it is gated on `lock.is_held`:
while an orchestrator is alive, reading a run writes nothing at all.

Finalizing needs no evidence beyond what the orchestrator already committed.
Phases it recorded as terminal keep their status; anything still `pending` or
`running` was interrupted, so it becomes `stopped`, and the attempt and run
follow from the resulting set.
"""

import pathlib
import time

import sqlalchemy
import sqlalchemy.orm

from . import db as dbmod
from . import lock

_TERMINAL = ("done", "failed", "stopped")


def _compute_run_status(statuses: dict[str, str]) -> str:
    if not statuses:
        return "failed"
    if all(s == "done" for s in statuses.values()):
        return "done"
    if any(s == "failed" for s in statuses.values()):
        return "failed"
    return "stopped"


def reconcile_attempt(
    run_id: str,
    attempt: int,
    session: sqlalchemy.orm.Session,
) -> None:
    """Finalize one attempt if its orchestrator is gone; otherwise do nothing."""
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

    if attempt_row["status"] in _TERMINAL:
        return

    run_dir = pathlib.Path(str(run_row["run_dir"]))
    attempt_dir = run_dir / "attempt" / str(attempt)
    pid = attempt_row["pid"]
    if lock.is_held(attempt_dir, int(pid) if pid is not None else None):
        return

    phase_rows = (
        session.execute(
            sqlalchemy.select(dbmod.run_phases)
            .where((dbmod.run_phases.c.run_id == run_id) & (dbmod.run_phases.c.attempt == attempt))
            .order_by(dbmod.run_phases.c.phase_order)
        )
        .mappings()
        .fetchall()
    )

    now = time.time()
    statuses: dict[str, str] = {}
    for phase_row in phase_rows:
        phase = str(phase_row["phase"])
        status = str(phase_row["status"])
        if status in _TERMINAL:
            # Includes phases served from cache, which the orchestrator marks
            # `done` without ever launching a container.
            statuses[phase] = status
            continue

        statuses[phase] = "stopped"
        session.execute(
            sqlalchemy.update(dbmod.run_phases)
            .where(
                (dbmod.run_phases.c.run_id == run_id)
                & (dbmod.run_phases.c.attempt == attempt)
                & (dbmod.run_phases.c.phase == phase)
            )
            # A phase that never started has no end: leave ended_at null rather
            # than dating it to the moment someone noticed the orchestrator was
            # gone.
            .values(
                status="stopped",
                ended_at=phase_row["ended_at"]
                or (now if phase_row["started_at"] is not None else None),
            )
        )

    final_status = _compute_run_status(statuses)
    session.execute(
        sqlalchemy.update(dbmod.run_attempts)
        .where((dbmod.run_attempts.c.run_id == run_id) & (dbmod.run_attempts.c.attempt == attempt))
        .values(
            status=final_status,
            ended_at=attempt_row["ended_at"] if attempt_row["ended_at"] else now,
        )
    )
    session.execute(
        sqlalchemy.update(dbmod.runs).where(dbmod.runs.c.id == run_id).values(status=final_status)
    )
