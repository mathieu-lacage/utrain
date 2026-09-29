"""How the dispatcher finalizes an attempt once its orchestrator is gone.

The orchestrator records each phase as it goes: `running`, then `done`,
`failed`, or `stopped` when it is asked to stop. It never records how the
attempt or the run ended. That is decided here, by the dispatcher, and only
after the orchestrator's lock is free -- which is proof it has exited, however
it exited, SIGKILL and OOM kills included (see `lock`). One rule covers a clean
finish and a crash alike: phases recorded as terminal keep their status,
anything still `pending` or `running` was interrupted and becomes `stopped`,
and the attempt and run follow from the resulting set.

Nothing else calls this. Reading a run writes nothing; a reader only makes sure
a dispatcher is alive to do it (`dispatcher.ensure`).
"""

import time

import sqlalchemy
import sqlalchemy.orm

from . import db as dbmod
from . import lock, types


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
) -> str | None:
    """Finalize one attempt if its orchestrator is gone; otherwise do nothing.

    Returns the status the attempt was finalized to, or None if it was not.
    """
    run_row = (
        session.execute(sqlalchemy.select(dbmod.runs).where(dbmod.runs.c.id == run_id))
        .mappings()
        .fetchone()
    )
    if run_row is None:
        return None

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
        return None

    if attempt_row["status"] in types.TERMINAL:
        return None

    run_dir = dbmod.run_dir(run_id, session)
    attempt_dir = run_dir / "attempt" / str(attempt)
    pid = attempt_row["pid"]
    if lock.is_held(attempt_dir, int(pid) if pid is not None else None):
        return None

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
    ends: list[float] = []
    for phase_row in phase_rows:
        phase = str(phase_row["phase"])
        status = str(phase_row["status"])
        if status in types.TERMINAL:
            # Includes phases served from cache, which the orchestrator marks
            # `done` without ever launching a container.
            statuses[phase] = status
            if phase_row["ended_at"] is not None:
                ends.append(float(phase_row["ended_at"]))
            continue

        statuses[phase] = "stopped"
        if phase_row["started_at"] is not None:
            ends.append(now)
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
    # The attempt ended when its last phase did, not when the dispatcher got
    # to it; an attempt none of whose phases started ends now.
    session.execute(
        sqlalchemy.update(dbmod.run_attempts)
        .where((dbmod.run_attempts.c.run_id == run_id) & (dbmod.run_attempts.c.attempt == attempt))
        .values(status=final_status, ended_at=max(ends) if ends else now)
    )
    # Only a run still `running`: a run restarted while this attempt was
    # stopping is `queued` for its next attempt, and stays so.
    session.execute(
        sqlalchemy.update(dbmod.runs)
        .where((dbmod.runs.c.id == run_id) & (dbmod.runs.c.status == "running"))
        .values(status=final_status)
    )
    return final_status
