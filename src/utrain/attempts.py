import pathlib

import sqlalchemy
import sqlalchemy.orm

from . import db as dbmod
from . import exceptions, reconcile, runs, types


def list_attempts(
    run_id_prefix: str,
    session: sqlalchemy.orm.Session,
) -> list[types.AttemptRow]:
    run_id = dbmod.resolve_run_id(run_id_prefix, session)

    attempt_n = dbmod.latest_attempt(run_id, session)
    if attempt_n is not None:
        reconcile.reconcile_attempt(run_id, attempt_n, session)

    rows = (
        session.execute(
            sqlalchemy.select(dbmod.run_attempts)
            .where(dbmod.run_attempts.c.run_id == run_id)
            .order_by(dbmod.run_attempts.c.attempt.desc())
        )
        .mappings()
        .fetchall()
    )

    rid = dbmod.short_run_id(run_id, session)
    return [_attempt_row(run_id, rid, r) for r in rows]


def _attempt_row(
    run_id: str,
    short_id: str,
    row: sqlalchemy.engine.RowMapping,
) -> types.AttemptRow:
    return types.AttemptRow(
        run_id=run_id,
        attempt=int(row["attempt"]),
        address=f"{short_id}/{row['attempt']}",
        from_phase=str(row["from_phase"]) if row["from_phase"] else None,
        status=str(row["status"]),
        started_at=row["started_at"],
        ended_at=row["ended_at"],
    )


def list_attempt_ids(run_id_prefix: str, session: sqlalchemy.orm.Session) -> list[str]:
    run_id = dbmod.resolve_run_id(run_id_prefix, session)

    rows = (
        session.execute(
            sqlalchemy.select(dbmod.run_attempts.c.attempt)
            .where(dbmod.run_attempts.c.run_id == run_id)
            .order_by(dbmod.run_attempts.c.attempt.desc())
        )
        .scalars()
        .fetchall()
    )
    return [f"{run_id}/{attempt}" for attempt in rows]


def show_attempt(
    run_id_prefix: str,
    attempt_n: int,
    session: sqlalchemy.orm.Session,
) -> types.AttemptDetail:
    run_id = dbmod.resolve_run_id(run_id_prefix, session)
    reconcile.reconcile_attempt(run_id, attempt_n, session)

    run_row = dbmod.get_run(run_id, session)
    run_dir = pathlib.Path(str(run_row["run_dir"]))

    attempt_row = (
        session.execute(
            sqlalchemy.select(dbmod.run_attempts).where(
                (dbmod.run_attempts.c.run_id == run_id)
                & (dbmod.run_attempts.c.attempt == attempt_n)
            )
        )
        .mappings()
        .fetchone()
    )
    if attempt_row is None:
        raise exceptions.UI(f"attempt {attempt_n} not found for run '{run_id}'")

    attempt_dir = run_dir / "attempt" / str(attempt_n)
    return types.AttemptDetail(
        run_id=run_id,
        run_name=str(run_row["name"]),
        attempt=_attempt_row(run_id, dbmod.short_run_id(run_id, session), attempt_row),
        phases=runs.phase_rows(run_id, attempt_n, session),
        logs_dir=attempt_dir / "logs",
        data_dir=attempt_dir / "wandb",
    )
