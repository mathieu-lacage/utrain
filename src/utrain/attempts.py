import sqlalchemy
import sqlalchemy.orm

from . import address, dispatcher, exceptions, runs, types
from . import db as dbmod


def list_attempts(
    addr: str,
    session: sqlalchemy.orm.Session,
) -> list[types.AttemptRow]:
    a = address.parse(addr, session)
    run_id = a.run_id

    dispatcher.ensure(session)

    query = (
        sqlalchemy.select(dbmod.run_attempts)
        .where(dbmod.run_attempts.c.run_id == run_id)
        .order_by(dbmod.run_attempts.c.attempt.desc())
    )
    if a.attempt is not None:
        query = query.where(dbmod.run_attempts.c.attempt == a.attempt)
    rows = session.execute(query).mappings().fetchall()

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


def list_attempt_ids(addr: str, session: sqlalchemy.orm.Session) -> list[str]:
    a = address.parse(addr, session)
    run_id = a.run_id

    query = (
        sqlalchemy.select(dbmod.run_attempts.c.attempt)
        .where(dbmod.run_attempts.c.run_id == run_id)
        .order_by(dbmod.run_attempts.c.attempt.desc())
    )
    if a.attempt is not None:
        query = query.where(dbmod.run_attempts.c.attempt == a.attempt)
    rows = session.execute(query).scalars().fetchall()
    return [f"{run_id}/{attempt}" for attempt in rows]


def show_attempt(
    run_id_prefix: str,
    attempt_n: int,
    session: sqlalchemy.orm.Session,
) -> types.AttemptDetail:
    run_id = dbmod.resolve_run_id(run_id_prefix, session)
    dispatcher.ensure(session)

    run_row = dbmod.get_run(run_id, session)
    run_dir = dbmod.run_dir(run_id, session)

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
