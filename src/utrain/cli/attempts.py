import pathlib

import sqlalchemy
import sqlalchemy.orm

from . import db as dbmod
from . import exceptions, output, reconcile


def list_attempts(run_id_prefix: str, session: sqlalchemy.orm.Session) -> None:
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

    headers = ["ATTEMPT", "FROM_PHASE", "STATUS", "STARTED", "ENDED"]
    table_rows = [
        [
            f"{rid}/{r['attempt']}",
            str(r["from_phase"]) if r["from_phase"] else "--",
            str(r["status"]),
            output.format_time(r["started_at"]),
            output.format_time(r["ended_at"]),
        ]
        for r in rows
    ]
    print(output.format_table(headers, table_rows))


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


def show_attempt(run_id_prefix: str, attempt_n: int, session: sqlalchemy.orm.Session) -> None:
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
        raise exceptions.UI(f"abort: attempt {attempt_n} not found for run '{run_id}'")

    phase_rows = (
        session.execute(
            sqlalchemy.select(dbmod.run_phases)
            .where(
                (dbmod.run_phases.c.run_id == run_id) & (dbmod.run_phases.c.attempt == attempt_n)
            )
            .order_by(dbmod.run_phases.c.phase_order)
        )
        .mappings()
        .fetchall()
    )

    print(f"run:        {run_id} ({run_row['name']})")
    print(f"attempt:    {attempt_n}")
    print(f"from_phase: {attempt_row['from_phase'] or '--'}")
    print(f"status:     {attempt_row['status']}")
    print(f"started:    {output.format_time(attempt_row['started_at'])}")
    print(f"ended:      {output.format_time(attempt_row['ended_at'])}")

    if phase_rows:
        print()
        headers = ["PHASE", "ORDER", "STATUS", "STARTED", "ENDED"]
        table_rows = [
            [
                str(p["phase"]),
                str(p["phase_order"]),
                str(p["status"]),
                output.format_time(p["started_at"]),
                output.format_time(p["ended_at"]),
            ]
            for p in phase_rows
        ]
        print(output.format_table(headers, table_rows))

    logs_dir = run_dir / "attempt" / str(attempt_n) / "logs"
    wandb_dir = run_dir / "attempt" / str(attempt_n) / "wandb"
    print()
    print(f"logs: {logs_dir}")
    print(f"data: {wandb_dir}")
