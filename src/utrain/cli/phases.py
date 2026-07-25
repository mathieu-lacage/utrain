import pathlib

import sqlalchemy
import sqlalchemy.orm

from .. import container
from . import db as dbmod
from . import exceptions, output, reconcile


def _parse_phase_addr(
    addr: str,
    session: sqlalchemy.orm.Session,
) -> tuple[str, int | None, str | None]:
    """Parse <run_id_prefix>[/<attempt>][/<phase>] into (run_id, attempt, phase)."""
    parts = addr.split("/")
    run_id_prefix = parts[0]
    run_id = dbmod.resolve_run_id(run_id_prefix, session)

    attempt: int | None = None
    phase: str | None = None

    if len(parts) == 1:
        pass
    elif len(parts) == 2:
        token = parts[1]
        if token.isdigit():
            attempt = int(token)
        else:
            phase = token
    elif len(parts) == 3:
        token1 = parts[1]
        if not token1.isdigit():
            raise exceptions.UI(f"abort: expected attempt number, got '{token1}'")
        attempt = int(token1)
        phase = parts[2]
    else:
        raise exceptions.UI(f"abort: invalid phase address '{addr}'")

    return run_id, attempt, phase


def list_phases(addr: str, session: sqlalchemy.orm.Session) -> None:
    run_id, attempt_n, _ = _parse_phase_addr(addr, session)

    if attempt_n is None:
        attempt_n = dbmod.latest_attempt(run_id, session)
    if attempt_n is None:
        raise exceptions.UI("abort: run has no attempts yet")

    reconcile.reconcile_attempt(run_id, attempt_n, session)

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
    from_phase = attempt_row["from_phase"] if attempt_row else None

    # Get all phases from the image to show pre-from_phase entries
    run_row = dbmod.get_run(run_id, session)
    image_key = str(run_row["image"])
    presets = container.enroot.list_presets()
    describe = container.enroot.describe(presets[image_key])
    all_phases = describe.phase_order

    from_phase_order = 0
    if from_phase and from_phase in all_phases:
        from_phase_order = all_phases.index(from_phase)

    phase_rows_in_db = {
        str(r["phase"]): r
        for r in (
            session.execute(
                sqlalchemy.select(dbmod.run_phases)
                .where(
                    (dbmod.run_phases.c.run_id == run_id)
                    & (dbmod.run_phases.c.attempt == attempt_n)
                )
                .order_by(dbmod.run_phases.c.phase_order)
            )
            .mappings()
            .fetchall()
        )
    }

    # Find the last attempt where each pre-from_phase phase ran
    def _last_attempt_for_phase(phase: str) -> int | None:
        result = session.execute(
            sqlalchemy.select(sqlalchemy.func.max(dbmod.run_phases.c.attempt)).where(
                (dbmod.run_phases.c.run_id == run_id)
                & (dbmod.run_phases.c.phase == phase)
                & (dbmod.run_phases.c.attempt < attempt_n)
            )
        ).scalar_one_or_none()
        return int(result) if result is not None else None

    rid = dbmod.short_run_id(run_id, session)

    headers = ["PHASE", "ORDER", "STATUS", "STARTED", "ENDED"]
    table_rows: list[list[str]] = []

    for i, phase in enumerate(all_phases):
        if i < from_phase_order:
            last_att = _last_attempt_for_phase(phase)
            att_label = f"-- (att.{last_att})" if last_att is not None else "--"
            phase_id = f"{rid}/{last_att}/{phase}" if last_att is not None else phase
            table_rows.append([phase_id, str(i), att_label, "--", "--"])
        else:
            if phase in phase_rows_in_db:
                p = phase_rows_in_db[phase]
                table_rows.append(
                    [
                        f"{rid}/{attempt_n}/{phase}",
                        str(p["phase_order"]),
                        str(p["status"]),
                        output.format_time(p["started_at"]),
                        output.format_time(p["ended_at"]),
                    ]
                )

    print(output.format_table(headers, table_rows))


def list_phase_ids(addr: str, session: sqlalchemy.orm.Session) -> list[str]:
    run_id, attempt_n, _ = _parse_phase_addr(addr, session)

    if attempt_n is None:
        attempt_n = dbmod.latest_attempt(run_id, session)
    if attempt_n is None:
        raise exceptions.UI("abort: run has no attempts yet")

    run_row = dbmod.get_run(run_id, session)
    image_key = str(run_row["image"])
    presets = container.enroot.list_presets()
    describe = container.enroot.describe(presets[image_key])

    return [f"{run_id}/{attempt_n}/{phase}" for phase in describe.phase_order]


def read_phase(
    addr: str,
    session: sqlalchemy.orm.Session,
    metric: str | None = None,
    since_step: int = 0,
) -> None:
    run_id, attempt_n, phase = _parse_phase_addr(addr, session)

    if phase is None:
        raise exceptions.UI("abort: phase address must include a phase name")

    if attempt_n is None:
        attempt_n = dbmod.latest_attempt(run_id, session)
    if attempt_n is None:
        raise exceptions.UI("abort: run has no attempts yet")

    reconcile.reconcile_attempt(run_id, attempt_n, session)

    run_row = dbmod.get_run(run_id, session)
    run_dir = pathlib.Path(str(run_row["run_dir"]))
    attempt_dir = run_dir / "attempt" / str(attempt_n)

    phase_row = (
        session.execute(
            sqlalchemy.select(dbmod.run_phases).where(
                (dbmod.run_phases.c.run_id == run_id)
                & (dbmod.run_phases.c.attempt == attempt_n)
                & (dbmod.run_phases.c.phase == phase)
            )
        )
        .mappings()
        .fetchone()
    )
    if phase_row is None:
        raise exceptions.UI(f"abort: phase '{phase}' not found in attempt {attempt_n}")

    # Resolve phase label from image describe
    image_key = str(run_row["image"])
    presets = container.enroot.list_presets()
    describe = container.enroot.describe(presets[image_key])
    phase_label = phase
    for pi in describe.phases:
        if pi.name == phase:
            phase_label = f"{phase} ({pi.label})"
            break

    if metric is not None:
        # Print full series for one metric
        metrics = container.run_data.read_metrics(attempt_dir, phase, name=metric)
        if since_step:
            metrics = [m for m in metrics if m.step >= since_step]
        headers = ["STEP", "VALUE"]
        rows = [[str(m.step), str(m.value)] for m in metrics]
        print(output.format_table(headers, rows))
        return

    print(f"phase:   {phase_label}")
    print(f"attempt: {attempt_n}")
    print(f"status:  {phase_row['status']}")
    print(f"run:     {run_id} ({run_row['name']})")
    print(f"started: {output.format_time(phase_row['started_at'])}")
    print(f"ended:   {output.format_time(phase_row['ended_at'])}")

    all_metrics = container.run_data.read_metrics(attempt_dir, phase)
    if all_metrics:
        # Last value per metric name
        last_by_name: dict[str, container.run_data.Metric] = {}
        for m in all_metrics:
            last_by_name[m.name] = m
        print()
        headers = ["METRIC", "LAST", "STEP"]
        rows = [
            [name, str(round(m.value, 3)), str(m.step)] for name, m in sorted(last_by_name.items())
        ]
        print(output.format_table(headers, rows))

    log_file = attempt_dir / "logs" / f"{phase}_stdout.log"
    if log_file.exists():
        lines = log_file.read_text(errors="replace").splitlines()
        print()
        print(f"--- logs/{phase}_stdout.log (tail 20) ---")
        for line in lines[-20:]:
            print(line)
