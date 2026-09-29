import pathlib

import sqlalchemy
import sqlalchemy.orm

from . import address, container, dispatcher, exceptions, logs, metrics, runs, types
from . import db as dbmod


def list_phases(
    addr: str,
    session: sqlalchemy.orm.Session,
) -> list[types.PhaseListEntry]:
    run_id, attempt_n, _ = address.parse(addr, session)

    if attempt_n is None:
        attempt_n = dbmod.latest_attempt(run_id, session)
    if attempt_n is None:
        raise exceptions.UI("run has no attempts yet")

    dispatcher.ensure(session)

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
    all_phases = dbmod.run_description(run_row, session).phase_order

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

    # The last attempt that actually ran each pre-from_phase phase, and what
    # came of it there. The whole row rather than just the attempt number: a
    # phase this attempt skipped still has a status, and it is the one it
    # finished with -- reporting nothing would say "unknown" about a phase
    # whose output the current attempt is built on.
    def _inherited(phase: str) -> sqlalchemy.RowMapping | None:
        return (
            session.execute(
                sqlalchemy.select(dbmod.run_phases)
                .where(
                    (dbmod.run_phases.c.run_id == run_id)
                    & (dbmod.run_phases.c.phase == phase)
                    & (dbmod.run_phases.c.attempt < attempt_n)
                )
                .order_by(dbmod.run_phases.c.attempt.desc())
                .limit(1)
            )
            .mappings()
            .fetchone()
        )

    rid = dbmod.short_run_id(run_id, session)

    entries: list[types.PhaseListEntry] = []
    for i, phase in enumerate(all_phases):
        if i < from_phase_order:
            # Not re-run by this attempt: report where it last ran, and how it
            # went there.
            previous = _inherited(phase)
            last_att = None if previous is None else int(previous["attempt"])
            entries.append(
                types.PhaseListEntry(
                    phase=phase,
                    phase_order=i,
                    address=f"{rid}/{last_att}/{phase}" if last_att is not None else phase,
                    status=None if previous is None else str(previous["status"]),
                    started_at=None if previous is None else previous["started_at"],
                    ended_at=None if previous is None else previous["ended_at"],
                    inherited_from=last_att,
                )
            )
        elif phase in phase_rows_in_db:
            p = phase_rows_in_db[phase]
            entries.append(
                types.PhaseListEntry(
                    phase=phase,
                    phase_order=int(p["phase_order"]),
                    address=f"{rid}/{attempt_n}/{phase}",
                    status=str(p["status"]),
                    started_at=p["started_at"],
                    ended_at=p["ended_at"],
                    inherited_from=None,
                )
            )

    return entries


def restart_phase(
    addr: str,
    session: sqlalchemy.orm.Session,
) -> str:
    """Restart a run from the phase a phase address names, as a new attempt.

    The address is the same one `phase show` takes. The attempt in it, if any,
    says where the caller saw the phase; the restart always branches off the
    run's latest state, exactly what `run restart RUN/PHASE` does -- this is
    that same operation spelled from the phase's side, so a failed run can be
    sent back to the phase that broke by pointing at the phase itself.
    """
    run_id, _, phase = address.parse(addr, session)
    if phase is None:
        raise exceptions.UI("phase address must include a phase name")
    return runs.restart_run(run_id, phase, session)


def list_phase_ids(addr: str, session: sqlalchemy.orm.Session) -> list[str]:
    run_id, attempt_n, _ = address.parse(addr, session)

    if attempt_n is None:
        attempt_n = dbmod.latest_attempt(run_id, session)
    if attempt_n is None:
        raise exceptions.UI("run has no attempts yet")

    run_row = dbmod.get_run(run_id, session)
    described = dbmod.run_description(run_row, session)
    return [f"{run_id}/{attempt_n}/{phase}" for phase in described.phase_order]


def read_metric(
    addr: str,
    session: sqlalchemy.orm.Session,
    metric: str,
    since_step: int = 0,
) -> list[types.Metric]:
    """The full recorded series for one metric of a phase."""
    _, _, phase, attempt_dir = _resolve_phase(addr, session)
    series = container.run_data.read_metrics(attempt_dir, phase, name=metric)
    if since_step:
        series = [m for m in series if m.step >= since_step]
    return [types.Metric(name=m.name, step=m.step, value=m.value) for m in series]


def metrics_path(addr: str, session: sqlalchemy.orm.Session) -> pathlib.Path | None:
    """The metrics file a phase logs to, or None before it has written one.

    A caller following a live phase wants to hold one `metrics.Tail` open over
    this rather than re-read the series each refresh, so it gets the path rather
    than the points.
    """
    _, _, phase, attempt_dir = _resolve_phase(addr, session)
    return metrics.find_rtsdb(attempt_dir, phase)


def _resolve_phase(
    addr: str,
    session: sqlalchemy.orm.Session,
) -> tuple[str, int, str, pathlib.Path]:
    """Resolve a phase address to (run_id, attempt, phase, attempt_dir)."""
    run_id, attempt_n, phase = address.parse(addr, session)

    if phase is None:
        raise exceptions.UI("phase address must include a phase name")

    if attempt_n is None:
        attempt_n = dbmod.latest_attempt(run_id, session)
    if attempt_n is None:
        raise exceptions.UI("run has no attempts yet")

    dispatcher.ensure(session)

    run_dir = dbmod.run_dir(run_id, session)
    return run_id, attempt_n, phase, run_dir / "attempt" / str(attempt_n)


def show_phase(
    addr: str,
    session: sqlalchemy.orm.Session,
) -> types.PhaseDetail:
    run_id, attempt_n, phase, attempt_dir = _resolve_phase(addr, session)
    run_row = dbmod.get_run(run_id, session)

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
        raise exceptions.UI(f"phase '{phase}' not found in attempt {attempt_n}")

    described = dbmod.run_description(run_row, session)
    phase_label = phase
    for pi in described.phases:
        if pi.name == phase:
            phase_label = f"{phase} ({pi.label})"
            break

    # Last value seen for each metric name.
    last_by_name: dict[str, container.run_data.Metric] = {}
    for m in container.run_data.read_metrics(attempt_dir, phase):
        last_by_name[m.name] = m

    log_file = logs.phase_log_path(attempt_dir, phase)
    return types.PhaseDetail(
        run_id=run_id,
        run_name=str(run_row["name"]),
        attempt=attempt_n,
        phase=phase,
        phase_label=phase_label,
        status=str(phase_row["status"]),
        started_at=phase_row["started_at"],
        ended_at=phase_row["ended_at"],
        last_metrics=[
            types.Metric(name=name, step=m.step, value=m.value)
            for name, m in sorted(last_by_name.items())
        ],
        log_file=log_file if log_file.exists() else None,
    )


def read_log_tail(detail: types.PhaseDetail, n: int) -> list[str]:
    """Tail of a phase's output log, or nothing when it has not written one."""
    if detail.log_file is None:
        return []
    return logs.tail_lines(detail.log_file, n)
