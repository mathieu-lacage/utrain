import asyncio
import json
import pathlib
import socket
import time
import uuid

import fastapi
import pydantic
import sqlalchemy
import sqlalchemy.engine
import sqlalchemy.orm
import yaml

from .. import container, ctx, models

router = fastapi.APIRouter(prefix="/api/projects/{project_id}/runs", tags=["runs"])

_RUN_COLS = [
    models.runs.c.id,
    models.runs.c.project_id,
    models.runs.c.run_dir,
    models.runs.c.status,
    models.runs.c.pid,
    models.runs.c.started_at,
    models.runs.c.ended_at,
]


class RunResponse(pydantic.BaseModel):
    id: str
    project_id: str
    run_dir: str
    status: str
    pid: int | None
    started_at: float | None
    ended_at: float | None
    phase_events: list[container.run_data.PhaseEvent] = []


def _phase_order(session: sqlalchemy.orm.Session, run_id: str) -> list[str]:
    rows = (
        session.execute(
            sqlalchemy.select(models.run_phases.c.phase)
            .where(models.run_phases.c.run_id == run_id)
            .order_by(models.run_phases.c.phase_order)
        )
        .scalars()
        .fetchall()
    )
    return [str(r) for r in rows]


def _load_phase_events(
    run_dir: pathlib.Path,
    phase_order: list[str],
) -> list[container.run_data.PhaseEvent]:
    all_events: list[container.run_data.PhaseEvent] = []
    for phase in phase_order:
        all_events.extend(container.run_data.read_phase_events(run_dir, phase))
    all_events.sort(key=lambda e: e.timestamp)
    return all_events


def _mapping_to_response(
    row: sqlalchemy.engine.RowMapping,
    phase_events: list[container.run_data.PhaseEvent],
) -> RunResponse:
    started_at = row["started_at"]
    ended_at = row["ended_at"]
    pid = row["pid"]
    return RunResponse(
        id=str(row["id"]),
        project_id=str(row["project_id"]),
        run_dir=str(row["run_dir"]),
        status=str(row["status"]),
        pid=int(pid) if pid is not None else None,
        started_at=float(started_at) if started_at is not None else None,
        ended_at=float(ended_at) if ended_at is not None else None,
        phase_events=phase_events,
    )


def _get_run(
    db: sqlalchemy.orm.Session,
    project_id: str,
    run_id: str,
) -> sqlalchemy.engine.RowMapping:
    row = (
        db.execute(
            sqlalchemy.select(*_RUN_COLS).where(
                (models.runs.c.id == run_id) & (models.runs.c.project_id == project_id)
            )
        )
        .mappings()
        .fetchone()
    )
    if row is None:
        raise fastapi.HTTPException(status_code=404, detail="Run not found")
    return row


def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


def _phase_status_from_events(events: list[container.run_data.PhaseEvent]) -> str:
    if not events:
        return "pending"
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


def _sync_phase_statuses(
    run_id: str,
    run_dir: pathlib.Path,
    phase_order: list[str],
    engine: sqlalchemy.Engine,
    finalize: bool,
) -> dict[str, str]:
    """Reconcile run_phases rows from the events written by the container.

    While a run is alive (``finalize=False``) only phases that already have
    events are touched, so optimistic "running" markers are not clobbered.
    When finalizing, every phase is written: event-derived where events exist
    and "stopped" for phases the container never reached.
    """
    statuses: dict[str, str] = {}
    with sqlalchemy.orm.Session(engine) as session:
        for phase in phase_order:
            events = container.run_data.read_phase_events(run_dir, phase)
            if not events and not finalize:
                continue
            status = _phase_status_from_events(events) if events else "stopped"
            started_at, ended_at = _phase_boundaries(events)
            statuses[phase] = status
            session.execute(
                sqlalchemy.update(models.run_phases)
                .where(
                    (models.run_phases.c.run_id == run_id) & (models.run_phases.c.phase == phase)
                )
                .values(status=status, started_at=started_at, ended_at=ended_at)
            )
        session.commit()
    return statuses


def _compute_run_status(statuses: dict[str, str]) -> str:
    if not statuses:
        return "failed"
    if all(s == "done" for s in statuses.values()):
        return "done"
    return "failed"


def _set_run_status(
    run_id: str,
    status: str,
    engine: sqlalchemy.Engine,
    ended_at: float | None = None,
) -> None:
    with sqlalchemy.orm.Session(engine) as session:
        session.execute(
            sqlalchemy.update(models.runs)
            .where(models.runs.c.id == run_id)
            .values(status=status, ended_at=ended_at)
        )
        session.commit()


async def _monitor_run(
    run_id: str,
    run_dir: pathlib.Path,
    phase_order: list[str],
    engine: sqlalchemy.Engine,
) -> None:
    """Track a single ``run-all`` container process to completion.

    The container sequences the phases itself and writes progress as events to
    the run dir; this task only mirrors those events into the database so the
    API can report them. If the backend is restarted, ``reconcile_running_runs``
    re-attaches this monitor (or finalizes from the events the container kept
    writing while the backend was down).
    """
    pid = container.runner.get_pid(run_id)
    if pid is None:
        return
    try:
        while True:
            await asyncio.sleep(2)
            # If the run was stopped/finished externally, stop mirroring.
            with sqlalchemy.orm.Session(engine) as session:
                run_status = session.execute(
                    sqlalchemy.select(models.runs.c.status).where(models.runs.c.id == run_id)
                ).scalar_one_or_none()
            if run_status in ("stopped", "done", "failed"):
                container.runner.finish_run(run_id)
                return
            is_alive = container.runner.is_pid_alive(pid)
            if is_alive:
                _sync_phase_statuses(run_id, run_dir, phase_order, engine, finalize=False)
                continue
            statuses = _sync_phase_statuses(run_id, run_dir, phase_order, engine, finalize=True)
            final_status = _compute_run_status(statuses)
            _set_run_status(run_id, final_status, engine, ended_at=time.time())
            container.runner.finish_run(run_id)
            break
    except asyncio.CancelledError:
        pass


def reconcile_running_runs(engine: sqlalchemy.Engine) -> None:
    """Re-attach monitors to runs that were in progress when the backend died.

    Because runs are driven by a single long-lived ``run-all`` container
    process, the phase sequence keeps running (and keeps writing events) even
    while the backend is down. On restart we either resume monitoring a still
    living process, or finalize a run from the events the container already
    wrote.
    """
    with sqlalchemy.orm.Session(engine) as session:
        run_rows = (
            session.execute(
                sqlalchemy.select(
                    models.runs.c.id,
                    models.runs.c.run_dir,
                    models.runs.c.pid,
                ).where(models.runs.c.status == "running")
            )
            .mappings()
            .fetchall()
        )

    for run_row in run_rows:
        run_id = str(run_row["id"])
        run_dir = pathlib.Path(str(run_row["run_dir"]))
        pid = run_row["pid"]

        with sqlalchemy.orm.Session(engine) as session:
            phase_order = _phase_order(session, run_id)

        if not phase_order:
            _set_run_status(run_id, "failed", engine, ended_at=time.time())
            continue

        if pid is not None and container.runner.is_pid_alive(int(pid)):
            container.runner.register_run_pid(run_id, int(pid))
            _sync_phase_statuses(run_id, run_dir, phase_order, engine, finalize=False)
            asyncio.get_running_loop().create_task(
                _monitor_run(run_id, run_dir, phase_order, engine)
            )
        else:
            statuses = _sync_phase_statuses(run_id, run_dir, phase_order, engine, finalize=True)
            final_status = _compute_run_status(statuses)
            _set_run_status(run_id, final_status, engine, ended_at=time.time())


@router.post("", status_code=201)
async def start_run(project_id: str, request: fastapi.Request) -> RunResponse:
    db = ctx.db.get()
    settings = ctx.settings.get()

    project_row = (
        db.execute(
            sqlalchemy.select(
                models.projects.c.preset_name,
                models.projects.c.config,
            ).where(models.projects.c.id == project_id)
        )
        .mappings()
        .fetchone()
    )
    if project_row is None:
        raise fastapi.HTTPException(status_code=404, detail="Project not found")

    preset_name = str(project_row["preset_name"])
    project_config: dict[str, object] = json.loads(str(project_row["config"]))
    image = container.enroot.list_presets().get(preset_name)
    if image is None:
        raise fastapi.HTTPException(
            status_code=422, detail=f"Preset image '{preset_name}' not found in podman"
        )

    description = container.enroot.describe(image)
    phase_order = description.phase_order
    if not phase_order:
        raise fastapi.HTTPException(status_code=422, detail="Preset has no phases")

    run_id = str(uuid.uuid4())
    run_dir = settings.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config_data: dict[str, object] = {
        "run_id": run_id,
        "output_dir": "/run",
        **project_config,
    }
    (run_dir / "config.yaml").write_text(yaml.dump(config_data))
    (run_dir / "control.json").write_text(json.dumps({"action": "continue"}))
    (run_dir / "logs").mkdir(exist_ok=True)

    # A single process drives the whole phase sequence, so it survives a
    # backend crash and completes on its own; the backend only mirrors events.
    proc = container.enroot.start_run_all(image, run_dir)
    now = time.time()

    db.execute(
        sqlalchemy.insert(models.runs).values(
            id=run_id,
            project_id=project_id,
            run_dir=str(run_dir),
            status="running",
            pid=proc.pid,
            started_at=now,
            ended_at=None,
        )
    )

    for i, phase in enumerate(phase_order):
        db.execute(
            sqlalchemy.insert(models.run_phases).values(
                run_id=run_id,
                phase=phase,
                phase_order=i,
                status="running" if i == 0 else "pending",
                pid=None,
                started_at=now if i == 0 else None,
                ended_at=None,
            )
        )

    container.runner.register_run(run_id, proc)
    asyncio.create_task(_monitor_run(run_id, run_dir, phase_order, request.app.state.engine))

    phase_events = _load_phase_events(run_dir, phase_order)
    return RunResponse(
        id=run_id,
        project_id=project_id,
        run_dir=str(run_dir),
        status="running",
        pid=proc.pid,
        started_at=now,
        ended_at=None,
        phase_events=phase_events,
    )


@router.get("")
def list_runs(project_id: str) -> list[RunResponse]:
    db = ctx.db.get()
    rows = (
        db.execute(
            sqlalchemy.select(*_RUN_COLS)
            .where(models.runs.c.project_id == project_id)
            .order_by(models.runs.c.started_at.desc())
        )
        .mappings()
        .fetchall()
    )
    result: list[RunResponse] = []
    for row in rows:
        run_dir = pathlib.Path(str(row["run_dir"]))
        phase_order = _phase_order(db, str(row["id"]))
        phase_events = _load_phase_events(run_dir, phase_order)
        result.append(_mapping_to_response(row, phase_events))
    return result


@router.get("/{run_id}")
def get_run(project_id: str, run_id: str) -> RunResponse:
    db = ctx.db.get()
    row = _get_run(db, project_id, run_id)
    run_dir = pathlib.Path(str(row["run_dir"]))
    phase_order = _phase_order(db, run_id)
    phase_events = _load_phase_events(run_dir, phase_order)
    return _mapping_to_response(row, phase_events)


@router.post("/{run_id}/stop", status_code=204)
def stop_run(project_id: str, run_id: str) -> None:
    db = ctx.db.get()
    row = _get_run(db, project_id, run_id)
    run_dir = pathlib.Path(str(row["run_dir"]))
    container.runner.stop_run(run_id, run_dir)
    now = time.time()
    db.execute(
        sqlalchemy.update(models.run_phases)
        .where(
            (models.run_phases.c.run_id == run_id)
            & (models.run_phases.c.status.in_(["pending", "running"]))
        )
        .values(status="stopped", ended_at=now)
    )
    db.execute(
        sqlalchemy.update(models.runs)
        .where(models.runs.c.id == run_id)
        .values(status="stopped", ended_at=now)
    )


@router.get("/{run_id}/metrics")
def get_metrics(
    project_id: str,
    run_id: str,
    phase: str,
    name: str | None = None,
    since_rowid: int = 0,
) -> list[container.run_data.Metric]:
    row = _get_run(ctx.db.get(), project_id, run_id)
    run_dir = pathlib.Path(str(row["run_dir"]))
    return container.run_data.read_metrics(run_dir, phase=phase, name=name, since_rowid=since_rowid)


@router.get("/{run_id}/logs")
def get_logs(
    project_id: str,
    run_id: str,
    phase: str | None = None,
    stderr: bool = False,
    tail: int = 200,
) -> dict[str, str]:
    row = _get_run(ctx.db.get(), project_id, run_id)
    run_dir = pathlib.Path(str(row["run_dir"]))
    suffix = "stderr" if stderr else "stdout"
    phase_log = run_dir / "logs" / f"{phase}_{suffix}.log" if phase is not None else None
    # run-all writes a single combined log; fall back to it when no per-phase
    # log exists (or none was requested).
    log_file = (
        phase_log
        if (phase_log is not None and phase_log.exists())
        else (run_dir / "logs" / f"run_all_{suffix}.log")
    )
    if not log_file.exists():
        return {"content": ""}
    lines = log_file.read_text(errors="replace").splitlines()
    return {"content": "\n".join(lines[-tail:])}


class ServeResponse(pydantic.BaseModel):
    port: int


@router.post("/{run_id}/serve", status_code=201)
def start_serve(project_id: str, run_id: str) -> ServeResponse:
    db = ctx.db.get()
    row = _get_run(db, project_id, run_id)
    run_dir = pathlib.Path(str(row["run_dir"]))

    project_row = (
        db.execute(
            sqlalchemy.select(models.projects.c.preset_name).where(
                models.projects.c.id == project_id
            )
        )
        .mappings()
        .fetchone()
    )
    if project_row is None:
        raise fastapi.HTTPException(status_code=404, detail="Project not found")

    image = container.enroot.list_presets().get(str(project_row["preset_name"]))
    if image is None:
        raise fastapi.HTTPException(status_code=422, detail="Preset image not found in podman")

    port = _find_free_port()
    proc = container.enroot.start_serve(image, run_dir, port)
    db.execute(
        sqlalchemy.insert(models.serve_processes).values(
            run_id=run_id, port=port, pid=proc.pid, started_at=time.time()
        )
    )
    container.runner.register_serve(run_id, proc)
    return ServeResponse(port=port)


@router.delete("/{run_id}/serve", status_code=204)
def stop_serve_endpoint(project_id: str, run_id: str) -> None:
    _get_run(ctx.db.get(), project_id, run_id)
    container.runner.stop_serve(run_id)
    ctx.db.get().execute(
        sqlalchemy.delete(models.serve_processes).where(models.serve_processes.c.run_id == run_id)
    )
