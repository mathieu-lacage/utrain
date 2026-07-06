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

from .. import ctx, models, services

router = fastapi.APIRouter(prefix="/api/projects/{project_id}/runs", tags=["runs"])


class RunResponse(pydantic.BaseModel):
    id: str
    project_id: str
    run_dir: str
    status: str
    pid: int | None
    started_at: float | None
    ended_at: float | None
    phase_events: list[services.metrics_reader.PhaseEvent] = []


_RUN_COLS = [
    models.runs.c.id,
    models.runs.c.project_id,
    models.runs.c.run_dir,
    models.runs.c.status,
    models.runs.c.pid,
    models.runs.c.started_at,
    models.runs.c.ended_at,
]


def _mapping_to_response(
    row: sqlalchemy.engine.RowMapping,
    phase_events: list[services.metrics_reader.PhaseEvent],
) -> RunResponse:
    pid = row["pid"]
    started_at = row["started_at"]
    ended_at = row["ended_at"]
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
    image = settings.presets.get(preset_name)
    if image is None:
        raise fastapi.HTTPException(
            status_code=422, detail=f"Preset image '{preset_name}' not configured"
        )

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

    proc = services.enroot.start_run(image, run_dir)
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

    services.runner.register_run(run_id, proc)
    asyncio.create_task(services.runner.monitor_run(run_id, run_dir, request.app.state.engine))

    return RunResponse(
        id=run_id,
        project_id=project_id,
        run_dir=str(run_dir),
        status="running",
        pid=proc.pid,
        started_at=now,
        ended_at=None,
    )


@router.get("")
def list_runs(project_id: str) -> list[RunResponse]:
    rows = (
        ctx.db.get()
        .execute(
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
        events = services.metrics_reader.read_phase_events(run_dir / "metrics.db")
        result.append(_mapping_to_response(row, events))
    return result


@router.get("/{run_id}")
def get_run(project_id: str, run_id: str) -> RunResponse:
    row = _get_run(ctx.db.get(), project_id, run_id)
    run_dir = pathlib.Path(str(row["run_dir"]))
    events = services.metrics_reader.read_phase_events(run_dir / "metrics.db")
    return _mapping_to_response(row, events)


@router.post("/{run_id}/stop", status_code=204)
def stop_run(project_id: str, run_id: str) -> None:
    row = _get_run(ctx.db.get(), project_id, run_id)
    run_dir = pathlib.Path(str(row["run_dir"]))
    services.runner.stop_run(run_id, run_dir)
    ctx.db.get().execute(
        sqlalchemy.update(models.runs)
        .where(models.runs.c.id == run_id)
        .values(status="stopped", ended_at=time.time())
    )


@router.get("/{run_id}/metrics")
def get_metrics(
    project_id: str,
    run_id: str,
    phase: str | None = None,
    name: str | None = None,
    since_step: int = 0,
) -> list[services.metrics_reader.Metric]:
    row = _get_run(ctx.db.get(), project_id, run_id)
    run_dir = pathlib.Path(str(row["run_dir"]))
    return services.metrics_reader.read_metrics(
        run_dir / "metrics.db", phase=phase, name=name, since_step=since_step
    )


@router.get("/{run_id}/logs")
def get_logs(
    project_id: str,
    run_id: str,
    stderr: bool = False,
    tail: int = 200,
) -> dict[str, str]:
    row = _get_run(ctx.db.get(), project_id, run_id)
    run_dir = pathlib.Path(str(row["run_dir"]))
    log_file = run_dir / "logs" / ("stderr.log" if stderr else "stdout.log")
    if not log_file.exists():
        return {"content": ""}
    lines = log_file.read_text(errors="replace").splitlines()
    return {"content": "\n".join(lines[-tail:])}


class ServeResponse(pydantic.BaseModel):
    port: int


@router.post("/{run_id}/serve", status_code=201)
def start_serve(project_id: str, run_id: str) -> ServeResponse:
    db = ctx.db.get()
    settings = ctx.settings.get()
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

    image = settings.presets.get(str(project_row["preset_name"]))
    if image is None:
        raise fastapi.HTTPException(status_code=422, detail="Preset image not configured")

    port = _find_free_port()
    proc = services.enroot.start_serve(image, run_dir, port)
    db.execute(
        sqlalchemy.insert(models.serve_processes).values(
            run_id=run_id, port=port, pid=proc.pid, started_at=time.time()
        )
    )
    services.runner.register_serve(run_id, proc)
    return ServeResponse(port=port)


@router.delete("/{run_id}/serve", status_code=204)
def stop_serve_endpoint(project_id: str, run_id: str) -> None:
    _get_run(ctx.db.get(), project_id, run_id)
    services.runner.stop_serve(run_id)
    ctx.db.get().execute(
        sqlalchemy.delete(models.serve_processes).where(models.serve_processes.c.run_id == run_id)
    )
