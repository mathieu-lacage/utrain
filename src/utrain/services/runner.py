import asyncio
import json
import pathlib
import subprocess
import time

import sqlalchemy
import sqlalchemy.orm

from .. import models
from . import metrics_reader

_active: dict[str, subprocess.Popen[bytes]] = {}
_serve_active: dict[str, subprocess.Popen[bytes]] = {}


def register_run(run_id: str, proc: subprocess.Popen[bytes]) -> None:
    _active[run_id] = proc


def register_serve(run_id: str, proc: subprocess.Popen[bytes]) -> None:
    _serve_active[run_id] = proc


def write_control(run_dir: pathlib.Path, action: str) -> None:
    control_path = run_dir / "control.json"
    control_path.write_text(json.dumps({"action": action}))


def stop_run(run_id: str, run_dir: pathlib.Path) -> None:
    write_control(run_dir, "stop")
    proc = _active.get(run_id)
    if proc is not None:
        try:
            proc.terminate()
        except ProcessLookupError:
            pass


def stop_serve(run_id: str) -> None:
    proc = _serve_active.get(run_id)
    if proc is not None:
        try:
            proc.terminate()
        except ProcessLookupError:
            pass
        del _serve_active[run_id]


def _infer_status(events: list[metrics_reader.PhaseEvent], returncode: int | None) -> str:
    if returncode is None:
        return "running"
    last_event = events[-1] if events else None
    if returncode == 0:
        return "done"
    if last_event and last_event.event == "failed":
        return "failed"
    return "failed"


async def monitor_run(
    run_id: str,
    run_dir: pathlib.Path,
    engine: sqlalchemy.Engine,
) -> None:
    metrics_db = run_dir / "metrics.db"
    proc = _active.get(run_id)
    if proc is None:
        return
    try:
        while True:
            await asyncio.sleep(2)
            returncode = proc.poll()
            events = metrics_reader.read_phase_events(metrics_db)
            status = _infer_status(events, returncode)
            ended_at: float | None = time.time() if returncode is not None else None
            with sqlalchemy.orm.Session(engine) as session:
                session.execute(
                    sqlalchemy.update(models.runs)
                    .where(models.runs.c.id == run_id)
                    .values(status=status, ended_at=ended_at)
                )
                session.commit()
            if returncode is not None:
                _active.pop(run_id, None)
                break
    except asyncio.CancelledError:
        pass
