import hashlib
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import time
import uuid

import sqlalchemy
import sqlalchemy.orm
import yaml

from . import compute, config, container, exceptions, logs, orchestrator, reconcile, types
from . import db as dbmod


def _config_hash(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_compute(compute_spec: str) -> str:
    if compute_spec == "cpu":
        return "cpu"
    if compute_spec.startswith("gpu"):
        try:
            gpu_index = int(compute_spec[3:])
            info = compute.collect_compute()
            for gpu in info.gpus:
                if gpu.index == gpu_index:
                    return compute_spec
        except (ValueError, IndexError):
            pass
    raise exceptions.UI(
        f"unknown --compute value '{compute_spec}'; run 'utrain compute list' to see valid values"
    )


def _write_config(
    run_dir: pathlib.Path,
    run_id: str,
    compute: str,
    describe: container.schema.DescribeOutput,
) -> None:
    schema = describe.config_schema
    cfg: dict[str, object] = {
        "run_id": run_id,
        "compute": compute,
    }
    # globals section
    globals_dict: dict[str, object] = {}
    for group in schema.globals.groups:
        for field in group.fields:
            globals_dict.setdefault(group.name, {})[field.key] = field.default  # type: ignore[index]
    if globals_dict:
        cfg["globals"] = globals_dict
    # phases section
    phases_dict: dict[str, object] = {}
    for phase_name, phase_schema in schema.phases.items():
        phase_cfg: dict[str, object] = {}
        for group in phase_schema.groups:
            for field in group.fields:
                phase_cfg[field.key] = field.default
        if phase_cfg:
            phases_dict[phase_name] = phase_cfg
    if phases_dict:
        cfg["phases"] = phases_dict
    config_path = run_dir / "config.yaml"
    config_path.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))


def _run_row(
    row: sqlalchemy.engine.RowMapping,
    session: sqlalchemy.orm.Session,
) -> types.RunRow:
    run_id = str(row["id"])
    attempt_n = dbmod.latest_attempt(run_id, session)

    phase: str | None = None
    if attempt_n is not None:
        phase_rows = (
            session.execute(
                sqlalchemy.select(dbmod.run_phases)
                .where(
                    (dbmod.run_phases.c.run_id == run_id)
                    & (dbmod.run_phases.c.attempt == attempt_n)
                    & (dbmod.run_phases.c.status.notin_(["done", "stopped", "failed"]))
                )
                .order_by(dbmod.run_phases.c.phase_order)
            )
            .mappings()
            .fetchall()
        )
        if phase_rows:
            phase = str(phase_rows[0]["phase"])

    return types.RunRow(
        id=run_id,
        name=str(row["name"]),
        image=str(row["image"]),
        compute=str(row["compute"]),
        status=str(row["status"]),
        created_at=float(row["created_at"]),
        attempt=attempt_n,
        phase=phase,
    )


def list_runs(session: sqlalchemy.orm.Session) -> list[types.RunRow]:
    for run_id in list_run_ids(session):
        attempt_n = dbmod.latest_attempt(run_id, session)
        if attempt_n is not None:
            reconcile.reconcile_attempt(run_id, attempt_n, session)

    rows = (
        session.execute(sqlalchemy.select(dbmod.runs).order_by(dbmod.runs.c.created_at.desc()))
        .mappings()
        .fetchall()
    )
    return [_run_row(r, session) for r in rows]


def list_run_ids(session: sqlalchemy.orm.Session) -> list[str]:
    rows = (
        session.execute(sqlalchemy.select(dbmod.runs.c.id).order_by(dbmod.runs.c.created_at.desc()))
        .scalars()
        .fetchall()
    )
    return [str(r) for r in rows]


def get_run(run_id_prefix: str, session: sqlalchemy.orm.Session) -> types.RunRow:
    run_id = dbmod.resolve_run_id(run_id_prefix, session)
    return _run_row(dbmod.get_run(run_id, session), session)


def config_path(run_id_prefix: str, session: sqlalchemy.orm.Session) -> pathlib.Path:
    """Path of a run's config.yaml, for a caller that wants to edit it."""
    run_id = dbmod.resolve_run_id(run_id_prefix, session)
    row = dbmod.get_run(run_id, session)
    return pathlib.Path(str(row["run_dir"])) / "config.yaml"


def get_run_detail(
    run_id_prefix: str,
    session: sqlalchemy.orm.Session,
    wait: bool = False,
    timeout: int = 600,
) -> types.RunDetail:
    run_id = dbmod.resolve_run_id(run_id_prefix, session)
    row = dbmod.get_run(run_id, session)

    if wait:
        wait_for_run(run_id, session, timeout)
        # Re-fetch after waiting
        row = dbmod.get_run(run_id, session)

    attempt_n = dbmod.latest_attempt(run_id, session)
    if attempt_n is not None:
        reconcile.reconcile_attempt(run_id, attempt_n, session)
        row = dbmod.get_run(run_id, session)

    return _run_detail(run_id, row, attempt_n, session)


def wait_for_run(run_id: str, session: sqlalchemy.orm.Session, timeout: int) -> None:
    deadline = time.time() + timeout if timeout > 0 else None
    while True:
        attempt_n = dbmod.latest_attempt(run_id, session)
        if attempt_n is not None:
            reconcile.reconcile_attempt(run_id, attempt_n, session)
        row = dbmod.get_run(run_id, session)
        if str(row["status"]) in ("done", "failed", "stopped"):
            return
        if deadline is not None and time.time() > deadline:
            raise exceptions.UI(f"run '{run_id}' did not finish within {timeout}s")
        # Commit so this poll's reconcile writes don't hold the SQLite write lock
        # across the sleep — the detached orchestrator needs to write concurrently.
        session.commit()
        time.sleep(1)


def phase_rows(
    run_id: str,
    attempt_n: int,
    session: sqlalchemy.orm.Session,
) -> list[types.PhaseRow]:
    rows = (
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
    return [
        types.PhaseRow(
            phase=str(p["phase"]),
            phase_order=int(p["phase_order"]),
            status=str(p["status"]),
            started_at=p["started_at"],
            ended_at=p["ended_at"],
        )
        for p in rows
    ]


def _run_detail(
    run_id: str,
    row: sqlalchemy.engine.RowMapping,
    attempt_n: int | None,
    session: sqlalchemy.orm.Session,
) -> types.RunDetail:
    all_attempts = (
        session.execute(
            sqlalchemy.select(dbmod.run_attempts)
            .where(dbmod.run_attempts.c.run_id == run_id)
            .order_by(dbmod.run_attempts.c.attempt)
        )
        .mappings()
        .fetchall()
    )

    return types.RunDetail(
        run=_run_row(row, session),
        run_dir=pathlib.Path(str(row["run_dir"])),
        n_attempts=len(all_attempts),
        latest_attempt_status=(
            str(all_attempts[-1]["status"]) if all_attempts else str(row["status"])
        ),
        phases=phase_rows(run_id, attempt_n, session) if attempt_n is not None else [],
    )


def create_run(
    name: str,
    image: str,
    compute_spec: str,
    settings: config.Settings,
    session: sqlalchemy.orm.Session,
) -> str:
    presets = container.podman.list_presets()
    if image not in presets:
        raise exceptions.UI(f"image '{image}' not found")

    describe = container.podman.describe(presets[image])
    if not describe.phase_order:
        raise exceptions.UI(f"image '{image}' has no phases")

    compute_value = _resolve_compute(compute_spec)
    run_id = uuid.uuid4().hex
    run_dir = settings.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_config(run_dir, run_id, compute_value, describe)

    now = time.time()
    session.execute(
        sqlalchemy.insert(dbmod.runs).values(
            id=run_id,
            name=name,
            image=image,
            compute=compute_value,
            run_dir=str(run_dir),
            status="configuring",
            config_hash=None,
            created_at=now,
        )
    )

    return run_id


def start_run(run_id_prefix: str, session: sqlalchemy.orm.Session) -> str:
    run_id = dbmod.resolve_run_id(run_id_prefix, session)
    row = dbmod.get_run(run_id, session)

    status = str(row["status"])
    if status == "running":
        raise exceptions.UI("run is already running")
    if status in ("done", "failed", "stopped"):
        raise exceptions.UI("run is terminal; use 'run restart' to re-run")
    if status != "configuring":
        raise exceptions.UI(f"unexpected run status '{status}'")

    # Reconcile (no-op if status is configuring)
    attempt_n = dbmod.latest_attempt(run_id, session)
    if attempt_n is not None:
        reconcile.reconcile_attempt(run_id, attempt_n, session)

    run_dir = pathlib.Path(str(row["run_dir"]))
    config_path = run_dir / "config.yaml"

    chash = _config_hash(config_path)
    session.execute(
        sqlalchemy.update(dbmod.runs).where(dbmod.runs.c.id == run_id).values(config_hash=chash)
    )
    os.chmod(config_path, 0o444)

    attempt = 1
    attempt_dir = run_dir / "attempt" / str(attempt)
    logs_dir = attempt_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    orchestrator.init_mount_dir(attempt_dir, config_path)

    image_key = str(row["image"])
    now = time.time()
    session.execute(
        sqlalchemy.insert(dbmod.run_attempts).values(
            run_id=run_id,
            attempt=attempt,
            from_phase=None,
            status="running",
            pid=None,
            started_at=now,
            ended_at=None,
        )
    )

    # Determine phase order from image
    presets = container.podman.list_presets()
    describe = container.podman.describe(presets[image_key])
    for i, phase in enumerate(describe.phase_order):
        session.execute(
            sqlalchemy.insert(dbmod.run_phases).values(
                run_id=run_id,
                attempt=attempt,
                phase=phase,
                phase_order=i,
                status="pending",
                started_at=None,
                ended_at=None,
            )
        )

    # Commit so the orchestrator (separate process) can read these rows
    session.commit()

    orch_log = attempt_dir / "orchestrator.log"
    orch_stdout = open(orch_log, "wb")
    proc = subprocess.Popen(
        [sys.executable, "-m", "utrain.cli.main", "_orchestrate", run_id, str(attempt)],
        stdout=orch_stdout,
        stderr=orch_stdout,
        start_new_session=True,
    )

    session.execute(
        sqlalchemy.update(dbmod.run_attempts)
        .where((dbmod.run_attempts.c.run_id == run_id) & (dbmod.run_attempts.c.attempt == attempt))
        .values(pid=proc.pid)
    )

    # Optimistically mark first phase as running
    if describe.phase_order:
        session.execute(
            sqlalchemy.update(dbmod.run_phases)
            .where(
                (dbmod.run_phases.c.run_id == run_id)
                & (dbmod.run_phases.c.attempt == attempt)
                & (dbmod.run_phases.c.phase == describe.phase_order[0])
            )
            .values(status="running", started_at=now)
        )

    session.execute(
        sqlalchemy.update(dbmod.runs).where(dbmod.runs.c.id == run_id).values(status="running")
    )

    return run_id


def stop_run(run_id_prefix: str, session: sqlalchemy.orm.Session) -> str:
    run_id = dbmod.resolve_run_id(run_id_prefix, session)
    row = dbmod.get_run(run_id, session)

    status = str(row["status"])
    if status != "running":
        # Idempotent
        if status == "stopped":
            return run_id
        raise exceptions.UI(f"run is not running (status: {status})")

    attempt_n = dbmod.latest_attempt(run_id, session)
    if attempt_n is not None:
        _stop_attempt(run_id, attempt_n, session)

    session.execute(
        sqlalchemy.update(dbmod.runs).where(dbmod.runs.c.id == run_id).values(status="stopped")
    )

    return run_id


def _stop_attempt(run_id: str, attempt_n: int, session: sqlalchemy.orm.Session) -> None:
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
        return

    run_row = dbmod.get_run(run_id, session)
    run_dir = pathlib.Path(str(run_row["run_dir"]))
    attempt_dir = run_dir / "attempt" / str(attempt_n)
    orchestrator.write_control(attempt_dir, "stop")

    pid = attempt_row["pid"]
    if pid is not None:
        try:
            os.kill(int(pid), signal.SIGTERM)
        except OSError:
            pass

    now = time.time()
    session.execute(
        sqlalchemy.update(dbmod.run_phases)
        .where(
            (dbmod.run_phases.c.run_id == run_id)
            & (dbmod.run_phases.c.attempt == attempt_n)
            & (dbmod.run_phases.c.status.in_(["running", "pending"]))
        )
        .values(status="stopped", ended_at=now)
    )
    session.execute(
        sqlalchemy.update(dbmod.run_attempts)
        .where(
            (dbmod.run_attempts.c.run_id == run_id) & (dbmod.run_attempts.c.attempt == attempt_n)
        )
        .values(status="stopped", ended_at=now)
    )


def restart_run(
    run_id_prefix: str,
    from_phase: str | None,
    session: sqlalchemy.orm.Session,
) -> str:
    run_id = dbmod.resolve_run_id(run_id_prefix, session)
    row = dbmod.get_run(run_id, session)

    status = str(row["status"])
    if status == "configuring":
        raise exceptions.UI("run has not started yet; use 'run start'")

    attempt_n = dbmod.latest_attempt(run_id, session)
    if attempt_n is not None:
        reconcile.reconcile_attempt(run_id, attempt_n, session)
        row = dbmod.get_run(run_id, session)

    if str(row["status"]) == "running" and attempt_n is not None:
        _stop_attempt(run_id, attempt_n, session)

    image_key = str(row["image"])
    run_dir = pathlib.Path(str(row["run_dir"]))
    config_path = run_dir / "config.yaml"

    # Validate from_phase
    from_phase_order: int | None = None
    presets = container.podman.list_presets()
    describe = container.podman.describe(presets[image_key])

    if from_phase is not None:
        if from_phase not in describe.phase_order:
            raise exceptions.UI(f"phase '{from_phase}' not found in image")
        from_phase_order = describe.phase_order.index(from_phase)

    new_attempt = (attempt_n or 0) + 1
    attempt_dir = run_dir / "attempt" / str(new_attempt)
    logs_dir = attempt_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    orchestrator.init_mount_dir(attempt_dir, config_path)

    chash = _config_hash(config_path)
    session.execute(
        sqlalchemy.update(dbmod.runs).where(dbmod.runs.c.id == run_id).values(config_hash=chash)
    )

    now = time.time()
    session.execute(
        sqlalchemy.insert(dbmod.run_attempts).values(
            run_id=run_id,
            attempt=new_attempt,
            from_phase=from_phase,
            status="running",
            pid=None,
            started_at=now,
            ended_at=None,
        )
    )

    start_order = from_phase_order if from_phase_order is not None else 0
    for i, phase in enumerate(describe.phase_order):
        if i < start_order:
            continue
        session.execute(
            sqlalchemy.insert(dbmod.run_phases).values(
                run_id=run_id,
                attempt=new_attempt,
                phase=phase,
                phase_order=i,
                status="pending",
                started_at=None,
                ended_at=None,
            )
        )

    session.flush()

    orch_log = attempt_dir / "orchestrator.log"
    orch_stdout = open(orch_log, "wb")
    orch_args = [
        sys.executable,
        "-m",
        "utrain.cli.main",
        "_orchestrate",
        run_id,
        str(new_attempt),
    ]
    if from_phase is not None:
        orch_args += ["--from-phase", from_phase]

    proc = subprocess.Popen(
        orch_args,
        stdout=orch_stdout,
        stderr=orch_stdout,
        start_new_session=True,
    )

    session.execute(
        sqlalchemy.update(dbmod.run_attempts)
        .where(
            (dbmod.run_attempts.c.run_id == run_id) & (dbmod.run_attempts.c.attempt == new_attempt)
        )
        .values(pid=proc.pid)
    )

    first_active = (
        describe.phase_order[start_order] if start_order < len(describe.phase_order) else None
    )
    if first_active:
        session.execute(
            sqlalchemy.update(dbmod.run_phases)
            .where(
                (dbmod.run_phases.c.run_id == run_id)
                & (dbmod.run_phases.c.attempt == new_attempt)
                & (dbmod.run_phases.c.phase == first_active)
            )
            .values(status="running", started_at=now)
        )

    session.execute(
        sqlalchemy.update(dbmod.runs).where(dbmod.runs.c.id == run_id).values(status="running")
    )

    return run_id


def delete_run(run_id_prefix: str, force: bool, session: sqlalchemy.orm.Session) -> str:
    run_id = dbmod.resolve_run_id(run_id_prefix, session)
    row = dbmod.get_run(run_id, session)

    attempt_n = dbmod.latest_attempt(run_id, session)
    if attempt_n is not None:
        reconcile.reconcile_attempt(run_id, attempt_n, session)
        row = dbmod.get_run(run_id, session)

    if str(row["status"]) == "running":
        if not force:
            raise exceptions.UI("run is running; use --force or stop it first")
        if attempt_n is not None:
            _stop_attempt(run_id, attempt_n, session)

    run_dir = pathlib.Path(str(row["run_dir"]))
    shutil.rmtree(run_dir, ignore_errors=True)

    session.execute(sqlalchemy.delete(dbmod.run_phases).where(dbmod.run_phases.c.run_id == run_id))
    session.execute(
        sqlalchemy.delete(dbmod.run_attempts).where(dbmod.run_attempts.c.run_id == run_id)
    )
    session.execute(sqlalchemy.delete(dbmod.runs).where(dbmod.runs.c.id == run_id))

    return run_id


def read_log_tail(
    run_id_prefix: str,
    session: sqlalchemy.orm.Session,
    attempt: int | None = None,
    phase: str | None = None,
    stderr: bool = False,
    tail: int = 200,
) -> list[str]:
    run_id = dbmod.resolve_run_id(run_id_prefix, session)
    row = dbmod.get_run(run_id, session)
    run_dir = pathlib.Path(str(row["run_dir"]))

    if attempt is None:
        attempt_n = dbmod.latest_attempt(run_id, session)
        if attempt_n is None:
            raise exceptions.UI("run has no attempts yet")
        attempt = attempt_n

    attempt_dir = run_dir / "attempt" / str(attempt)

    if phase is not None:
        suffix = "stderr" if stderr else "stdout"
        log_file = attempt_dir / "logs" / f"{phase}_{suffix}.log"
    else:
        log_file = attempt_dir / "orchestrator.log"

    if not log_file.exists():
        raise exceptions.UI(f"log file not found: {log_file}")

    return logs.tail_lines(log_file, tail)
