import hashlib
import json
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

from .. import config, container
from . import db as dbmod
from . import output, reconcile, exceptions


def _config_hash(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_gpu(gpu_spec: str) -> str:
    if gpu_spec != "auto":
        return gpu_spec
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            indices = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
            if indices:
                return indices[0]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "none"


def _write_config(
    run_dir: pathlib.Path,
    run_id: str,
    gpu: str,
    describe: container.schema.DescribeOutput,
) -> None:
    schema = describe.config_schema
    cfg: dict[str, object] = {
        "run_id": run_id,
        "output_dir": "/run",
        "gpu": gpu,
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


def _find_min_prefix_len(run_ids: list[str]) -> int:
    if not run_ids:
        return 1
    for prefix_len in range(1, 33):
        prefixes = {rid[:prefix_len] for rid in run_ids}
        if len(prefixes) == len(run_ids):
            return prefix_len
    return 32


def _format_run_row(
    row: sqlalchemy.engine.RowMapping,
    session: sqlalchemy.orm.Session,
    prefix_len: int = 8,
) -> list[str]:
    run_id = str(row["id"])
    attempt_n = dbmod.latest_attempt(run_id, session)
    attempt_str = str(attempt_n) if attempt_n is not None else "--"

    phase_str = "--"
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
            phase_str = str(phase_rows[0]["phase"])

    return [
        run_id[:prefix_len],
        str(row["name"]),
        str(row["image"]),
        str(row["gpu"]),
        str(row["status"]),
        attempt_str,
        phase_str,
        output.format_time(row["created_at"]),
    ]


def list_runs(session: sqlalchemy.orm.Session, short: bool = False) -> None:
    rows = (
        session.execute(sqlalchemy.select(dbmod.runs).order_by(dbmod.runs.c.created_at.desc()))
        .mappings()
        .fetchall()
    )
    for row in rows:
        attempt_n = dbmod.latest_attempt(str(row["id"]), session)
        if attempt_n is not None:
            reconcile.reconcile_attempt(str(row["id"]), attempt_n, session)

    run_ids = [str(r["id"]) for r in rows]
    prefix_len = _find_min_prefix_len(run_ids)

    headers = ["ID", "NAME", "IMAGE", "GPU", "STATUS", "ATTEMPT", "PHASE", "CREATED"]
    table_rows = [_format_run_row(r, session, prefix_len) for r in rows]
    print(output.format_table(headers, table_rows))


def list_run_ids(session: sqlalchemy.orm.Session) -> list[str]:
    rows = (
        session.execute(sqlalchemy.select(dbmod.runs.c.id).order_by(dbmod.runs.c.created_at.desc()))
        .scalars()
        .fetchall()
    )
    return [str(r) for r in rows]


def list_runs_json(session: sqlalchemy.orm.Session) -> None:
    import datetime

    rows = (
        session.execute(sqlalchemy.select(dbmod.runs).order_by(dbmod.runs.c.created_at.desc()))
        .mappings()
        .fetchall()
    )
    result: list[dict[str, str]] = []
    for row in rows:
        run_id = str(row["id"])
        attempt_n = dbmod.latest_attempt(run_id, session)
        if attempt_n is not None:
            reconcile.reconcile_attempt(run_id, attempt_n, session)
        result.append(
            {
                "id": run_id,
                "name": str(row["name"]),
                "image": str(row["image"]),
                "gpu": str(row["gpu"]),
                "status": str(row["status"]),
                "created_at": datetime.datetime.fromtimestamp(row["created_at"]).isoformat(),
            }
        )
    import json as _json

    print(_json.dumps(result))


def show_run(
    run_id_prefix: str,
    session: sqlalchemy.orm.Session,
    wait: bool = False,
    timeout: int = 600,
    edit: bool = False,
) -> None:
    run_id = dbmod.resolve_run_id(run_id_prefix, session)
    row = dbmod.get_run(run_id, session)

    if edit:
        config_path = pathlib.Path(str(row["run_dir"])) / "config.yaml"
        os.chmod(config_path, 0o644)
        editor = os.environ.get("EDITOR", "vi")
        subprocess.run([editor, str(config_path)])
        os.chmod(config_path, 0o444)
        return

    if wait:
        _wait_for_run(run_id, session, timeout)
        # Re-fetch after waiting
        row = dbmod.get_run(run_id, session)

    attempt_n = dbmod.latest_attempt(run_id, session)
    if attempt_n is not None:
        reconcile.reconcile_attempt(run_id, attempt_n, session)
        row = dbmod.get_run(run_id, session)

    _print_run_detail(run_id, row, attempt_n, session)


def _wait_for_run(run_id: str, session: sqlalchemy.orm.Session, timeout: int) -> None:
    deadline = time.time() + timeout if timeout > 0 else None
    while True:
        attempt_n = dbmod.latest_attempt(run_id, session)
        if attempt_n is not None:
            reconcile.reconcile_attempt(run_id, attempt_n, session)
        row = dbmod.get_run(run_id, session)
        if str(row["status"]) in ("done", "failed", "stopped"):
            return
        if deadline is not None and time.time() > deadline:
            raise exceptions.UI(f"abort: run '{run_id}' did not finish within {timeout}s")
        time.sleep(1)


def _print_run_detail(
    run_id: str,
    row: sqlalchemy.engine.RowMapping,
    attempt_n: int | None,
    session: sqlalchemy.orm.Session,
) -> None:
    run_dir = pathlib.Path(str(row["run_dir"]))
    all_attempts = (
        session.execute(
            sqlalchemy.select(dbmod.run_attempts)
            .where(dbmod.run_attempts.c.run_id == run_id)
            .order_by(dbmod.run_attempts.c.attempt)
        )
        .mappings()
        .fetchall()
    )
    n_attempts = len(all_attempts)
    latest_status = str(all_attempts[-1]["status"]) if all_attempts else str(row["status"])
    attempts_str = f"{n_attempts} (latest: {latest_status})" if n_attempts else "0"

    print(f"id:       {run_id}")
    print(f"name:     {row['name']}")
    print(f"image:    {row['image']}")
    print(f"gpu:      {row['gpu']}")
    print(f"status:   {row['status']}")
    print(f"attempts: {attempts_str}")
    print(f"created:  {output.format_time(row['created_at'])}")

    if attempt_n is not None:
        phase_rows = (
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
        print()
        print(f"config: {run_dir / 'config.yaml'}")
        print(f"logs:   {logs_dir}")


def print_run_row(run_id: str, session: sqlalchemy.orm.Session) -> None:
    row = dbmod.get_run(run_id, session)
    all_run_ids = list_run_ids(session)
    prefix_len = _find_min_prefix_len(all_run_ids)
    headers = ["ID", "NAME", "IMAGE", "GPU", "STATUS", "ATTEMPT", "PHASE", "CREATED"]
    print(output.format_table(headers, [_format_run_row(row, session, prefix_len)]))


def create_run(
    name: str,
    image: str,
    gpu_spec: str,
    settings: config.Settings,
    session: sqlalchemy.orm.Session,
    print_id: bool = False,
) -> str:
    presets = container.enroot.list_presets()
    if image not in presets:
        raise exceptions.UI(f"abort: image '{image}' not found")

    describe = container.enroot.describe(presets[image])
    if not describe.phase_order:
        raise exceptions.UI(f"abort: image '{image}' has no phases")

    gpu = _resolve_gpu(gpu_spec)
    run_id = uuid.uuid4().hex
    run_dir = settings.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_config(run_dir, run_id, gpu, describe)

    now = time.time()
    session.execute(
        sqlalchemy.insert(dbmod.runs).values(
            id=run_id,
            name=name,
            image=image,
            gpu=gpu,
            run_dir=str(run_dir),
            status="configuring",
            config_hash=None,
            created_at=now,
        )
    )

    if print_id:
        print(run_id)
    else:
        print_run_row(run_id, session)

    return run_id


def start_run(run_id_prefix: str, session: sqlalchemy.orm.Session) -> None:
    run_id = dbmod.resolve_run_id(run_id_prefix, session)
    row = dbmod.get_run(run_id, session)

    status = str(row["status"])
    if status == "running":
        raise exceptions.UI("abort: run is already running")
    if status in ("done", "failed", "stopped"):
        raise exceptions.UI("abort: run is terminal; use 'run restart' to re-run")
    if status != "configuring":
        raise exceptions.UI(f"abort: unexpected run status '{status}'")

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

    shutil.copy2(config_path, attempt_dir / "config.yaml")
    os.chmod(attempt_dir / "config.yaml", 0o444)

    control_path = attempt_dir / "control.json"
    control_path.write_text(json.dumps({"action": "continue"}))

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
    presets = container.enroot.list_presets()
    describe = container.enroot.describe(presets[image_key])
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

    print_run_row(run_id, session)


def stop_run(run_id_prefix: str, session: sqlalchemy.orm.Session) -> None:
    run_id = dbmod.resolve_run_id(run_id_prefix, session)
    row = dbmod.get_run(run_id, session)

    status = str(row["status"])
    if status != "running":
        # Idempotent
        if status == "stopped":
            print_run_row(run_id, session)
            return
        raise exceptions.UI(f"abort: run is not running (status: {status})")

    attempt_n = dbmod.latest_attempt(run_id, session)
    if attempt_n is not None:
        _stop_attempt(run_id, attempt_n, session)

    session.execute(
        sqlalchemy.update(dbmod.runs).where(dbmod.runs.c.id == run_id).values(status="stopped")
    )

    print_run_row(run_id, session)


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
    control_path = attempt_dir / "control.json"
    control_path.write_text(json.dumps({"action": "stop"}))

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
) -> None:
    run_id = dbmod.resolve_run_id(run_id_prefix, session)
    row = dbmod.get_run(run_id, session)

    status = str(row["status"])
    if status == "configuring":
        raise exceptions.UI("abort: run has not started yet; use 'run start'")

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
    presets = container.enroot.list_presets()
    describe = container.enroot.describe(presets[image_key])

    if from_phase is not None:
        if from_phase not in describe.phase_order:
            raise exceptions.UI(f"abort: phase '{from_phase}' not found in image")
        from_phase_order = describe.phase_order.index(from_phase)

    new_attempt = (attempt_n or 0) + 1
    attempt_dir = run_dir / "attempt" / str(new_attempt)
    logs_dir = attempt_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(config_path, attempt_dir / "config.yaml")
    os.chmod(attempt_dir / "config.yaml", 0o444)

    chash = _config_hash(config_path)
    session.execute(
        sqlalchemy.update(dbmod.runs).where(dbmod.runs.c.id == run_id).values(config_hash=chash)
    )

    control_path = attempt_dir / "control.json"
    control_path.write_text(json.dumps({"action": "continue"}))

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

    print_run_row(run_id, session)


def delete_run(run_id_prefix: str, force: bool, session: sqlalchemy.orm.Session) -> None:
    run_id = dbmod.resolve_run_id(run_id_prefix, session)
    row = dbmod.get_run(run_id, session)

    attempt_n = dbmod.latest_attempt(run_id, session)
    if attempt_n is not None:
        reconcile.reconcile_attempt(run_id, attempt_n, session)
        row = dbmod.get_run(run_id, session)

    if str(row["status"]) == "running":
        if not force:
            raise exceptions.UI("abort: run is running; use --force or stop it first")
        if attempt_n is not None:
            _stop_attempt(run_id, attempt_n, session)

    run_dir = pathlib.Path(str(row["run_dir"]))
    shutil.rmtree(run_dir, ignore_errors=True)

    session.execute(sqlalchemy.delete(dbmod.run_phases).where(dbmod.run_phases.c.run_id == run_id))
    session.execute(
        sqlalchemy.delete(dbmod.run_attempts).where(dbmod.run_attempts.c.run_id == run_id)
    )
    session.execute(sqlalchemy.delete(dbmod.runs).where(dbmod.runs.c.id == run_id))

    print(f"removed run {run_id}")


def logs_run(
    run_id_prefix: str,
    session: sqlalchemy.orm.Session,
    attempt: int | None = None,
    phase: str | None = None,
    stderr: bool = False,
    tail: int = 200,
) -> None:
    run_id = dbmod.resolve_run_id(run_id_prefix, session)
    row = dbmod.get_run(run_id, session)
    run_dir = pathlib.Path(str(row["run_dir"]))

    if attempt is None:
        attempt_n = dbmod.latest_attempt(run_id, session)
        if attempt_n is None:
            raise exceptions.UI("abort: run has no attempts yet")
        attempt = attempt_n

    attempt_dir = run_dir / "attempt" / str(attempt)

    if phase is not None:
        suffix = "stderr" if stderr else "stdout"
        log_file = attempt_dir / "logs" / f"{phase}_{suffix}.log"
    else:
        log_file = attempt_dir / "orchestrator.log"

    if not log_file.exists():
        raise exceptions.UI(f"abort: log file not found: {log_file}")

    lines = log_file.read_text(errors="replace").splitlines()
    for line in lines[-tail:]:
        print(line)
