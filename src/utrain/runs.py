import collections.abc
import contextlib
import hashlib
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import time
import typing
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
        image_id=None if row["image_id"] is None else str(row["image_id"]),
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
    return dbmod.run_dir(run_id, session) / "config.yaml"


@contextlib.contextmanager
def writable(path: pathlib.Path) -> collections.abc.Generator[None]:
    """Make `config.yaml` writable for the duration of an edit.

    `start_run` chmods the file to 0o444 once its hash has been recorded, so
    that an edit after the fact cannot silently diverge from `runs.config_hash`.
    Every deliberate edit -- `run show --edit`, the TUI's config form -- has to
    lift that and put it back, so the pair lives in one place.

    A file that is not there yet has nothing to lift and nothing to put back:
    `write_config` writes the first `config.yaml` of a run that has none.
    """
    if not path.exists():
        yield
        return
    mode = path.stat().st_mode
    os.chmod(path, 0o644)
    try:
        yield
    finally:
        os.chmod(path, mode)


def _mapping(value: object) -> dict[str, object]:
    """`value` as a mapping, or an empty one.

    Config comes off disk as `yaml.safe_load`'s `Any` and out of a form as
    `object`; every place that indexes into it wants the same guard, and wants
    the result typed rather than `Any`.
    """
    if not isinstance(value, dict):
        return {}
    return typing.cast(dict[str, object], value)


def read_config(run_id_prefix: str, session: sqlalchemy.orm.Session) -> dict[str, object]:
    """A run's config.yaml, parsed. Empty when the file is missing."""
    path = config_path(run_id_prefix, session)
    if not path.exists():
        return {}
    return _mapping(yaml.safe_load(path.read_text()))


def _coerce(field: container.schema.FieldSchema, value: object) -> int | float | str | bool | None:
    """One form value, checked against the field that declared it.

    The TUI edits through widgets that already restrict what can be typed, but
    the check belongs here rather than there: a validated write is a property of
    the query layer, not of one client that happens to be careful.
    """
    if value is None or value == "":
        if field.required:
            raise exceptions.UI(f"'{field.key}' is required")
        return None

    if field.type == "bool":
        if isinstance(value, bool):
            return value
        raise exceptions.UI(f"'{field.key}' must be true or false, got '{value}'")

    if field.type == "enum":
        text = str(value)
        if field.options and text not in field.options:
            raise exceptions.UI(
                f"'{field.key}' must be one of {', '.join(field.options)}, got '{text}'"
            )
        return text

    if field.type == "str":
        return str(value)

    kind = "an int" if field.type == "int" else "a float"
    # `bool` is an `int` subclass, so it would otherwise coerce silently to 0/1.
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise exceptions.UI(f"'{field.key}' must be {kind}, got '{value}'")
    try:
        number: int | float = int(value) if field.type == "int" else float(value)
    except ValueError:
        raise exceptions.UI(f"'{field.key}' must be {kind}, got '{value}'")
    if field.min is not None and number < field.min:
        raise exceptions.UI(f"'{field.key}' must be at least {field.min}, got {number}")
    if field.max is not None and number > field.max:
        raise exceptions.UI(f"'{field.key}' must be at most {field.max}, got {number}")
    return number


def _validated_section(
    groups: list[container.schema.FieldGroup],
    values: dict[str, object],
    flat: bool,
) -> dict[str, object]:
    """The fields `groups` declares, coerced out of `values`.

    Globals nest one level per group; phases are flat, mirroring what
    `_write_config` lays out. Unknown keys are dropped rather than rejected: the
    schema is what the image will read, so anything else could not have any
    effect anyway.
    """
    out: dict[str, object] = {}
    for group in groups:
        source = values if flat else _mapping(values.get(group.name))
        section = {f.key: _coerce(f, source.get(f.key, f.default)) for f in group.fields}
        if not section:
            continue
        if flat:
            out.update(section)
        else:
            out[group.name] = section
    return out


def write_config(
    run_id_prefix: str,
    values: dict[str, object],
    session: sqlalchemy.orm.Session,
    described: container.schema.DescribeOutput | None = None,
) -> None:
    """Replace a run's config.yaml with `values`, validated against its schema.

    Only a `configuring` run can be edited: once started, `runs.config_hash`
    records what the attempt actually ran with, and a later edit would make that
    a lie.

    `run_id` and `compute` are carried over from the file rather than taken from
    the caller. Neither is a config field -- `compute` was validated against the
    host at creation -- so neither is the form's to change.
    """
    run_id = dbmod.resolve_run_id(run_id_prefix, session)
    row = dbmod.get_run(run_id, session)
    status = str(row["status"])
    if status != "configuring":
        raise exceptions.UI(f"run '{run_id}' is {status}; only a configuring run can be edited")

    if described is None:
        described = container.podman.describe(dbmod.run_image_ref(row, session))
    schema = described.config_schema

    path = dbmod.run_dir(run_id, session) / "config.yaml"
    current = _mapping(yaml.safe_load(path.read_text())) if path.exists() else {}

    cfg: dict[str, object] = {"run_id": run_id, "compute": current.get("compute")}
    globals_out = _validated_section(
        schema.globals.groups, _mapping(values.get("globals")), flat=False
    )
    if globals_out:
        cfg["globals"] = globals_out

    phase_values = _mapping(values.get("phases"))
    phases_out: dict[str, object] = {}
    for phase_name, phase_schema in schema.phases.items():
        section = _validated_section(
            phase_schema.groups, _mapping(phase_values.get(phase_name)), flat=True
        )
        if section:
            phases_out[phase_name] = section
    if phases_out:
        cfg["phases"] = phases_out

    with writable(path):
        path.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))


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
    rid = dbmod.short_run_id(run_id, session)
    return [
        types.PhaseRow(
            phase=str(p["phase"]),
            phase_order=int(p["phase_order"]),
            status=str(p["status"]),
            started_at=p["started_at"],
            ended_at=p["ended_at"],
            address=f"{rid}/{attempt_n}/{p['phase']}",
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
        run_dir=dbmod.run_dir(run_id, session),
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

    # Freeze the id before describing: the run must keep pointing at exactly
    # this content even if the name is re-tagged to another image later.
    image_id = container.podman.image_id(presets[image])
    describe = container.podman.describe(image_id)
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
            image_id=image_id,
            compute=compute_value,
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

    # Preflight while still in the foreground process: the orchestrator runs
    # detached with its output in orchestrator.log, so a GPU run whose toolkit
    # is missing must be refused here for the user to see why (issue #26).
    orchestrator.ensure_gpu_toolkit(str(row["compute"]))

    # Reconcile (no-op if status is configuring)
    attempt_n = dbmod.latest_attempt(run_id, session)
    if attempt_n is not None:
        reconcile.reconcile_attempt(run_id, attempt_n, session)

    run_dir = dbmod.run_dir(run_id, session)
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

    # Determine phase order from the frozen image id
    describe = container.podman.describe(dbmod.run_image_ref(row, session))
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

    run_dir = dbmod.run_dir(run_id, session)
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

    # Same preflight as start_run: refuse before anything is stopped or created.
    orchestrator.ensure_gpu_toolkit(str(row["compute"]))

    attempt_n = dbmod.latest_attempt(run_id, session)
    if attempt_n is not None:
        reconcile.reconcile_attempt(run_id, attempt_n, session)
        row = dbmod.get_run(run_id, session)

    if str(row["status"]) == "running" and attempt_n is not None:
        _stop_attempt(run_id, attempt_n, session)

    run_dir = dbmod.run_dir(run_id, session)
    config_path = run_dir / "config.yaml"

    # Validate from_phase against the frozen image id
    from_phase_order: int | None = None
    describe = container.podman.describe(dbmod.run_image_ref(row, session))

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

    run_dir = dbmod.run_dir(run_id, session)
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
    tail: int = 200,
) -> list[str]:
    run_id = dbmod.resolve_run_id(run_id_prefix, session)
    run_dir = dbmod.run_dir(run_id, session)

    if attempt is None:
        attempt_n = dbmod.latest_attempt(run_id, session)
        if attempt_n is None:
            raise exceptions.UI("run has no attempts yet")
        attempt = attempt_n

    attempt_dir = run_dir / "attempt" / str(attempt)

    if phase is not None:
        log_file = logs.phase_log_path(attempt_dir, phase)
    else:
        log_file = attempt_dir / "orchestrator.log"

    if not log_file.exists():
        raise exceptions.UI(f"log file not found: {log_file}")

    return logs.tail_lines(log_file, tail)
