import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import time

import naw
import sqlalchemy
import sqlalchemy.orm

from . import config, container, lock
from . import db as dbmod

# Timeout for a `check-cache` call: it must be cheap (no GPU work, no heavy
# compute -- see docs/container-contract.md), so a generous ceiling is enough
# to catch a container that hangs or ignores the contract without slowing
# down every cacheable phase check.
_MANIFEST_TIMEOUT_S = 30

# Single dir placed first on the container's PYTHONPATH; holds the mounted naw
# package and the wandb shim (see _wandb_mount_args).
_WANDB_PYPATH = "/opt/utrain-py"

# all utrain-specific paths live below /utrain below. Handed over to the container
# as `--utrain-root`. Some paths are mounted ro, others rw, for the container to
# be able to write data
_RUN_MOUNT = "/utrain"
_DATA_MOUNT = f"{_RUN_MOUNT}/data"
_WANDB_MOUNT = f"{_RUN_MOUNT}/wandb"

# Writable, and mounted only for `serve`. It exists so the container has
# somewhere to publish the port it bound
_SERVE_MOUNT = f"{_RUN_MOUNT}/serve"
_PORT_FILE = f"{_SERVE_MOUNT}/port.json"


def _gpu_args(compute: str, attempt_dir: pathlib.Path) -> tuple[list[str], dict[str, str]]:
    """podman args + env that expose the selected GPU inside the container.

    GPUs reach a rootless container through a CDI spec, which pins driver
    library paths by version and so goes stale on every driver update. Rather
    than keep a persistent spec, generate one per run (~0.2s) into the attempt
    dir and point podman at it with CONTAINERS_CONF_OVERRIDE, which is layered
    on top of the system and user config -- unlike CONTAINERS_CONF, which would
    replace both.

    Returns no args and does no nvidia-ctk work for `compute == "cpu"`, so CPU
    runs stay independent of the NVIDIA toolchain.
    """
    env = dict(os.environ)
    if not compute.startswith("gpu"):
        return [], env

    if shutil.which("nvidia-ctk") is None:
        raise RuntimeError(
            "nvidia-ctk not found; install the NVIDIA container toolkit to run on a GPU"
        )

    cdi_dir = attempt_dir / ".cdi"
    cdi_dir.mkdir(parents=True, exist_ok=True)
    spec = cdi_dir / "nvidia.yaml"
    result = subprocess.run(
        ["nvidia-ctk", "cdi", "generate", f"--output={spec}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-ctk cdi generate failed: {result.stderr.strip()}")

    conf = cdi_dir / "containers.conf"
    conf.write_text(f'[engine]\ncdi_spec_dirs = ["{cdi_dir}"]\n')
    env["CONTAINERS_CONF_OVERRIDE"] = str(conf)
    return ["--device", f"nvidia.com/gpu={compute.removeprefix('gpu')}"], env


def _wandb_mount_args() -> list[str]:
    """podman args that shadow the image's real wandb with naw's rtsdb backend.

    Mounts the naw package plus a tiny ``wandb`` shim read-only under a single
    dir put first on PYTHONPATH, so ``import wandb`` inside the container
    resolves to the shim -> ``naw.wandb`` and metrics are written as rtsdb
    files, regardless of what wandb the image itself ships. podman creates the
    mountpoints itself, so no equivalent of enroot's `x-create=dir` is needed.
    """
    naw_dir = pathlib.Path(naw.__file__).parent
    shim_dir = pathlib.Path(container.__file__).parent / "wandb_shim" / "wandb"
    return [
        "-v",
        f"{naw_dir}:{_WANDB_PYPATH}/naw:ro",
        "-v",
        f"{shim_dir}:{_WANDB_PYPATH}/wandb:ro",
        "-e",
        f"PYTHONPATH={_WANDB_PYPATH}",
    ]


def mount_dir(attempt_dir: pathlib.Path) -> pathlib.Path:
    """Host dir bind-mounted read-only at `_RUN_MOUNT`."""
    return attempt_dir / "mnt"


def serve_dir(attempt_dir: pathlib.Path) -> pathlib.Path:
    """Host dir bind-mounted read-write at `_SERVE_MOUNT`, holding `port.json`."""
    return attempt_dir / "serve"


def port_file(attempt_dir: pathlib.Path) -> pathlib.Path:
    """Host path of the port file the container writes once it is listening."""
    return serve_dir(attempt_dir) / "port.json"


def write_control(attempt_dir: pathlib.Path, action: str) -> None:
    """Set the graceful-stop flag the running phase polls."""
    (mount_dir(attempt_dir) / "control.json").write_text(json.dumps({"action": action}))


def init_mount_dir(attempt_dir: pathlib.Path, config_path: pathlib.Path) -> None:
    """Lay out the container's read-only view of the run before starting it.

    The `data`, `wandb` and `serve` entries are deliberately empty: podman cannot
    create a mountpoint inside a read-only bind, so every writable mount layered
    on top of `_RUN_MOUNT` needs its target to already exist in the source dir.

    config.yaml is copied (not linked) and made read-only so the attempt keeps the
    config it was started with even if the user edits the run's copy afterwards.
    """
    mnt = mount_dir(attempt_dir)
    (mnt / "data").mkdir(parents=True, exist_ok=True)
    (mnt / "wandb").mkdir(parents=True, exist_ok=True)
    (mnt / "serve").mkdir(parents=True, exist_ok=True)
    # The real metrics dir, mounted over the stub above. Created here rather than
    # in _start_phase because _check_cache mounts it too, and runs first.
    (attempt_dir / "wandb").mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, mnt / "config.yaml")
    os.chmod(mnt / "config.yaml", 0o444)
    write_control(attempt_dir, "continue")


def _mount_args(attempt_dir: pathlib.Path, data_dir: pathlib.Path, *, data_ro: bool) -> list[str]:
    """The three `-v` flags making up the container's `/utrain` tree.

    `data_ro` is for `check-cache`, which the contract forbids from writing to the
    data dir; mounting it read-only makes that a guarantee rather than a request.
    A container that violates it fails, which the caller already treats as a cache
    miss -- the safe outcome.
    """
    return [
        "-v",
        f"{mount_dir(attempt_dir)}:{_RUN_MOUNT}:ro",
        "-v",
        f"{data_dir}:{_DATA_MOUNT}{':ro' if data_ro else ''}",
        "-v",
        f"{attempt_dir / 'wandb'}:{_WANDB_MOUNT}",
    ]


def _init_phase_data(
    run_id: str,
    attempt: int,
    phase: str,
    phase_order: int,
    run_dir: pathlib.Path,
    session: sqlalchemy.orm.Session,
) -> None:
    """Initialize a phase's data dir as a hardlinked copy of the previous phase's.

    The previous phase is the one with `phase_order - 1`, taken from the highest
    attempt <= this one (same attempt for a full run; an earlier attempt for a
    `--from-phase` restart). The first phase (order 0) is left empty.
    """
    target = run_dir / "attempt" / str(attempt) / "data" / phase
    target.mkdir(parents=True, exist_ok=True)
    if phase_order == 0:
        return
    pred = (
        session.execute(
            sqlalchemy.select(dbmod.run_phases.c.phase, dbmod.run_phases.c.attempt)
            .where(
                (dbmod.run_phases.c.run_id == run_id)
                & (dbmod.run_phases.c.phase_order == phase_order - 1)
                & (dbmod.run_phases.c.attempt <= attempt)
            )
            .order_by(dbmod.run_phases.c.attempt.desc())
            .limit(1)
        )
        .mappings()
        .fetchone()
    )
    if pred is None:
        return
    source = run_dir / "attempt" / str(pred["attempt"]) / "data" / str(pred["phase"])
    if not source.exists() or not any(source.iterdir()):
        return
    subprocess.run(["cp", "-rl", f"{source}/.", str(target)], check=True)


def _start_phase(
    image_key: str,
    attempt_dir: pathlib.Path,
    phase: str,
    compute: str,
) -> subprocess.Popen[bytes]:
    logs_dir = attempt_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout = open(logs_dir / f"{phase}_stdout.log", "wb")
    stderr = open(logs_dir / f"{phase}_stderr.log", "wb")

    data_dir = attempt_dir / "data" / phase
    data_dir.mkdir(parents=True, exist_ok=True)

    gpu_args, env = _gpu_args(compute, attempt_dir)

    # `podman run` proxies SIGTERM to PID 1 and returns the container's exit
    # status, so the caller's terminate()/wait() handling needs no adjustment.
    # `label=disable` avoids an SELinux denial on /dev/nvidia* without needing
    # root, and `--network=host` matches enroot's shared network namespace.
    return subprocess.Popen(
        [
            "podman",
            "run",
            "--rm",
            "--network=host",
            "--security-opt=label=disable",
            *gpu_args,
            *_wandb_mount_args(),
            *_mount_args(attempt_dir, data_dir, data_ro=False),
            container.podman.image_ref(image_key),
            "--utrain-root",
            _RUN_MOUNT,
            "run",
            "--phase",
            phase,
        ],
        stdout=stdout,
        stderr=stderr,
        env=env,
    )


def serve_argv(
    image_key: str,
    attempt_dir: pathlib.Path,
    data_dir: pathlib.Path,
    compute: str,
) -> tuple[list[str], dict[str, str]]:
    """podman argv + env to serve `data_dir` over an OpenAI-compatible endpoint.

    Lives here rather than in the serve command so the mount layout stays defined
    in exactly one place, beside _start_phase and _check_cache.

    `--port 0` tells the container to let the kernel pick a free port, and it
    publishes the one it bound to `_PORT_FILE` inside the writable `serve` mount.

    `data_ro=True` because serving is a read of a finished run: by the time a run
    is servable its data files are hardlinked into the content-addressed store,
    so a write here would corrupt every other run sharing them. And no wandb
    mounts, since serve logs no metrics -- the same reasoning as _check_cache.

    `compute` comes from the run, so a model trained on a GPU is served on one.
    """
    gpu_args, env = _gpu_args(compute, attempt_dir)
    host_serve_dir = serve_dir(attempt_dir)
    host_serve_dir.mkdir(parents=True, exist_ok=True)
    # Created lazily here as well as in init_mount_dir, so a run started before
    # `serve` existed can still be chatted with.
    (mount_dir(attempt_dir) / "serve").mkdir(parents=True, exist_ok=True)
    # A port file left by an earlier session would be read as this one's answer,
    # pointing the client at a dead port.
    port_file(attempt_dir).unlink(missing_ok=True)
    return [
        "podman",
        "run",
        "--rm",
        "--network=host",
        "--security-opt=label=disable",
        *gpu_args,
        *_mount_args(attempt_dir, data_dir, data_ro=True),
        "-v",
        f"{host_serve_dir}:{_SERVE_MOUNT}",
        container.podman.image_ref(image_key),
        "--utrain-root",
        _RUN_MOUNT,
        "serve",
        "--port",
        "0",
    ], env


def _check_cache(
    image_key: str,
    attempt_dir: pathlib.Path,
    phase: str,
    data_dir: pathlib.Path,
) -> container.schema.CacheManifest | None:
    """Ask a cacheable phase what it would produce, without running it.

    Returns None on any protocol violation (non-zero exit, timeout, unparsable
    output) -- the caller treats that exactly like a cache miss.
    """
    try:
        result = subprocess.run(
            [
                "podman",
                "run",
                "--rm",
                "--network=host",
                "--security-opt=label=disable",
                *_mount_args(attempt_dir, data_dir, data_ro=True),
                container.podman.image_ref(image_key),
                "--utrain-root",
                _RUN_MOUNT,
                "check-cache",
                "--phase",
                phase,
            ],
            capture_output=True,
            text=True,
            timeout=_MANIFEST_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        return None
    try:
        return container.schema.CacheManifest.model_validate(json.loads(result.stdout.strip()))
    except Exception:
        return None


def _try_serve_from_cache(
    manifest: container.schema.CacheManifest,
    data_dir: pathlib.Path,
    store_dir: pathlib.Path,
) -> bool:
    """Populate data_dir from the store if every declared file is already there.

    All-or-nothing: on any miss, returns False without touching data_dir, so
    the caller falls through to running the phase for real.
    """
    if not all((store_dir / f.sha256).exists() for f in manifest.files):
        return False
    for f in manifest.files:
        target = data_dir / f.path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        os.link(store_dir / f.sha256, target)
    return True


def run_orchestrator(
    run_id: str,
    attempt: int,
    from_phase: str | None,
    settings: config.Settings,
) -> None:
    engine = dbmod.create_engine(settings)
    dbmod.init_db(engine)

    current_proc: subprocess.Popen[bytes] | None = None
    shutting_down = False

    def handle_sigterm(signum: int, frame: object) -> None:
        nonlocal shutting_down
        shutting_down = True
        if current_proc is not None:
            try:
                current_proc.terminate()
            except OSError:
                pass

    signal.signal(signal.SIGTERM, handle_sigterm)

    with sqlalchemy.orm.Session(engine) as session:
        run_row = (
            session.execute(sqlalchemy.select(dbmod.runs).where(dbmod.runs.c.id == run_id))
            .mappings()
            .fetchone()
        )
        if run_row is None:
            print(f"orchestrator: run '{run_id}' not found", file=sys.stderr)
            sys.exit(1)

        run_dir = pathlib.Path(str(run_row["run_dir"]))
        image_key = str(run_row["image"])
        compute = str(run_row["compute"])
        presets = container.podman.list_presets()
        if image_key not in presets:
            print(f"orchestrator: image '{image_key}' not found", file=sys.stderr)
            sys.exit(1)

        phase_rows = (
            session.execute(
                sqlalchemy.select(dbmod.run_phases)
                .where(
                    (dbmod.run_phases.c.run_id == run_id) & (dbmod.run_phases.c.attempt == attempt)
                )
                .order_by(dbmod.run_phases.c.phase_order)
            )
            .mappings()
            .fetchall()
        )

    attempt_dir = run_dir / "attempt" / str(attempt)

    # Held for this process's whole life, and released by the kernel when it
    # ends however it ends. reconcile reads a free lock as proof the
    # orchestrator died, so letting go early would let a reader finalize a run
    # that is still going. Underscore-prefixed because the name exists only to
    # keep the file open -- closing it releases the lock.
    _orchestrator_lock = lock.hold(attempt_dir)

    phases_to_run = [str(r["phase"]) for r in phase_rows]
    phase_orders = {str(r["phase"]): int(r["phase_order"]) for r in phase_rows}

    describe_output = container.podman.describe(container.podman.image_ref(image_key))
    cacheable_phases = {p.name for p in describe_output.phases if p.cacheable}
    store_dir = settings.data_dir / "store"
    store_dir.mkdir(parents=True, exist_ok=True)

    for phase in phases_to_run:
        if shutting_down:
            break

        print(f"orchestrator: starting phase '{phase}'")
        sys.stdout.flush()

        now = time.time()
        data_dir = attempt_dir / "data" / phase
        cache_hit = False
        with sqlalchemy.orm.Session(engine) as session:
            _init_phase_data(run_id, attempt, phase, phase_orders[phase], run_dir, session)
            if phase in cacheable_phases:
                manifest = _check_cache(image_key, attempt_dir, phase, data_dir)
                if manifest is not None:
                    cache_hit = _try_serve_from_cache(manifest, data_dir, store_dir)
            values: dict[str, object] = {
                "status": "done" if cache_hit else "running",
                "started_at": now,
            }
            if cache_hit:
                values["ended_at"] = now
            session.execute(
                sqlalchemy.update(dbmod.run_phases)
                .where(
                    (dbmod.run_phases.c.run_id == run_id)
                    & (dbmod.run_phases.c.attempt == attempt)
                    & (dbmod.run_phases.c.phase == phase)
                )
                .values(**values)
            )
            session.commit()

        if cache_hit:
            print(f"orchestrator: phase '{phase}' served from cache")
            sys.stdout.flush()
            continue

        current_proc = _start_phase(image_key, attempt_dir, phase, compute)
        exit_code = current_proc.wait()
        current_proc = None

        now = time.time()
        if shutting_down or exit_code != 0:
            phase_status = "stopped" if shutting_down else "failed"
            with sqlalchemy.orm.Session(engine) as session:
                session.execute(
                    sqlalchemy.update(dbmod.run_phases)
                    .where(
                        (dbmod.run_phases.c.run_id == run_id)
                        & (dbmod.run_phases.c.attempt == attempt)
                        & (dbmod.run_phases.c.phase == phase)
                    )
                    .values(status=phase_status, ended_at=now)
                )
                # Mark remaining phases stopped
                remaining_idx = phases_to_run.index(phase) + 1
                for remaining_phase in phases_to_run[remaining_idx:]:
                    session.execute(
                        sqlalchemy.update(dbmod.run_phases)
                        .where(
                            (dbmod.run_phases.c.run_id == run_id)
                            & (dbmod.run_phases.c.attempt == attempt)
                            & (dbmod.run_phases.c.phase == remaining_phase)
                        )
                        .values(status="stopped", ended_at=now)
                    )
                final_status = "stopped" if shutting_down else "failed"
                session.execute(
                    sqlalchemy.update(dbmod.run_attempts)
                    .where(
                        (dbmod.run_attempts.c.run_id == run_id)
                        & (dbmod.run_attempts.c.attempt == attempt)
                    )
                    .values(status=final_status, ended_at=now)
                )
                session.execute(
                    sqlalchemy.update(dbmod.runs)
                    .where(dbmod.runs.c.id == run_id)
                    .values(status=final_status)
                )
                session.commit()
            sys.exit(1 if exit_code != 0 else 0)

        with sqlalchemy.orm.Session(engine) as session:
            session.execute(
                sqlalchemy.update(dbmod.run_phases)
                .where(
                    (dbmod.run_phases.c.run_id == run_id)
                    & (dbmod.run_phases.c.attempt == attempt)
                    & (dbmod.run_phases.c.phase == phase)
                )
                .values(status="done", ended_at=now)
            )
            session.commit()

        print(f"orchestrator: phase '{phase}' done")
        sys.stdout.flush()

    # All phases complete. Deduplicate into the store *before* marking the run
    # done: reconcile keeps the run non-terminal while this orchestrator is alive,
    # so `run show --wait` only returns once the store is fully populated.
    if not shutting_down:
        _deduplicate_data(attempt_dir, phases_to_run, settings)

        now = time.time()
        with sqlalchemy.orm.Session(engine) as session:
            session.execute(
                sqlalchemy.update(dbmod.run_attempts)
                .where(
                    (dbmod.run_attempts.c.run_id == run_id)
                    & (dbmod.run_attempts.c.attempt == attempt)
                )
                .values(status="done", ended_at=now)
            )
            session.execute(
                sqlalchemy.update(dbmod.runs).where(dbmod.runs.c.id == run_id).values(status="done")
            )
            session.commit()
        print("orchestrator: all phases done")


def _deduplicate_data(
    attempt_dir: pathlib.Path,
    phases: list[str],
    settings: config.Settings,
) -> None:
    import hashlib

    store_dir = settings.data_dir / "store"
    store_dir.mkdir(parents=True, exist_ok=True)

    for phase in phases:
        data_dir = attempt_dir / "data" / phase
        if not data_dir.exists():
            continue
        for fpath in data_dir.rglob("*"):
            if not fpath.is_file():
                continue
            content = fpath.read_bytes()
            sha = hashlib.sha256(content).hexdigest()
            store_path = store_dir / sha
            if not store_path.exists():
                store_path.write_bytes(content)
                os.chmod(store_path, 0o444)
            # Replace with hardlink
            fpath.unlink()
            os.link(store_path, fpath)
