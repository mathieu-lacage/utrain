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

from .. import config, container
from . import db as dbmod

# Single dir placed first on the container's PYTHONPATH; holds the mounted naw
# package and the wandb shim (see _wandb_mount_args).
_WANDB_PYPATH = "/opt/utrain-py"

# Where the attempt dir is bind-mounted inside the container. Deliberately not
# `/run`: podman populates a container's /run with its own bind mounts (a
# `secrets` dir from /usr/share/containers/mounts.conf, plus `nvidia-ctk-hook*`
# and `nvidia-persistenced` when a GPU device is attached), and those land in
# the attempt dir on the host. Images take the run dir as a positional argument,
# so the path is theirs to receive, not to hardcode.
_RUN_MOUNT = "/utrain"


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
    run_dir: pathlib.Path,
    compute: str,
) -> subprocess.Popen[bytes]:
    logs_dir = attempt_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout = open(logs_dir / f"{phase}_stdout.log", "wb")
    stderr = open(logs_dir / f"{phase}_stderr.log", "wb")

    data_dir = attempt_dir / "data" / phase
    data_dir.mkdir(parents=True, exist_ok=True)
    (attempt_dir / "wandb").mkdir(parents=True, exist_ok=True)

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
            "-v",
            f"{attempt_dir}:{_RUN_MOUNT}",
            "-v",
            f"{data_dir}:/data",
            container.podman.image_ref(image_key),
            "run",
            _RUN_MOUNT,
            "--phase",
            phase,
        ],
        stdout=stdout,
        stderr=stderr,
        env=env,
    )


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
    phases_to_run = [str(r["phase"]) for r in phase_rows]
    phase_orders = {str(r["phase"]): int(r["phase_order"]) for r in phase_rows}

    for phase in phases_to_run:
        if shutting_down:
            break

        print(f"orchestrator: starting phase '{phase}'")
        sys.stdout.flush()

        now = time.time()
        with sqlalchemy.orm.Session(engine) as session:
            _init_phase_data(run_id, attempt, phase, phase_orders[phase], run_dir, session)
            session.execute(
                sqlalchemy.update(dbmod.run_phases)
                .where(
                    (dbmod.run_phases.c.run_id == run_id)
                    & (dbmod.run_phases.c.attempt == attempt)
                    & (dbmod.run_phases.c.phase == phase)
                )
                .values(status="running", started_at=now)
            )
            session.commit()

        current_proc = _start_phase(image_key, attempt_dir, phase, run_dir, compute)
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
