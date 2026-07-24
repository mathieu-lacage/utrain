import os
import pathlib
import signal
import subprocess
import sys
import time

import sqlalchemy
import sqlalchemy.orm

from .. import config, container
from . import db as dbmod


def _gpu_env(compute: str) -> dict[str, str]:
    """Env that makes enroot's nvidia hook expose the selected GPU inside the container."""
    env = dict(os.environ)
    if compute.startswith("gpu"):
        env["NVIDIA_VISIBLE_DEVICES"] = compute.removeprefix("gpu")
        env["NVIDIA_DRIVER_CAPABILITIES"] = "all"
    return env


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

    return subprocess.Popen(
        [
            "enroot",
            "start",
            "--mount",
            f"{attempt_dir}:/run",
            "--mount",
            f"{data_dir}:/data",
            f"{image_key}+utrain",
            "run",
            "/run",
            "--phase",
            phase,
        ],
        stdout=stdout,
        stderr=stderr,
        env=_gpu_env(compute),
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
        presets = container.enroot.list_presets()
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

    for phase in phases_to_run:
        if shutting_down:
            break

        print(f"orchestrator: starting phase '{phase}'")
        sys.stdout.flush()

        now = time.time()
        with sqlalchemy.orm.Session(engine) as session:
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

    # All phases complete
    if not shutting_down:
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

        # Deduplicate data store
        _deduplicate_data(attempt_dir, phases_to_run, settings)


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
