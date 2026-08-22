"""Bringing a run's trained model up as a local server.

Speaks the container contract's `serve` endpoint, which is the OpenAI
`/v1/chat/completions` API rather than anything of utrain's own devising -- so
the same container is equally usable from the `openai` SDK, `curl`, or an
off-the-shelf chat UI, with utrain out of the loop entirely.

The container picks its own port (`--port 0`) and publishes it to a file in the
writable `serve` mount once it is listening, so starting one is two steps:
`start` spawns it and returns immediately, `Server.wait_for_port` blocks until
it is reachable. They are separate so a caller can report a failed startup in
its own way -- this module never writes to a terminal.
"""

import json
import pathlib
import subprocess
import time
import typing

import sqlalchemy
import sqlalchemy.orm

from . import container, exceptions, logs, orchestrator, reconcile
from . import db as dbmod

# How long to wait for the container to publish its port. Generous, because it
# covers importing torch and loading a checkpoint -- tens of seconds is normal.
# It is a backstop rather than the usual exit: a container that dies is detected
# straight away, so this only bites on one that hangs while still alive.
_PORT_TIMEOUT_S = 300
_PORT_POLL_S = 0.05


class Server:
    """A `serve` container that has been spawned but may not yet be listening."""

    def __init__(
        self,
        proc: subprocess.Popen[bytes],
        log: typing.IO[bytes],
        log_path: pathlib.Path,
        port_file: pathlib.Path,
        run_name: str,
        image: str,
        phase: str,
    ) -> None:
        self.proc = proc
        self.log_path = log_path
        self.run_name = run_name
        self.image = image
        self.phase = phase
        self._log = log
        self._port_file = port_file

    @property
    def returncode(self) -> int | None:
        return self.proc.returncode

    def crashed(self) -> bool:
        """Whether the container is already gone, i.e. it died rather than us stopping it."""
        return self.proc.poll() is not None

    def wait_for_port(self) -> int:
        return _wait_for_port(self.proc, self._port_file)

    def log_tail(self, n: int = 20) -> list[str]:
        try:
            return logs.tail_lines(self.log_path, n)
        except OSError:
            return []

    def shutdown(self) -> None:
        """Stop the server. `podman run` proxies SIGTERM through to the container."""
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self._log.close()


def start(run_id_prefix: str, session: sqlalchemy.orm.Session) -> Server:
    """Spawn a serve container for a run's newest completed phase.

    Returns as soon as the process exists; call `Server.wait_for_port` to wait
    for it to become reachable.
    """
    run_id = dbmod.resolve_run_id(run_id_prefix, session)
    row = dbmod.get_run(run_id, session)
    attempt_n = dbmod.latest_attempt(run_id, session)
    if attempt_n is not None:
        reconcile.reconcile_attempt(run_id, attempt_n, session)

    image_key = str(row["image"])
    presets = container.podman.list_presets()
    if image_key not in presets:
        raise exceptions.UI(f"image '{image_key}' not found")
    describe = container.podman.describe(presets[image_key])
    # The first real consumer of can_serve: until now the flag was parsed into
    # DescribeOutput and never read.
    if not describe.can_serve:
        raise exceptions.UI(f"image '{image_key}' does not support serve")

    run_dir = pathlib.Path(str(row["run_dir"]))
    attempt_dir, data_dir, phase = _model_location(run_id, run_dir, session)
    if not data_dir.exists():
        raise exceptions.UI(f"data dir for phase '{phase}' is missing: {data_dir}")

    # Release the SQLite write lock before returning: a chat session lasts as
    # long as the user keeps typing, and the detached orchestrator of any other
    # run needs to write meanwhile.
    session.commit()

    argv, env = orchestrator.serve_argv(image_key, attempt_dir, data_dir, str(row["compute"]))
    logs_dir = attempt_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "serve.log"

    # Both streams straight to the log, no pipes. The container's output is
    # nobody's protocol -- the port arrives through a file instead -- so there is
    # no buffer for utrain to have to keep draining, and a container is free to
    # log as noisily as it likes without wedging the session.
    log = open(log_path, "wb")
    proc = subprocess.Popen(
        argv,
        stdout=log,
        stderr=log,
        env=env,
        # Its own process group, so a Ctrl-C at the terminal is delivered only
        # to the caller. Otherwise the signal would reach podman too and kill
        # the server, when what Ctrl-C means there is "stop this reply".
        start_new_session=True,
    )
    return Server(
        proc=proc,
        log=log,
        log_path=log_path,
        port_file=orchestrator.port_file(attempt_dir),
        run_name=str(row["name"]),
        image=image_key,
        phase=phase,
    )


def _model_location(
    run_id: str,
    run_dir: pathlib.Path,
    session: sqlalchemy.orm.Session,
) -> tuple[pathlib.Path, pathlib.Path, str]:
    """Locate the data dir holding the trained model: attempt_dir, data_dir, phase.

    Data dirs are per-phase, and each one starts as a hardlinked copy of its
    predecessor's, so the *last* completed phase's dir is the only one holding
    everything the run produced -- that is where the checkpoint is.

    Ordering by phase_order then attempt mirrors _init_phase_data: after a
    `--from-phase` restart the newest attempt may not have re-run the later
    phases, so the highest-numbered phase can live in an older attempt.
    """
    row = (
        session.execute(
            sqlalchemy.select(dbmod.run_phases.c.phase, dbmod.run_phases.c.attempt)
            .where((dbmod.run_phases.c.run_id == run_id) & (dbmod.run_phases.c.status == "done"))
            .order_by(dbmod.run_phases.c.phase_order.desc(), dbmod.run_phases.c.attempt.desc())
            .limit(1)
        )
        .mappings()
        .fetchone()
    )
    if row is None:
        raise exceptions.UI(f"run '{run_id}' has no completed phase, so there is no model to serve")
    attempt_dir = run_dir / "attempt" / str(row["attempt"])
    return attempt_dir, attempt_dir / "data" / str(row["phase"]), str(row["phase"])


def _wait_for_port(proc: subprocess.Popen[bytes], path: pathlib.Path) -> int:
    """Poll `path` until the container publishes the port it bound.

    The file appearing *is* the readiness signal: the contract has the container
    write it only once its socket is listening. Polling rather than reading a
    pipe keeps the container's stdout and stderr free to be ordinary logs, which
    is worth more than the few lines of loop it costs here.

    Two things make the loop honest rather than a fixed sleep. A container that
    dies is noticed immediately, instead of waiting out the timeout. And a
    partially written file simply fails to parse and is retried, so there is no
    torn-read window even though the container also writes it atomically.
    """
    deadline = time.monotonic() + _PORT_TIMEOUT_S
    while True:
        try:
            port = json.loads(path.read_text())["port"]
            if not isinstance(port, int):
                raise ValueError(f"non-integer port: {port!r}")
            return port
        except (OSError, ValueError, KeyError, TypeError):
            pass
        if proc.poll() is not None:
            raise exceptions.UI(f"container exited ({proc.returncode}) before it published a port")
        if time.monotonic() > deadline:
            raise exceptions.UI(
                f"container did not publish a port within {_PORT_TIMEOUT_S}s ({path})"
            )
        time.sleep(_PORT_POLL_S)
