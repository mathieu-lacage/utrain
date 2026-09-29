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

from . import container, dispatcher, exceptions, logs, orchestrator, types
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
        cid_file: pathlib.Path,
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
        self._cid_file = cid_file

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
        # Killing the client above stops nothing on its own, and even a clean
        # exit is only evidence about the client. Removing by id is what makes
        # "the server is stopped" true of the container as well.
        orchestrator.force_remove_container(self._cid_file)
        self._log.close()


def servable_phases(describe: container.schema.DescribeOutput) -> list[str]:
    """The phases of an image whose snapshot is worth chatting with.

    Only a phase that says so is one, so an image that names none serves
    nothing and `utrain run chat` refuses it outright.
    """
    return [phase.name for phase in describe.phases if phase.can_serve]


def start(
    run_id_prefix: str,
    session: sqlalchemy.orm.Session,
    phase: str | None = None,
    attempt: int | None = None,
) -> Server:
    """Spawn a serve container for one phase's snapshot of a finished run.

    `phase` names which snapshot to talk to; the default is the newest servable
    phase that completed, which for a run of one model is the only one there
    is. `attempt` narrows both to one attempt of the run, as in the address
    `RUN/ATTEMPT/PHASE`; the default is the newest attempt that ran the phase.

    Returns as soon as the process exists; call `Server.wait_for_port` to wait
    for it to become reachable.
    """
    run_id = dbmod.resolve_run_id(run_id_prefix, session)
    row = dbmod.get_run(run_id, session)
    if attempt is not None:
        found = session.execute(
            sqlalchemy.select(dbmod.run_attempts.c.attempt).where(
                (dbmod.run_attempts.c.run_id == run_id) & (dbmod.run_attempts.c.attempt == attempt)
            )
        ).scalar_one_or_none()
        if found is None:
            raise exceptions.UI(f"attempt {attempt} of run '{run_id}' not found")
    dispatcher.ensure(session)

    # What the image can do comes first: it is a fact about the image, true
    # whatever state the run is in, so an image that will never serve should say
    # so rather than send the caller off to wait for a run to finish. Asked of
    # the run's frozen image id, so a re-tagged name cannot change the answer.
    image_key = str(row["image"])
    image = dbmod.run_image_ref(row)
    servable = servable_phases(dbmod.run_description(row, session))
    if not servable:
        raise exceptions.UI(f"image '{image_key}' does not support serve")
    if phase is not None and phase not in servable:
        offered = ", ".join(servable)
        raise exceptions.UI(
            f"image '{image_key}' does not serve phase '{phase}'; it serves: {offered}"
        )

    # Re-read the status: the row fetched above predates the commit in
    # `dispatcher.ensure`, so it may be stale. Waiting for the run to end is
    # about the GPU -- the phase's own data has been final since it ended.
    status = str(dbmod.get_run(run_id, session)["status"])
    if status not in types.TERMINAL:
        raise exceptions.UI(
            f"run '{run_id}' is still {status}; chat is available once the run has finished"
        )

    run_dir = dbmod.run_dir(run_id, session)
    attempt_dir, data_dir, phase = _model_location(
        run_id, run_dir, servable, phase, attempt, session
    )
    if not data_dir.exists():
        raise exceptions.UI(f"data dir for phase '{phase}' is missing: {data_dir}")

    # Release the SQLite write lock before returning: a chat session lasts as
    # long as the user keeps typing, and the detached orchestrator of any other
    # run needs to write meanwhile.
    session.commit()

    argv, env = orchestrator.serve_argv(image, attempt_dir, data_dir, phase, str(row["compute"]))
    logs_dir = attempt_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{phase}_serve.log"

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
        port_file=orchestrator.port_file(attempt_dir, phase),
        cid_file=orchestrator.cid_file(attempt_dir, phase),
        run_name=str(row["name"]),
        image=image_key,
        phase=phase,
    )


def _model_location(
    run_id: str,
    run_dir: pathlib.Path,
    servable: list[str],
    phase: str | None,
    attempt: int | None,
    session: sqlalchemy.orm.Session,
) -> tuple[pathlib.Path, pathlib.Path, str]:
    """Locate the data dir holding one phase's model: attempt_dir, data_dir, phase.

    Each phase's data dir starts as a hardlinked copy of its predecessor's and
    is never written again once the phase ends, so every one of them is a
    complete snapshot of the run as it stood at that point. `phase` picks which;
    the default is the newest servable phase that completed.

    Either way the attempt is the highest one that ran the chosen phase, not the
    run's newest: a `--from-phase` restart leaves the phases before it in the
    attempt that did run them, and `_init_phase_data` reads them from there too.
    `attempt`, when given, pins that choice to one attempt instead.
    """
    where = (dbmod.run_phases.c.run_id == run_id) & (dbmod.run_phases.c.status == "done")
    if phase is not None:
        where = where & (dbmod.run_phases.c.phase == phase)
    else:
        where = where & dbmod.run_phases.c.phase.in_(servable)
    if attempt is not None:
        where = where & (dbmod.run_phases.c.attempt == attempt)
    row = (
        session.execute(
            sqlalchemy.select(dbmod.run_phases.c.phase, dbmod.run_phases.c.attempt)
            .where(where)
            .order_by(dbmod.run_phases.c.phase_order.desc(), dbmod.run_phases.c.attempt.desc())
            .limit(1)
        )
        .mappings()
        .fetchone()
    )
    if row is None:
        if phase is not None and attempt is not None:
            raise exceptions.UI(
                f"phase '{phase}' of attempt {attempt} of run '{run_id}' has not completed"
            )
        if phase is not None:
            raise exceptions.UI(f"phase '{phase}' of run '{run_id}' has not completed")
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
