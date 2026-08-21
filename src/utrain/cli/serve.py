"""`utrain run chat`: an interactive session against a run's trained model.

Speaks the container contract's `serve` endpoint, which is the OpenAI
`/v1/chat/completions` API rather than anything of utrain's own devising -- so
the same container is equally usable from the `openai` SDK, `curl`, or an
off-the-shelf chat UI, with utrain out of the loop entirely.

Two consequences of that choice shape this module. The API is **stateless**, so
the conversation lives here, in `_Chat.messages`, and is resent whole each turn.
And the container picks its own port (`--port 0`), publishing it to a file in the
writable `serve` mount once it is listening; `_wait_for_port` polls for that file
and turns it into a connection.
"""

import http.client
import importlib
import json
import pathlib
import subprocess
import sys
import time
import typing

import sqlalchemy
import sqlalchemy.orm

from .. import container
from . import db as dbmod
from . import exceptions, orchestrator, reconcile

# How long to wait for the container to publish its port. Generous, because it
# covers importing torch and loading a checkpoint -- tens of seconds is normal.
# It is a backstop rather than the usual exit: a container that dies is detected
# straight away, so this only bites on one that hangs while still alive.
_PORT_TIMEOUT_S = 300
_PORT_POLL_S = 0.05

_HELP = """commands:
  /reset   forget the conversation so far
  /quit    end the session (or Ctrl-D)
  /help    this message
anything else is sent to the model as the next turn."""


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
        raise exceptions.UI(
            f"abort: run '{run_id}' has no completed phase, so there is no model to serve"
        )
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
            raise exceptions.UI(
                f"abort: container exited ({proc.returncode}) before it published a port"
            )
        if time.monotonic() > deadline:
            raise exceptions.UI(
                f"abort: container did not publish a port within {_PORT_TIMEOUT_S}s ({path})"
            )
        time.sleep(_PORT_POLL_S)


class _Chat:
    """The conversation, and the HTTP calls that advance it.

    `messages` is the whole history in OpenAI's format. The server keeps none of
    it -- a stateless API means every turn resends everything, which is also what
    makes editing or retrying an earlier turn possible at all.
    """

    def __init__(self, port: int, max_tokens: int, temperature: float) -> None:
        self.port = port
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.messages: list[dict[str, str]] = []

    def _connect(self) -> http.client.HTTPConnection:
        return http.client.HTTPConnection("127.0.0.1", self.port, timeout=600)

    def model_info(self) -> dict[str, object]:
        """`GET /v1/models`, for the banner. Never fatal -- it is decoration."""
        try:
            conn = self._connect()
            conn.request("GET", "/v1/models")
            data = json.loads(conn.getresponse().read())
            conn.close()
            entries = data.get("data") or []
            return entries[0] if entries else {}
        except Exception:
            return {}

    def send(self, text: str) -> None:
        """Send one turn and stream the reply to stdout as it arrives."""
        self.messages.append({"role": "user", "content": text})
        body = json.dumps(
            {
                "model": "utrain",
                "messages": self.messages,
                "stream": True,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }
        )
        conn = self._connect()
        # Accumulated by _consume_sse rather than returned, so a Ctrl-C part way
        # through still leaves the caller holding what did arrive.
        parts: list[str] = []
        try:
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=body,
                headers={"Content-Type": "application/json"},
            )
            response = conn.getresponse()
            if response.status != 200:
                detail = _error_message(response.read())
                print(f"error: {detail}")
                # Drop the turn that failed, so the next one is not sent with a
                # user message the model never answered.
                self.messages.pop()
                return
            self._consume_sse(response, parts)
        except KeyboardInterrupt:
            # Ctrl-C mid-reply. Closing the connection is how cancellation is
            # expressed over HTTP; the container sees the broken pipe and stops
            # generating. Whatever streamed so far is kept as the model's turn,
            # so the conversation stays coherent.
            print()
            print("(cancelled)")
        finally:
            conn.close()
        self.messages.append({"role": "assistant", "content": "".join(parts)})

    def _consume_sse(self, response: http.client.HTTPResponse, parts: list[str]) -> None:
        """Read `text/event-stream` chunks, printing deltas and collecting them."""
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue  # blank separators and any comment lines
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            try:
                choices = json.loads(data)["choices"]
                delta = choices[0]["delta"].get("content")
            except Exception:
                continue
            if delta:
                parts.append(delta)
                sys.stdout.write(delta)
                sys.stdout.flush()
        print()

    def reset(self) -> None:
        self.messages.clear()


def _error_message(body: bytes) -> str:
    """Pull a message out of OpenAI's error envelope, falling back to raw text."""
    try:
        payload = json.loads(body)
        message = payload["error"]["message"]
        return str(message)
    except Exception:
        return body.decode("utf-8", "replace").strip() or "unknown error"


def _repl(chat: _Chat, interactive: bool) -> None:
    if interactive:
        try:
            # Imported for its side effect only: importing readline is what gives
            # input() line editing and history. Via import_module so it does not
            # read as an unused import.
            importlib.import_module("readline")
        except ImportError:
            pass
    # No prompt when driven from a pipe: it would interleave with the replies and
    # leave a dangling "> " at EOF, which makes scripted use awkward to read and
    # to test.
    prompt = "> " if interactive else ""
    while True:
        try:
            line = input(prompt)
        except EOFError:
            if interactive:
                print()
            return
        except KeyboardInterrupt:
            print()
            continue
        line = line.strip()
        if not line:
            continue
        if line in ("/quit", "/exit"):
            return
        if line == "/reset":
            chat.reset()
            print("(conversation reset)")
            continue
        if line == "/help":
            print(_HELP)
            continue
        chat.send(line)


def chat_run(
    run_id_prefix: str,
    session: sqlalchemy.orm.Session,
    *,
    max_tokens: int,
    temperature: float,
) -> None:
    run_id = dbmod.resolve_run_id(run_id_prefix, session)
    row = dbmod.get_run(run_id, session)
    attempt_n = dbmod.latest_attempt(run_id, session)
    if attempt_n is not None:
        reconcile.reconcile_attempt(run_id, attempt_n, session)

    image_key = str(row["image"])
    presets = container.podman.list_presets()
    if image_key not in presets:
        raise exceptions.UI(f"abort: image '{image_key}' not found")
    describe = container.podman.describe(presets[image_key])
    # The first real consumer of can_serve: until now the flag was parsed into
    # DescribeOutput and never read.
    if not describe.can_serve:
        raise exceptions.UI(f"abort: image '{image_key}' does not support serve")

    run_dir = pathlib.Path(str(row["run_dir"]))
    attempt_dir, data_dir, phase = _model_location(run_id, run_dir, session)
    if not data_dir.exists():
        raise exceptions.UI(f"abort: data dir for phase '{phase}' is missing: {data_dir}")

    # Release the SQLite write lock before the REPL: a session lasts as long as
    # the user keeps typing, and the detached orchestrator of any other run needs
    # to write meanwhile.
    session.commit()

    argv, env = orchestrator.serve_argv(image_key, attempt_dir, data_dir, str(row["compute"]))
    logs_dir = attempt_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "serve.log"

    # Both streams straight to the log, no pipes. The container's output is
    # nobody's protocol -- the port arrives through a file instead -- so there is
    # no buffer for utrain to have to keep draining, and a container is free to
    # log as noisily as it likes without wedging the session.
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(
            argv,
            stdout=log,
            stderr=log,
            env=env,
            # Its own process group, so a Ctrl-C at the terminal is delivered
            # only to us. Otherwise the signal would reach podman too and kill
            # the server, when what Ctrl-C means here is "stop this reply".
            start_new_session=True,
        )
        try:
            port = _wait_for_port(proc, orchestrator.port_file(attempt_dir))
        except exceptions.UI:
            _report_log(proc, log_path)
            raise

        chat = _Chat(port, max_tokens, temperature)
        interactive = sys.stdin.isatty()
        print(f"serving {row['name']} ({image_key}, phase '{phase}')")
        print(f"endpoint: http://127.0.0.1:{port}/v1  (OpenAI-compatible)")
        info = chat.model_info()
        if info:
            shown = {k: v for k, v in sorted(info.items()) if k not in ("object", "created")}
            print(f"model: {', '.join(f'{k}={v}' for k, v in shown.items())}")
        if interactive:
            print(_HELP)
        try:
            _repl(chat, interactive)
        finally:
            # Distinguish "the server died on us" from "we stopped it because
            # the session ended". Shutting it down sets a non-zero returncode
            # either way, so the check has to happen first.
            crashed = proc.poll() is not None
            _shutdown(proc)
        if crashed:
            _report_log(proc, log_path)


def _shutdown(proc: subprocess.Popen[typing.Any]) -> None:
    """Stop the server. `podman run` proxies SIGTERM through to the container."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _report_log(proc: subprocess.Popen[typing.Any], log_path: pathlib.Path) -> None:
    """Surface the tail of the container's log, which is otherwise unseen."""
    _shutdown(proc)
    print(f"container exited {proc.returncode}; log in {log_path}", file=sys.stderr)
    try:
        tail = log_path.read_text().splitlines()[-20:]
    except OSError:
        return
    for line in tail:
        print(f"  {line}", file=sys.stderr)
