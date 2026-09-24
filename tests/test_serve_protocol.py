"""The serve endpoint, container and client halves, without podman.

`tests/cram/serve.t` covers `utrain run chat` end to end, but it needs podman and
so skips in CI. This drives the same client code (`utrain.serve`) against the
same reference container (`tests/containers/fake/fake.py`), just spawned directly
instead of inside an image -- which is exactly what the contract's "a container
runs from a checkout too" rule makes possible. It runs everywhere.
"""

import collections.abc
import http.client
import http.server
import io
import json
import os
import pathlib
import subprocess
import sys
import threading
import typing
import uuid

import pytest
import sqlalchemy
import sqlalchemy.orm

import utrain.chat
import utrain.cli.chat
import utrain.config
import utrain.container.podman
import utrain.container.schema
import utrain.db
import utrain.exceptions
import utrain.orchestrator
import utrain.serve

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent
_FAKE = _PROJECT_ROOT / "tests" / "containers" / "fake" / "fake.py"
# The wandb shim, so fake.py's module-level `import wandb` resolves without the
# real package installed -- the same substitution utrain does inside a container.
_SHIM = _PROJECT_ROOT / "src" / "utrain" / "container" / "wandb_shim"


@pytest.fixture()
def served(
    tmp_path: pathlib.Path,
    spawn_serve: collections.abc.Callable[[pathlib.Path], subprocess.Popen[bytes]],
) -> int:
    """A running fake container; the port it published.

    The container is spawned directly rather than through podman, which the
    contract's "a container runs from a checkout too" rule makes possible and
    which is what lets this file run in CI. `spawn_serve` lives in conftest
    because the TUI's chat tests need the same container.
    """
    return utrain.serve._wait_for_port(spawn_serve(tmp_path), tmp_path / "serve" / "port.json")


def _post(port: int, path: str, payload: dict[str, object]) -> http.client.HTTPResponse:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    conn.request(
        "POST", path, body=json.dumps(payload), headers={"Content-Type": "application/json"}
    )
    return conn.getresponse()


def test_port_file_appears_only_once_the_port_is_usable(
    served: int, tmp_path: pathlib.Path
) -> None:
    """The port file's appearance is a readiness signal, not just an address.

    The container writes it after `listen()`, so a client may connect the
    instant it reads the file -- no retry loop on the connection itself. That
    only holds because bind-then-publish would leave a window where the kernel
    refuses connections; this asserts the container does not have that window.
    """
    conn = http.client.HTTPConnection("127.0.0.1", served, timeout=30)
    conn.request("GET", "/v1/models")
    assert conn.getresponse().status == 200

    # And it is where the contract says it is, holding what the contract says.
    published = json.loads((tmp_path / "serve" / "port.json").read_text())
    assert published["port"] == served
    # The temporary file the atomic rename went through must not survive.
    assert list((tmp_path / "serve").iterdir()) == [tmp_path / "serve" / "port.json"]


def test_models_endpoint_reports_the_data_dir(served: int, tmp_path: pathlib.Path) -> None:
    conn = http.client.HTTPConnection("127.0.0.1", served, timeout=30)
    conn.request("GET", "/v1/models")
    data = json.loads(conn.getresponse().read())
    assert data["data"][0]["id"] == "fake"
    # Proves the filesystem contract reached `serve`.
    assert data["data"][0]["data_dir"] == str(tmp_path / "data")


def test_non_streaming_completion(served: int) -> None:
    response = _post(
        served, "/v1/chat/completions", {"messages": [{"role": "user", "content": "hello"}]}
    )
    assert response.status == 200
    body = json.loads(response.read())
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "[fake model] turn 1: 'hello'"


def test_streaming_completion_is_sse_and_terminates(served: int) -> None:
    response = _post(
        served,
        "/v1/chat/completions",
        {"messages": [{"role": "user", "content": "hello"}], "stream": True},
    )
    assert response.status == 200
    # startswith, not ==: Starlette appends "; charset=utf-8", which is a valid
    # media type and which SSE clients handle.
    content_type = response.getheader("Content-Type") or ""
    assert content_type.startswith("text/event-stream")
    raw = response.read().decode()

    chunks = [line[len("data:") :].strip() for line in raw.splitlines() if line.startswith("data:")]
    assert chunks[-1] == "[DONE]", "the stream must end with OpenAI's [DONE] sentinel"
    parsed = [json.loads(c) for c in chunks[:-1]]
    assert all(c["object"] == "chat.completion.chunk" for c in parsed)
    assert parsed[-1]["choices"][0]["finish_reason"] == "stop"
    # More than one content chunk: it really streams rather than arriving whole.
    content = [c["choices"][0]["delta"].get("content") for c in parsed]
    assert len([c for c in content if c]) > 1
    assert "".join(c for c in content if c) == "[fake model] turn 1: 'hello'"


def test_history_is_carried_by_the_client_not_the_server(served: int) -> None:
    """The API is stateless: turn 2 is turn 2 only because the client said so.

    This is the property that makes the KV-cache a prefix-match rather than an
    append -- the server is told the whole conversation every time and has to
    work out for itself what it has already seen.
    """
    first = json.loads(
        _post(
            served, "/v1/chat/completions", {"messages": [{"role": "user", "content": "one"}]}
        ).read()
    )
    assert "turn 1" in first["choices"][0]["message"]["content"]

    resent = json.loads(
        _post(
            served,
            "/v1/chat/completions",
            {
                "messages": [
                    {"role": "user", "content": "one"},
                    {"role": "assistant", "content": first["choices"][0]["message"]["content"]},
                    {"role": "user", "content": "two"},
                ]
            },
        ).read()
    )
    assert "turn 2" in resent["choices"][0]["message"]["content"]

    # A fresh single-message request is turn 1 again: nothing was retained.
    again = json.loads(
        _post(
            served, "/v1/chat/completions", {"messages": [{"role": "user", "content": "three"}]}
        ).read()
    )
    assert "turn 1" in again["choices"][0]["message"]["content"]


def test_unknown_path_is_404(served: int) -> None:
    response = _post(served, "/v1/nonsense", {})
    assert response.status == 404


def test_openapi_schema_is_published(served: int) -> None:
    """A container built on FastAPI describes its own endpoint, for free."""
    conn = http.client.HTTPConnection("127.0.0.1", served, timeout=30)
    conn.request("GET", "/openapi.json")
    schema = json.loads(conn.getresponse().read())
    conn.close()
    assert "/v1/chat/completions" in schema["paths"]
    assert "/v1/models" in schema["paths"]
    assert "ChatCompletionRequest" in schema["components"]["schemas"]


def test_empty_messages_is_rejected(served: int) -> None:
    """`messages` is required to be non-empty, and the rejection uses OpenAI's
    envelope rather than FastAPI's default 422 `{"detail": ...}`."""
    response = _post(served, "/v1/chat/completions", {"messages": []})
    assert response.status == 400
    assert "messages" in json.loads(response.read())["error"]["message"]


# The client half, `utrain.chat`, which prints nothing and which both the CLI's
# REPL and the TUI's chat screen are built on.


def test_client_streams_a_reply_in_pieces(served: int) -> None:
    client = utrain.chat.Client(served, max_tokens=10, temperature=0.8)
    deltas = list(client.stream("hello"))
    assert len(deltas) > 1, "the reply should arrive in pieces, not whole"
    assert "".join(deltas) == "[fake model] turn 1: 'hello'"


def test_client_carries_the_history_the_server_does_not(served: int) -> None:
    """Turn 2 is turn 2 only because the client resent turn 1."""
    client = utrain.chat.Client(served, max_tokens=10, temperature=0.8)
    assert "".join(client.stream("one")) == "[fake model] turn 1: 'one'"
    assert "".join(client.stream("two")) == "[fake model] turn 2: 'two'"
    assert [(m.role, m.content) for m in client.messages] == [
        ("user", "one"),
        ("assistant", "[fake model] turn 1: 'one'"),
        ("user", "two"),
        ("assistant", "[fake model] turn 2: 'two'"),
    ]


def test_client_reset_starts_the_conversation_over(served: int) -> None:
    client = utrain.chat.Client(served, max_tokens=10, temperature=0.8)
    list(client.stream("one"))
    client.reset()
    assert not client.messages
    assert "".join(client.stream("two")) == "[fake model] turn 1: 'two'"


def test_client_abandoned_part_way_keeps_what_arrived(served: int) -> None:
    """What a Ctrl-C at the REPL and an escape in the TUI both come down to.

    Closing the generator closes the connection, which is how a stop is said
    over HTTP; the partial reply is still the model's turn, so the next one is
    not sent with a user message that has no answer.
    """
    client = utrain.chat.Client(served, max_tokens=10, temperature=0.8)
    stream = client.stream("hello")
    first = next(stream)
    stream.close()
    assert [m.role for m in client.messages] == ["user", "assistant"]
    assert client.messages[-1].content == first
    assert first and not client.messages[-1].content.endswith("'hello'")


@pytest.fixture()
def refusing() -> typing.Iterator[int]:
    """A server that answers every turn with OpenAI's error envelope.

    The reference container refuses only an empty `messages`, which a `Client`
    cannot send -- it always has the turn it was just given. So the refusal
    path gets a server of its own rather than a contrived request.
    """

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.dumps({"error": {"message": "context length exceeded"}}).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: typing.Any) -> None:
            """Silence: the default writes a line to stderr per request."""

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        thread.join(timeout=10)


def test_client_raises_on_a_refused_turn_and_drops_it(refusing: int) -> None:
    """The message comes out of OpenAI's envelope, and the turn is not kept.

    Dropping it matters: the next turn would otherwise be sent with a user
    message the model never answered.
    """
    client = utrain.chat.Client(refusing, max_tokens=10, temperature=0.8)
    with pytest.raises(utrain.exceptions.UI, match="context length exceeded"):
        list(client.stream("hello"))
    assert not client.messages


def test_client_model_info_is_never_fatal() -> None:
    """It decorates a banner; a port with nothing on it is not a session ender."""
    assert utrain.chat.Client(1, max_tokens=10, temperature=0.8).model_info() == {}


# `utrain run chat` itself. Same trick as above -- the container is spawned
# directly rather than through podman -- so everything from resolving the run to
# driving the REPL is covered where `tests/cram/serve.t` cannot run.


def _make_run(
    session: sqlalchemy.orm.Session,
    tmp_path: pathlib.Path,
    *,
    phase_statuses: dict[str, str],
    run_status: str = "done",
) -> str:
    run_id = uuid.uuid4().hex
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True)
    session.execute(
        sqlalchemy.insert(utrain.db.runs).values(
            id=run_id,
            name="chat",
            image="utrain-fake",
            image_id="f" * 64,
            compute="cpu",
            status=run_status,
            created_at=0.0,
        )
    )
    session.execute(
        sqlalchemy.insert(utrain.db.run_attempts).values(
            run_id=run_id, attempt=1, status="done", started_at=0.0
        )
    )
    for order, (phase, status) in enumerate(phase_statuses.items()):
        session.execute(
            sqlalchemy.insert(utrain.db.run_phases).values(
                run_id=run_id, attempt=1, phase=phase, phase_order=order, status=status
            )
        )
        if status == "done":
            (run_dir / "attempt" / "1" / "data" / phase).mkdir(parents=True)
    return run_id


def _patch_describe(monkeypatch: pytest.MonkeyPatch, *, can_serve: bool) -> None:
    """The image the chat tests talk to: two phases, one of them chattable.

    `tokenizer` is deliberately left unservable, so a run of this image has a
    completed phase that chat must still refuse to serve.
    """
    described = utrain.container.schema.DescribeOutput(
        name="fake",
        phases=[
            utrain.container.schema.PhaseInfo(name="tokenizer", label="Tokenizer"),
            utrain.container.schema.PhaseInfo(
                name="pretrain", label="Pretrain", can_serve=can_serve
            ),
        ],
        phase_order=["tokenizer", "pretrain"],
    )
    monkeypatch.setattr(utrain.container.podman, "describe", lambda ref: described)


@pytest.fixture()
def chat_env(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> typing.Iterator[sqlalchemy.orm.Session]:
    """A DB with utrain's podman call replaced by a directly-spawned fake."""
    monkeypatch.setenv("UTRAIN_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        utrain.container.podman,
        "list_presets",
        lambda: {"utrain-fake": "localhost/utrain-fake:utrain"},
    )
    # Resolving a run's image checks the frozen id against the local store;
    # answer yes rather than shelling out to podman.
    monkeypatch.setattr(utrain.container.podman, "image_exists", lambda ref: True)
    _patch_describe(monkeypatch, can_serve=True)

    def fake_serve_argv(
        image_key: str,
        attempt_dir: pathlib.Path,
        data_dir: pathlib.Path,
        phase: str,
        compute: str,
    ) -> tuple[list[str], dict[str, str]]:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(_SHIM), env.get("PYTHONPATH", "")])
        # Stand in for the two bind mounts the real serve_argv sets up. They come
        # from unrelated host directories -- data_dir is a *phase's* data dir,
        # while the serve dir hangs off the attempt -- so a single root cannot
        # contain both; symlinks reproduce that faithfully. Getting this wrong
        # means the container publishes its port somewhere utrain never looks.
        serve_host_dir = utrain.orchestrator.serve_dir(attempt_dir, phase)
        serve_host_dir.mkdir(parents=True, exist_ok=True)
        root = attempt_dir / "mount-root"
        root.mkdir(parents=True, exist_ok=True)
        for name, target in (("data", data_dir), ("serve", serve_host_dir)):
            link = root / name
            if not link.exists():
                link.symlink_to(target)
        return [
            sys.executable,
            str(_FAKE),
            "--utrain-root",
            str(root),
            "serve",
            "--port",
            "0",
            "--phase",
            phase,
        ], env

    monkeypatch.setattr(utrain.orchestrator, "serve_argv", fake_serve_argv)

    with utrain.db.with_db(utrain.config.Settings()) as session:
        yield session


def _feed(monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(text))


def test_chat_run_streams_turns(
    chat_env: sqlalchemy.orm.Session,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A full session, and proof a chatty container cannot wedge one.

    The fake serves with uvicorn's access log on, so this runs against a
    container emitting a line of output per request. That is survivable because
    the container's output is not a protocol -- the port arrives through a file
    -- so utrain never reads these streams and never has a pipe to keep drained.
    """
    run_id = _make_run(chat_env, tmp_path, phase_statuses={"tokenizer": "done", "pretrain": "done"})
    _feed(monkeypatch, "hello\nagain\n/quit\n")
    utrain.cli.chat.chat_run(run_id, chat_env, max_tokens=10, temperature=0.8)
    lines = capsys.readouterr().out.splitlines()
    # Exact shape, because tests/cram/serve.t asserts the same output verbatim.
    # With no --phase, the newest servable completed phase is what answers.
    assert lines[0] == "serving chat (utrain-fake, phase 'pretrain')"
    assert lines[1].startswith("endpoint: http://127.0.0.1:")
    assert lines[2].startswith("model: data_dir=")
    # turn 2 proves the client resent the history, since the server keeps none.
    assert lines[3:] == [
        "[fake model] turn 1: 'hello'",
        "[fake model] turn 2: 'again'",
    ]


def test_chat_run_reset_clears_the_client_side_history(
    chat_env: sqlalchemy.orm.Session,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_id = _make_run(chat_env, tmp_path, phase_statuses={"pretrain": "done"})
    _feed(monkeypatch, "one\n/reset\ntwo\n/quit\n")
    utrain.cli.chat.chat_run(run_id, chat_env, max_tokens=10, temperature=0.8)
    out = capsys.readouterr().out
    assert "[fake model] turn 1: 'one'" in out
    assert "[fake model] turn 1: 'two'" in out


def test_chat_run_reports_a_container_that_fails_to_start(
    chat_env: sqlalchemy.orm.Session,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_id = _make_run(chat_env, tmp_path, phase_statuses={"pretrain": "done"})

    def broken_argv(
        image_key: str,
        attempt_dir: pathlib.Path,
        data_dir: pathlib.Path,
        phase: str,
        compute: str,
    ) -> tuple[list[str], dict[str, str]]:
        return [sys.executable, "-c", "raise SystemExit('no model here')"], dict(os.environ)

    monkeypatch.setattr(utrain.orchestrator, "serve_argv", broken_argv)
    _feed(monkeypatch, "")
    with pytest.raises(utrain.exceptions.UI, match="before it published a port"):
        utrain.cli.chat.chat_run(run_id, chat_env, max_tokens=10, temperature=0.8)
    # The container's stderr is otherwise only in a log file, so it gets echoed.
    assert "no model here" in capsys.readouterr().err


def test_chat_run_refuses_a_run_with_no_completed_phase(
    chat_env: sqlalchemy.orm.Session, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = _make_run(chat_env, tmp_path, phase_statuses={"pretrain": "pending"})
    _feed(monkeypatch, "")
    with pytest.raises(utrain.exceptions.UI, match="no completed phase"):
        utrain.cli.chat.chat_run(run_id, chat_env, max_tokens=10, temperature=0.8)


def test_chat_run_refuses_an_image_that_cannot_serve(
    chat_env: sqlalchemy.orm.Session, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = _make_run(chat_env, tmp_path, phase_statuses={"pretrain": "done"})
    _patch_describe(monkeypatch, can_serve=False)
    _feed(monkeypatch, "")
    with pytest.raises(utrain.exceptions.UI, match="does not support serve"):
        utrain.cli.chat.chat_run(run_id, chat_env, max_tokens=10, temperature=0.8)


def test_chat_run_serves_the_phase_it_is_asked_for(
    chat_env: sqlalchemy.orm.Session,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--phase` reaches back past the newest one.

    Each phase's data dir is frozen when the phase ends, so an earlier one is a
    snapshot of the run as it stood then -- which is the whole point of being
    able to ask for it. The banner names which one answered.
    """
    _patch_describe(monkeypatch, can_serve=True)
    # Both phases servable here, so the choice is the caller's rather than
    # the only one on offer.
    described = utrain.container.schema.DescribeOutput(
        name="fake",
        phases=[
            utrain.container.schema.PhaseInfo(name="tokenizer", label="Tokenizer", can_serve=True),
            utrain.container.schema.PhaseInfo(name="pretrain", label="Pretrain", can_serve=True),
        ],
        phase_order=["tokenizer", "pretrain"],
    )
    monkeypatch.setattr(utrain.container.podman, "describe", lambda ref: described)

    run_id = _make_run(chat_env, tmp_path, phase_statuses={"tokenizer": "done", "pretrain": "done"})
    _feed(monkeypatch, "hello\n/quit\n")
    utrain.cli.chat.chat_run(run_id, chat_env, phase="tokenizer", max_tokens=10, temperature=0.8)
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "serving chat (utrain-fake, phase 'tokenizer')"
    # The mount followed the choice, not just the label: the harness stands in
    # for the bind mount with a symlink, so its target is what was served.
    mount = tmp_path / "runs" / run_id / "attempt" / "1" / "mount-root" / "data"
    assert mount.resolve().name == "tokenizer"
    # And the container was told, rather than left to work it out from the
    # mount: the fake echoes the `--phase` it was started with.
    assert lines[2].endswith("phase=tokenizer")


def test_chat_run_refuses_a_phase_the_image_does_not_serve(
    chat_env: sqlalchemy.orm.Session, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A phase that leaves no model behind is not on offer, even though it ran."""
    run_id = _make_run(chat_env, tmp_path, phase_statuses={"tokenizer": "done", "pretrain": "done"})
    _feed(monkeypatch, "")
    with pytest.raises(utrain.exceptions.UI, match="does not serve phase 'tokenizer'"):
        utrain.cli.chat.chat_run(
            run_id, chat_env, phase="tokenizer", max_tokens=10, temperature=0.8
        )


def test_chat_run_refuses_a_phase_that_has_not_completed(
    chat_env: sqlalchemy.orm.Session, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = _make_run(
        chat_env, tmp_path, phase_statuses={"tokenizer": "done", "pretrain": "failed"}
    )
    _feed(monkeypatch, "")
    with pytest.raises(utrain.exceptions.UI, match="phase 'pretrain' .* has not completed"):
        utrain.cli.chat.chat_run(run_id, chat_env, phase="pretrain", max_tokens=10, temperature=0.8)


def test_chat_run_refuses_a_run_that_is_still_going(
    chat_env: sqlalchemy.orm.Session, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Chat waits for the run to finish, even though the phase's data is ready.

    One GPU per run: a serve container started now would be competing with the
    training one for it.
    """
    run_id = _make_run(
        chat_env,
        tmp_path,
        phase_statuses={"tokenizer": "done", "pretrain": "running"},
        run_status="running",
    )
    _feed(monkeypatch, "")
    with pytest.raises(utrain.exceptions.UI, match="is still running"):
        utrain.cli.chat.chat_run(run_id, chat_env, max_tokens=10, temperature=0.8)
