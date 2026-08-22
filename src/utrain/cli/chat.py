"""The interactive `utrain run chat` session.

The server side lives in `utrain.serve`; everything here is terminal-facing.

The OpenAI chat API is stateless, so the conversation lives in `_Chat.messages`
and is resent whole each turn.
"""

import http.client
import importlib
import json
import sys

import sqlalchemy.orm

from .. import exceptions, serve

_HELP = """commands:
  /reset   forget the conversation so far
  /quit    end the session (or Ctrl-D)
  /help    this message
anything else is sent to the model as the next turn."""


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
    server = serve.start(run_id_prefix, session)
    try:
        port = server.wait_for_port()
    except exceptions.UI:
        _report_log(server)
        raise

    chat = _Chat(port, max_tokens, temperature)
    interactive = sys.stdin.isatty()
    print(f"serving {server.run_name} ({server.image}, phase '{server.phase}')")
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
        # Distinguish "the server died on us" from "we stopped it because the
        # session ended". Shutting it down sets a non-zero returncode either
        # way, so the check has to happen first.
        crashed = server.crashed()
        server.shutdown()
    if crashed:
        _report_log(server)


def _report_log(server: serve.Server) -> None:
    """Surface the tail of the container's log, which is otherwise unseen."""
    server.shutdown()
    print(f"container exited {server.returncode}; log in {server.log_path}", file=sys.stderr)
    for line in server.log_tail(20):
        print(f"  {line}", file=sys.stderr)
