"""Talking to a served model, without a terminal in sight.

`serve` brings the container up; this drives the OpenAI
`/v1/chat/completions` API it exposes. Nothing here prints: the CLI's REPL and
the TUI's chat screen are both clients of it, and they display a reply in ways
that have nothing in common.

The API is stateless, so the conversation lives in `Client.messages` and is
resent whole each turn. That is also what makes editing or retrying an earlier
turn possible at all.
"""

import collections.abc
import http.client
import json

from . import exceptions, types

# Long, because a turn is a model generating tokens one at a time on whatever
# hardware the run was trained on, and a slow reply is not a broken one.
_TIMEOUT_S = 600


class Client:
    """One conversation with one served model."""

    def __init__(self, port: int, max_tokens: int, temperature: float) -> None:
        self.port = port
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.messages: list[types.ChatMessage] = []

    def _connect(self) -> http.client.HTTPConnection:
        return http.client.HTTPConnection("127.0.0.1", self.port, timeout=_TIMEOUT_S)

    def model_info(self) -> dict[str, object]:
        """`GET /v1/models`, for a banner. Never fatal -- it is decoration."""
        try:
            conn = self._connect()
            conn.request("GET", "/v1/models")
            data = json.loads(conn.getresponse().read())
            conn.close()
            entries = data.get("data") or []
            first = entries[0] if entries else {}
            return first if isinstance(first, dict) else {}
        except Exception:
            return {}

    def stream(self, text: str) -> collections.abc.Iterator[str]:
        """Send one turn, yielding the reply in the pieces it arrives in.

        A generator rather than a callback, because that is what makes
        cancellation the caller's to express: abandoning it -- a Ctrl-C at the
        REPL, an escape in the TUI -- closes the connection, which is how a
        stop is said over HTTP, and the container sees the broken pipe and
        stops generating. Whatever streamed before that is still recorded as
        the model's turn, so the conversation stays coherent either way.

        Raises `exceptions.UI` if the server refuses the turn, having first
        dropped it from the history so the next one is not sent with a user
        message the model never answered.
        """
        self.messages.append(types.ChatMessage(role="user", content=text))
        body = json.dumps(
            {
                "model": "utrain",
                "messages": [{"role": m.role, "content": m.content} for m in self.messages],
                "stream": True,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }
        )
        conn = self._connect()
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
                self.messages.pop()
                raise exceptions.UI(detail)
            yield from self._record(_deltas(response))
        finally:
            conn.close()

    def _record(self, deltas: collections.abc.Iterator[str]) -> collections.abc.Iterator[str]:
        """Pass deltas through, keeping what went past as the assistant's turn.

        The append is in a `finally` so that it happens on the ways out that
        are not "the reply finished": a closed generator, or a connection that
        broke part way.
        """
        parts: list[str] = []
        try:
            for delta in deltas:
                parts.append(delta)
                yield delta
        finally:
            self.messages.append(types.ChatMessage(role="assistant", content="".join(parts)))

    def reset(self) -> None:
        self.messages.clear()


def _deltas(response: http.client.HTTPResponse) -> collections.abc.Iterator[str]:
    """The content of each `text/event-stream` chunk, in order."""
    for raw in response:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue  # blank separators and any comment lines
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            return
        try:
            choices = json.loads(data)["choices"]
            delta = choices[0]["delta"].get("content")
        except Exception:
            continue
        if delta:
            yield delta


def _error_message(body: bytes) -> str:
    """Pull a message out of OpenAI's error envelope, falling back to raw text."""
    try:
        payload = json.loads(body)
        message = payload["error"]["message"]
        return str(message)
    except Exception:
        return body.decode("utf-8", "replace").strip() or "unknown error"
