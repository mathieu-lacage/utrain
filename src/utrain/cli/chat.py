"""The interactive `utrain run chat` session.

Everything here is terminal-facing: the server side is `utrain.serve` and the
conversation itself is `utrain.chat`, neither of which prints. What is left is
a prompt, deltas written as they arrive, and the handful of slash commands.
"""

import importlib
import sys

import sqlalchemy.orm

from .. import chat, exceptions, serve

_HELP = """commands:
  /reset   forget the conversation so far
  /quit    end the session (or Ctrl-D)
  /help    this message
anything else is sent to the model as the next turn."""


def _send(client: chat.Client, text: str) -> None:
    """One turn, streamed to stdout as it arrives.

    Ctrl-C part way through closes the generator, which closes the connection;
    `chat.Client` keeps what did arrive as the model's turn, so the history is
    the same as what was on screen.
    """
    try:
        for delta in client.stream(text):
            sys.stdout.write(delta)
            sys.stdout.flush()
        print()
    except KeyboardInterrupt:
        print()
        print("(cancelled)")
    except exceptions.UI as e:
        print(f"error: {e}")


def _repl(client: chat.Client, interactive: bool) -> None:
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
            client.reset()
            print("(conversation reset)")
            continue
        if line == "/help":
            print(_HELP)
            continue
        _send(client, line)


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

    client = chat.Client(port, max_tokens, temperature)
    interactive = sys.stdin.isatty()
    print(f"serving {server.run_name} ({server.image}, phase '{server.phase}')")
    print(f"endpoint: http://127.0.0.1:{port}/v1  (OpenAI-compatible)")
    info = client.model_info()
    if info:
        shown = {k: v for k, v in sorted(info.items()) if k not in ("object", "created")}
        print(f"model: {', '.join(f'{k}={v}' for k, v in shown.items())}")
    if interactive:
        print(_HELP)
    try:
        _repl(client, interactive)
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
