#!/usr/bin/env python3
"""Record the asciinema tour of the TUI that `docs/tour.md` embeds.

    make tour                      # re-record docs/assets/tour.cast
    make tour ARGS="--no-build"    # skip rebuilding the fixture image

What is recorded is the real `utrain tui`, driven by real keystrokes through a
pty, against a real run of `tests/containers/nanochat-fake` -- an image that
serves nanochat's own `describe.py` but fakes the training, so the five phases
are over in half a minute instead of a day. Nothing about the UI is stubbed:
the orchestrator, the metrics files, the serve container and the streamed chat
reply are all the ones a user gets. That is the point. A hand-authored cast
would go stale the first time a pane moved, and nobody would notice.

Recording needs podman, and a GPU unless `--compute cpu` is passed. CI has
neither, so `docs/assets/tour.cast` is a committed artifact and this is run by
hand when the TUI changes.

The output is asciicast v2 written directly -- a header line and one JSON array
per event -- because the `asciinema` binary is not a dependency of anything
else here and its `rec` would only wrap this same pty in a subshell.
"""

import argparse
import codecs
import collections.abc
import contextlib
import dataclasses
import fcntl
import json
import os
import pathlib
import pty
import re
import select
import shutil
import struct
import subprocess
import sys
import termios
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
CONTAINERFILE = ROOT / "tests" / "containers" / "nanochat-fake" / "Containerfile"
DEFAULT_OUT = ROOT / "docs" / "assets" / "tour.cast"

# Wide enough for the sidebar plus a plot that is worth looking at, short enough
# that the player does not shrink the text to nothing on a phone. Every cell is
# a cell the cast carries on every repaint, so this is also the biggest single
# lever on its size.
COLS, ROWS = 100, 30

# Presets are podman tags, so they are shared with whatever the person
# recording has in their own store. `utrain image add` namespaces by
# UTRAIN_IMAGE_TAG -- the trick `tests/conftest.py` uses to isolate tests --
# which is what keeps their images out of the images pane, and lets the fixture
# be called `nanochat` without colliding with the real preset of that name.
IMAGE_TAG = "utrain-tour"
FIXTURE_REF = "localhost/nanochat-fake:utrain"
PRESET_REF = f"localhost/nanochat:{IMAGE_TAG}"
PRESET = "utrain-nanochat"

RUN_NAME = "nanochat-d6"

# The machine the tour appears to run on. Pinned rather than read from the host
# so the compute pane and the new-run dialog say the same thing wherever this
# is recorded; `compute.collect_compute` reads this file when the env var
# points at it.
COMPUTE = {
    "cpu": {
        "name": "AMD Ryzen 9 7940HS",
        "cores": 16,
        "mem_total_gb": 31.0,
        "mem_available_gb": 22.6,
    },
    "gpus": [
        {
            "index": 0,
            "name": "NVIDIA RTX 2000 Ada",
            "power_draw": 9.5,
            "power_limit": 140.0,
            "util": 8,
            "mem_used_mb": 100.0,
            "mem_total_mb": 8192.0,
        }
    ],
}

# Dead air longer than this is cut back to it. A tour is not obliged to sit
# through a container start, and the viewer cannot tell how long one took.
MAX_GAP = 1.0

# Writes closer together than this are one frame as far as a viewer is
# concerned, and are written to the cast as one event.
FRAME = 0.04

KEYS = {
    "enter": "\r",
    "escape": "\x1b",
    "tab": "\t",
    "up": "\x1b[A",
    "down": "\x1b[B",
    "space": " ",
}


class TourError(Exception):
    """A beat did not happen -- almost always the UI moved under the script."""


@dataclasses.dataclass
class Beat:
    """One chapter: the events it produced, and how long it gets on screen."""

    label: str
    start: int
    at: float
    budget: float | None
    hold: float
    end: int = 0


class Recorder:
    """A pty with `utrain tui` in it, and every byte it wrote, timed."""

    def __init__(self, argv: list[str], env: dict[str, str]) -> None:
        self.master, slave = pty.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))
        # Its own session, so a stray signal in this process group does not
        # reach it and the pty is its controlling terminal.
        self.proc = subprocess.Popen(
            argv, stdin=slave, stdout=slave, stderr=slave, env=env, start_new_session=True
        )
        os.close(slave)
        self.events: list[tuple[float, str]] = []
        self.beats: list[Beat] = []
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._carry = ""
        # ANSI stripped, for `expect`. Not a screen model: this is the stream,
        # so it can only be asked about text that has just been written. Every
        # `expect` below follows a keypress that repaints, which is what makes
        # that enough; anything positional would need a real emulator.
        self.text = ""
        self._mark = 0

    def pump(self, timeout: float) -> bool:
        """Read once. False when the child has closed the pty."""
        try:
            ready, _, _ = select.select([self.master], [], [], timeout)
        except OSError:
            return False
        if not ready:
            return True
        try:
            chunk = os.read(self.master, 65536)
        except OSError:
            return False
        if not chunk:
            return False
        text = self._decoder.decode(chunk)
        self.events.append((time.monotonic(), text))
        self.text += self._strip(text)
        return True

    def _strip(self, text: str) -> str:
        """Drop the escape sequences, across reads.

        A read lands wherever the kernel buffer ended, so an escape sequence is
        regularly cut in half by one -- which used to leave the second half in
        `text` as if it were something the app had printed. Anything after a
        trailing incomplete `ESC` is carried into the next read instead.
        """
        pending = self._carry + text
        start = pending.rfind("\x1b")
        if start != -1 and _ANSI.match(pending, start) is None:
            self._carry, pending = pending[start:], pending[:start]
        else:
            self._carry = ""
        return _ANSI.sub("", pending)

    def wait(self, seconds: float) -> None:
        """Let the screen run for a while, recording what it does."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if not self.pump(min(0.2, max(0.01, deadline - time.monotonic()))):
                return

    def expect(self, pattern: str, timeout: float = 30.0) -> None:
        """Wait for the screen to say something, then let the next key go."""
        compiled = re.compile(pattern)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if compiled.search(self.text, self._mark):
                return
            if not self.pump(0.1):
                break
        # The screen as it was, because "the UI moved" is the only thing
        # this ever means and the next question is always "moved to what".
        raise TourError(f"timed out waiting for {pattern!r}; screen was:\n{self.screen()}")

    def screen(self) -> str:
        """The tail of what was written, for an error message to quote."""
        return "\n".join(line for line in self.text[-2500:].splitlines() if line.strip())

    def send(self, *keys: str) -> None:
        """Press keys. A name in KEYS is that key; anything else is typed."""
        for key in keys:
            os.write(self.master, KEYS.get(key, key).encode())
            self.wait(0.12)
        self._mark = len(self.text)

    def type(self, text: str) -> None:
        """Type at a human rate, because a name that appears at once reads as a paste."""
        for char in text:
            os.write(self.master, char.encode())
            self.wait(0.06)
        self._mark = len(self.text)

    @contextlib.contextmanager
    def beat(
        self, label: str, budget: float | None = None, hold: float = 1.2
    ) -> collections.abc.Generator[None]:
        """A chapter of the tour, marked in the cast's timeline.

        `hold` is the pause after it, so a viewer can read what just happened;
        `budget` caps it, scaling the whole chapter down if what really
        happened took longer than the tour can spend on it.
        """
        entry = Beat(label, len(self.events), time.monotonic(), budget, hold)
        self.beats.append(entry)
        yield
        self.wait(0.4)
        entry.end = len(self.events)

    def finish(self) -> None:
        """Drain what is left and make sure the child is gone."""
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and self.proc.poll() is None:
            if not self.pump(0.2):
                break
        if self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=10)
        os.close(self.master)


_ANSI = re.compile(
    r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[=>()][0-9A-Za-z]?"
)


def cast(rec: Recorder) -> str:
    """The recording as asciicast v2, with the timeline each beat asked for.

    Timestamps are rewritten rather than replayed: the real recording is a
    33-second fake training run with container starts in the middle of it, and
    what the tour wants is each beat legible and none of the dead air. Order is
    preserved and nothing is dropped -- only the gaps between events move.
    """
    lines = [
        json.dumps(
            {
                "version": 2,
                "width": COLS,
                "height": ROWS,
                "timestamp": int(time.time()),
                "title": "utrain tui",
                "env": {"TERM": "xterm-256color", "SHELL": "/bin/bash"},
            }
        )
    ]
    cursor = 0.0
    for beat in rec.beats:
        lines.append(dump([round(cursor, 3), "m", beat.label]))
        events = coalesce(rec.events[beat.start : beat.end])
        stamps: list[float] = []
        elapsed, previous = 0.0, beat.at
        for at, _ in events:
            elapsed += min(at - previous, MAX_GAP)
            stamps.append(elapsed)
            previous = at
        if beat.budget is not None and elapsed > beat.budget:
            scale = beat.budget / elapsed
            stamps = [stamp * scale for stamp in stamps]
            elapsed = beat.budget
        for stamp, (_, data) in zip(stamps, events):
            lines.append(dump([round(cursor + stamp, 3), "o", data]))
        cursor += elapsed + beat.hold
    return "\n".join(lines) + "\n"


def dump(event: list[float | str]) -> str:
    """One asciicast line.

    `ensure_ascii=False` because the plots are drawn in half-block glyphs, and
    escaping each of them as `\\uXXXX` doubles the size of every frame that has
    a curve in it for no gain -- the file is UTF-8 either way.
    """
    return json.dumps(event, ensure_ascii=False)


def coalesce(events: list[tuple[float, str]]) -> list[tuple[float, str]]:
    """Join writes that landed within a frame of each other.

    A repaint reaches the pty in however many chunks the kernel felt like, and
    each one costs another timestamped array in the file. Concatenating the
    ones nobody could see apart is lossless -- the bytes and their order are
    unchanged -- and takes a large bite out of the event count.
    """
    merged: list[tuple[float, str]] = []
    for at, data in events:
        if merged and at - merged[-1][0] < FRAME:
            merged[-1] = (merged[-1][0], merged[-1][1] + data)
        else:
            merged.append((at, data))
    return merged


# -- the tour -------------------------------------------------------------


def storyboard(rec: Recorder, gpu: bool) -> None:
    """What the viewer is shown, in order.

    Written as calls rather than as a table because the beats genuinely differ
    -- one types a name, one waits for a phase to reach `running`, one waits
    for a whole run to end -- and a table with an escape hatch per row reads
    worse than this does. The keys are the ones `_HELP` in
    `src/utrain/tui/screens.py` lists, which is the map this has to agree with.
    """
    with rec.beat("the runs you have", hold=2.0):
        rec.expect("runs", timeout=60)

    with rec.beat("images", hold=2.2):
        rec.send("i")
        rec.expect(re.escape(PRESET))

    with rec.beat("compute", hold=2.4):
        rec.send("c")
        rec.expect("RTX 2000 Ada")

    with rec.beat("a new run", hold=1.6):
        rec.send("escape")
        rec.expect("runs")
        rec.send("n")
        rec.expect("new run")
        rec.type(RUN_NAME)
        # name -> image -> compute. One image, so the picker is passed over;
        # the compute one is opened, because choosing the GPU is the choice.
        # `--cpu` records the same beat on a machine that has no GPU to pick.
        rec.send("tab", "tab", "enter")
        rec.expect("gpu0")
        if gpu:
            rec.send("down")
        rec.send("enter")
        rec.wait(1.0)
        rec.send("tab", "enter")
        rec.expect(re.escape(RUN_NAME))

    with rec.beat("the config utrain wrote for it", hold=3.0):
        rec.send("2")
        rec.expect("Transformer Depth")

    with rec.beat("make it smaller", hold=2.2):
        # `down` from a pane with no cursor lands on the first *config* row --
        # `ConfigPane.rows` leaves out the run's own summary lines -- which is
        # Transformer Depth, the field worth changing on a one-GPU run.
        rec.send("down", "e")
        rec.wait(0.9)
        rec.send("\x7f", "6", "enter")
        rec.expect("Transformer Depth")
        rec.wait(0.9)

    with rec.beat("start it", hold=1.6):
        rec.send("1")
        rec.send("s")
        rec.expect("running", timeout=60)

    with rec.beat("the phases it runs", budget=13.0, hold=1.5):
        rec.send("enter")
        rec.expect(r"tokenizer\s+\w")
        rec.wait(4.0)

    with rec.beat("the curves, as they fill", budget=15.0, hold=1.5):
        # The cursor lands on whichever phase was current when the list opened,
        # so it is walked to a known row rather than nudged from one: `up` stops
        # at the top of the table instead of wrapping.
        rec.expect(r"pretrain\s+running", timeout=120)
        rec.send(*["up"] * 6)
        rec.send("down", "down")
        rec.send("3")
        rec.expect("train/loss", timeout=60)
        rec.wait(5.0)

    with rec.beat("which curves are drawn", hold=2.0):
        rec.send("m")
        rec.expect("mfu")
        rec.send("down", "space")
        rec.wait(0.7)
        rec.send("y")
        rec.wait(0.7)
        rec.send("l")
        rec.wait(0.7)
        # There and back: `b` shows the half-blocks, and the second press
        # leaves the rest of the tour on the braille the plots opened in.
        rec.send("b")
        rec.wait(1.2)
        rec.send("b")
        rec.wait(0.7)
        rec.send("m")

    with rec.beat("the log", budget=7.0, hold=1.8):
        rec.send("4")
        rec.wait(4.0)

    with rec.beat("until it is done", budget=7.0, hold=1.4):
        rec.send("1")
        rec.expect(r"rl\s+done", timeout=240)
        rec.wait(1.2)

    with rec.beat("talk to what it made", budget=6.0, hold=1.5):
        rec.send(*["up"] * 6)
        rec.send("down", "down", "down")
        rec.wait(0.8)
        rec.send("t")
        rec.expect(r"phase 'sft' on http", timeout=180)

    with rec.beat("a question", hold=2.5):
        rec.type("capital of France?")
        rec.send("enter")
        rec.expect("Seine", timeout=120)
        rec.wait(1.5)

    with rec.beat("out", hold=1.0):
        rec.send("escape")
        rec.wait(1.5)
        rec.send("q")


# -- setup ----------------------------------------------------------------


def run(
    argv: list[str],
    env: dict[str, str] | None = None,
    quiet: bool = False,
) -> None:
    """Run something that has to succeed, and say which one did not."""
    result = subprocess.run(argv, cwd=ROOT, env=env, capture_output=quiet)
    if result.returncode != 0:
        raise TourError(f"failed: {' '.join(argv)}")


def build_fixture() -> None:
    run(["podman", "build", "-t", FIXTURE_REF, "-f", str(CONTAINERFILE), str(ROOT)])


def sweep() -> None:
    """Drop the tour's preset tag. Image data is shared, so this costs nothing."""
    subprocess.run(["podman", "rmi", PRESET_REF], capture_output=True)


def prepare(root: pathlib.Path) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    (root / "compute.json").write_text(json.dumps(COMPUTE))
    env = dict(os.environ)
    env.pop("COLORTERM", None)
    env.update(
        UTRAIN_DATA_DIR=str(root),
        UTRAIN_IMAGE_TAG=IMAGE_TAG,
        UTRAIN_COMPUTE_FIXTURE=str(root / "compute.json"),
        # Pinned, so the plots do not depend on what `render.default_charset`
        # guesses about the recording machine's font. Braille, which is what
        # that guess lands on for a modern terminal and what the curves look
        # best in; `docs/assets/tour.js` pins the player to a font that has it.
        UTRAIN_TUI_CHARSET="braille",
        # No COLORTERM, deliberately: rich then writes `38;5;n` instead of
        # `38;2;r;g;b`, which is most of a kilobyte off every full repaint and
        # a difference nobody can see in a recording of a TUI.
        TERM="xterm-256color",
        LINES=str(ROWS),
        COLUMNS=str(COLS),
    )
    # `nanochat`, not `nanochat-fake`: the preset a viewer would have is what
    # the tour should show, and UTRAIN_IMAGE_TAG keeps it off the real one.
    run(["podman", "tag", FIXTURE_REF, PRESET_REF])
    run(["utrain", "image", "add", f"podman://{PRESET_REF}"], env=env, quiet=True)
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument("--no-build", action="store_true", help="reuse the fixture image")
    parser.add_argument("--cpu", action="store_true", help="record on the cpu, not a gpu")
    parser.add_argument("--keep", action="store_true", help="leave the recording's data dir")
    args = parser.parse_args()

    for tool in ("podman", "utrain"):
        if shutil.which(tool) is None:
            print(f"{tool} not found", file=sys.stderr)
            return 1

    if not args.no_build:
        build_fixture()

    root = pathlib.Path(os.environ.get("TMPDIR", "/tmp")) / f"utrain-tour-{os.getpid()}"
    env = prepare(root)
    rec = Recorder([str(shutil.which("utrain")), "tui"], env)
    try:
        storyboard(rec, gpu=not args.cpu)
    finally:
        rec.finish()
        sweep()
        if not args.keep:
            shutil.rmtree(root, ignore_errors=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(cast(rec))
    size = args.out.stat().st_size
    print(f"wrote {args.out} ({size // 1024} KiB, {len(rec.beats)} beats)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
