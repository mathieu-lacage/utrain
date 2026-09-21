"""Where a phase's output lives, and what the readers do about the two layouts.

Phases write stdout and stderr into one merged file (see
`orchestrator._start_phase`); runs from before the merge kept the two streams
in separate files, and the readers fall back to their stdout. These tests pin
both halves: the path rule, the read through the query layer for each layout,
and the capture contract itself -- one file object for both fds, and the env
var that keeps the interleaving honest.

None of this needs podman: the capture test monkeypatches `Popen`, so unlike
the cram suite it runs in CI.
"""

import pathlib
import types

import pytest
import sqlalchemy

import utrain.config
import utrain.db
import utrain.logs
import utrain.orchestrator
import utrain.runs

RUN_ID = "a" * 32


def test_phase_log_path_prefers_the_merged_file(tmp_path: pathlib.Path) -> None:
    """A run captured since the merge is read from its one output file."""
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "pretrain_output.log").write_text("merged\n")
    (logs / "pretrain_stdout.log").write_text("legacy\n")
    assert utrain.logs.phase_log_path(tmp_path, "pretrain") == logs / "pretrain_output.log"


def test_phase_log_path_falls_back_to_legacy_stdout(tmp_path: pathlib.Path) -> None:
    """A pre-merge run keeps being read from the stdout file it has."""
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "pretrain_stdout.log").write_text("legacy\n")
    assert utrain.logs.phase_log_path(tmp_path, "pretrain") == logs / "pretrain_stdout.log"


def test_phase_log_path_when_nothing_was_written(tmp_path: pathlib.Path) -> None:
    """A pending phase has neither file; the caller's not-found names stdout's."""
    (tmp_path / "logs").mkdir()
    assert (
        utrain.logs.phase_log_path(tmp_path, "pretrain")
        == tmp_path / "logs" / "pretrain_stdout.log"
    )


def _seed_run(data_dir: pathlib.Path) -> pathlib.Path:
    """A finished run with one attempt, in the shape the query layer expects."""
    run_dir = data_dir / "runs" / RUN_ID
    attempt_dir = run_dir / "attempt" / "1"
    (attempt_dir / "logs").mkdir(parents=True)
    with utrain.db.with_db(utrain.config.Settings(data_dir=data_dir)) as session:
        session.execute(
            sqlalchemy.insert(utrain.db.runs).values(
                id=RUN_ID,
                name="hello",
                image="utrain-fake",
                compute="cpu",
                status="done",
                config_hash=None,
                created_at=100.0,
            )
        )
        session.execute(
            sqlalchemy.insert(utrain.db.run_attempts).values(
                run_id=RUN_ID,
                attempt=1,
                from_phase=None,
                status="done",
                pid=None,
                started_at=100.0,
                ended_at=200.0,
            )
        )
    return attempt_dir


def test_read_log_tail_reads_the_merged_stream(tmp_path: pathlib.Path) -> None:
    """What the container put on stderr is in the tail, where it was emitted."""
    attempt_dir = _seed_run(tmp_path)
    (attempt_dir / "logs" / "pretrain_output.log").write_text(
        "step 1\nTraceback (most recent call last)\n"
    )
    with utrain.db.with_db(utrain.config.Settings(data_dir=tmp_path)) as session:
        lines = utrain.runs.read_log_tail(RUN_ID, session, phase="pretrain")
    assert lines == ["step 1", "Traceback (most recent call last)"]


def test_read_log_tail_falls_back_for_a_legacy_run(tmp_path: pathlib.Path) -> None:
    attempt_dir = _seed_run(tmp_path)
    (attempt_dir / "logs" / "pretrain_stdout.log").write_text("counting\nvocab built\n")
    with utrain.db.with_db(utrain.config.Settings(data_dir=tmp_path)) as session:
        lines = utrain.runs.read_log_tail(RUN_ID, session, phase="pretrain")
    assert lines == ["counting", "vocab built"]


class _FakePopen:
    """Records how the phase container was launched, and launches nothing."""

    def __init__(self, argv: list[str], **kwargs: object) -> None:
        self.argv = argv
        self.kwargs = kwargs


def test_start_phase_writes_one_merged_stream(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both fds are the same open file, and the container is told not to buffer.

    Sharing the file object is what makes the interleaving faithful -- the
    kernel serialises writes to it -- and `PYTHONUNBUFFERED` is what stops the
    container's block-buffered stdout from letting stderr jump ahead of the
    stdout it belongs with.

    The name inside `orchestrator` is replaced rather than `subprocess.Popen`
    itself: that module is shared with everything else in the process, and
    conftest's fixtures shell out through it on their way out.
    """
    monkeypatch.setattr(utrain.orchestrator, "subprocess", types.SimpleNamespace(Popen=_FakePopen))
    attempt_dir = tmp_path / "attempt" / "1"

    proc = utrain.orchestrator._start_phase("utrain-fake", attempt_dir, "pretrain", "cpu")

    assert isinstance(proc, _FakePopen)
    stdout = proc.kwargs["stdout"]
    stderr = proc.kwargs["stderr"]
    assert stdout is stderr
    assert getattr(stdout, "name") == str(attempt_dir / "logs" / "pretrain_output.log")
    # `open` created it, so a phase that prints nothing still leaves a log.
    assert (attempt_dir / "logs" / "pretrain_output.log").exists()
    env = proc.kwargs["env"]
    assert isinstance(env, dict)
    assert env["PYTHONUNBUFFERED"] == "1"
    # The file object is the real code's to keep -- the container outlives the
    # call -- but the fake never starts one, so close it here.
    getattr(stdout, "close")()
