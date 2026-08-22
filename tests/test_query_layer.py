"""Unit tests for the query layer.

These exist because the query layer no longer prints: `runs.list_runs` and
friends return dataclasses, so they can be asserted on directly instead of
through captured stdout. Nothing here needs podman, so unlike the cram suite
these do run in CI.
"""

import os
import pathlib
import random
import subprocess
import sys
import tempfile
import typing

import pytest
import sqlalchemy
import sqlalchemy.orm

import utrain.cli.render
import utrain.config
import utrain.db
import utrain.exceptions
import utrain.lock
import utrain.logs
import utrain.runs
import utrain.types


@pytest.fixture()
def session(tmp_path: pathlib.Path) -> typing.Iterator[sqlalchemy.orm.Session]:
    settings = utrain.config.Settings(data_dir=tmp_path)
    with utrain.db.with_db(settings) as s:
        yield s


def _insert_run(
    session: sqlalchemy.orm.Session,
    run_id: str,
    name: str,
    created_at: float,
    status: str = "configuring",
) -> None:
    session.execute(
        sqlalchemy.insert(utrain.db.runs).values(
            id=run_id,
            name=name,
            image="img",
            compute="cpu",
            run_dir=f"/runs/{run_id}",
            status=status,
            config_hash=None,
            created_at=created_at,
        )
    )


def test_list_runs_returns_rows_newest_first(session: sqlalchemy.orm.Session) -> None:
    _insert_run(session, "a" * 32, "older", created_at=100.0)
    _insert_run(session, "b" * 32, "newer", created_at=200.0)

    rows = utrain.runs.list_runs(session)

    assert [r.name for r in rows] == ["newer", "older"]
    assert all(isinstance(r, utrain.types.RunRow) for r in rows)


def test_list_runs_keeps_timestamps_and_ids_raw(session: sqlalchemy.orm.Session) -> None:
    """Formatting is the caller's job, so nothing here may be pre-rendered."""
    _insert_run(session, "c" * 32, "one", created_at=1700000000.0)

    row = utrain.runs.list_runs(session)[0]

    assert row.created_at == 1700000000.0
    assert row.id == "c" * 32


def test_run_with_no_attempts_reports_none(session: sqlalchemy.orm.Session) -> None:
    _insert_run(session, "d" * 32, "fresh", created_at=1.0)

    row = utrain.runs.list_runs(session)[0]

    assert row.attempt is None
    assert row.phase is None


def test_run_phase_is_the_first_unfinished_one(session: sqlalchemy.orm.Session) -> None:
    run_id = "e" * 32
    _insert_run(session, run_id, "busy", created_at=1.0, status="running")
    session.execute(
        sqlalchemy.insert(utrain.db.run_attempts).values(
            run_id=run_id, attempt=1, from_phase=None, status="running", pid=None, started_at=1.0
        )
    )
    for order, (phase, status) in enumerate([("prepare", "done"), ("train", "running")]):
        session.execute(
            sqlalchemy.insert(utrain.db.run_phases).values(
                run_id=run_id, attempt=1, phase=phase, phase_order=order, status=status
            )
        )

    # get_run rather than list_runs: the latter reconciles against the run dir,
    # which does not exist here, so it would correctly mark every phase stopped.
    row = utrain.runs.get_run(run_id, session)

    assert row.attempt == 1
    assert row.phase == "train"


def test_min_prefix_len_shortens_only_as_far_as_stays_unique() -> None:
    assert utrain.cli.render.min_prefix_len([]) == 1
    assert utrain.cli.render.min_prefix_len(["abc", "bcd"]) == 1
    assert utrain.cli.render.min_prefix_len(["aac", "abe"]) == 2


def test_run_table_truncates_ids_to_a_unique_prefix() -> None:
    rows = [
        utrain.types.RunRow(
            id=rid,
            name="n",
            image="i",
            compute="cpu",
            status="done",
            created_at=0.0,
            attempt=None,
            phase=None,
        )
        for rid in ("aac0", "abd1")
    ]

    table = utrain.cli.render.run_table(rows)

    assert [line.split()[0] for line in table.splitlines()[1:]] == ["aa", "ab"]


@pytest.mark.parametrize("n", [0, 1, 20, 200, 5000])
def test_tail_lines_matches_a_naive_slice(n: int) -> None:
    """`n <= 0` means everything, matching the lines[-n:] the callers used to do."""
    random.seed(n)
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "f.log"
        for trial in range(25):
            lines = [
                "".join(random.choice("abcé€\t ") for _ in range(random.randint(0, 400)))
                for _ in range(random.randint(0, 500))
            ]
            text = "\n".join(lines) + ("\n" if trial % 2 else "")
            path.write_bytes(text.encode())

            expected = path.read_text(errors="replace").splitlines()[-n:]
            assert utrain.logs.tail_lines(path, n) == expected


def _hold_lock_forever(attempt_dir: str) -> subprocess.Popen[bytes]:
    """A child process that takes the lock and then blocks until killed."""
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import pathlib, sys, utrain.lock;"
            " f = utrain.lock.hold(pathlib.Path(sys.argv[1]));"
            " print('held', flush=True);"
            " __import__('time').sleep(300)",
            attempt_dir,
        ],
        stdout=subprocess.PIPE,
    )
    assert proc.stdout is not None
    assert proc.stdout.readline().strip() == b"held"
    return proc


def test_lock_is_held_while_holder_lives_and_free_once_killed(tmp_path: pathlib.Path) -> None:
    """SIGKILL is the case reconciliation exists for, so it is the case to test."""
    proc = _hold_lock_forever(str(tmp_path))
    try:
        assert utrain.lock.is_held(tmp_path, fallback_pid=None) is True
    finally:
        proc.kill()
        proc.wait()

    # The kernel drops the lock when the process dies, with no chance for it to
    # clean up -- which is what makes a free lock proof the holder is gone.
    assert utrain.lock.is_held(tmp_path, fallback_pid=None) is False


def test_lock_refuses_a_second_holder(tmp_path: pathlib.Path) -> None:
    proc = _hold_lock_forever(str(tmp_path))
    try:
        with pytest.raises(utrain.exceptions.UI):
            utrain.lock.hold(tmp_path)
    finally:
        proc.kill()
        proc.wait()


def test_lock_probe_does_not_block_the_holder(tmp_path: pathlib.Path) -> None:
    """A read must never cost the orchestrator its lock.

    Readers take a shared lock, so several can probe at once and none of them
    excludes the exclusive holder.
    """
    proc = _hold_lock_forever(str(tmp_path))
    try:
        for _ in range(20):
            assert utrain.lock.is_held(tmp_path, fallback_pid=None) is True
    finally:
        proc.kill()
        proc.wait()


def test_lock_falls_back_to_pid_when_there_is_no_lock_file(tmp_path: pathlib.Path) -> None:
    """Attempts started before the lock existed must not read as dead."""
    assert not utrain.lock.path(tmp_path).exists()

    assert utrain.lock.is_held(tmp_path, fallback_pid=os.getpid()) is True
    assert utrain.lock.is_held(tmp_path, fallback_pid=None) is False


def test_lock_falls_back_to_pid_while_the_file_is_still_empty(tmp_path: pathlib.Path) -> None:
    """The gap between creating the lock file and locking it reads as alive."""
    utrain.lock.path(tmp_path).touch()

    assert utrain.lock.is_held(tmp_path, fallback_pid=os.getpid()) is True
