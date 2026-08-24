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
import utrain.container.schema
import utrain.db
import utrain.exceptions
import utrain.lock
import utrain.logs
import utrain.phases
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


def _described() -> utrain.container.schema.DescribeOutput:
    return utrain.container.schema.DescribeOutput(
        name="img",
        phases=[
            utrain.container.schema.PhaseInfo(name="prepare", label="Prepare"),
            utrain.container.schema.PhaseInfo(name="train", label="Train"),
        ],
        phase_order=["prepare", "train"],
        config_schema=utrain.container.schema.ConfigSchema(),
    )


def test_a_phase_a_restart_skipped_keeps_the_status_it_finished_with(
    session: sqlalchemy.orm.Session,
) -> None:
    """`restart --from-phase train` does not re-run `prepare`.

    `prepare` is not in attempt 2's rows at all, but the run is standing on its
    output, so the list has to say what came of it -- tagged with the attempt
    that earned it, not reported as unknown.
    """
    run_id = "f" * 32
    _insert_run(session, run_id, "restarted", created_at=1.0, status="running")
    for attempt, from_phase in ((1, None), (2, "train")):
        session.execute(
            sqlalchemy.insert(utrain.db.run_attempts).values(
                run_id=run_id,
                attempt=attempt,
                from_phase=from_phase,
                status="running" if attempt == 2 else "done",
                pid=None,
                started_at=float(attempt),
            )
        )
    for order, phase in enumerate(["prepare", "train"]):
        session.execute(
            sqlalchemy.insert(utrain.db.run_phases).values(
                run_id=run_id,
                attempt=1,
                phase=phase,
                phase_order=order,
                status="done",
                started_at=10.0,
                ended_at=20.0,
            )
        )
    session.execute(
        sqlalchemy.insert(utrain.db.run_phases).values(
            run_id=run_id,
            attempt=2,
            phase="train",
            phase_order=1,
            status="running",
            started_at=30.0,
            ended_at=None,
        )
    )

    entries = utrain.phases.list_phases(run_id, session, _described())

    assert [e.phase for e in entries] == ["prepare", "train"]
    prepare, train = entries
    assert prepare.status == "done"
    assert prepare.inherited_from == 1
    # The timings come with the status: a phase that ran has a duration, and
    # the one it has is attempt 1's.
    assert (prepare.started_at, prepare.ended_at) == (10.0, 20.0)
    # Addressed by the attempt that ran it, so `phase show` reaches its logs.
    assert prepare.address.endswith("/1/prepare")
    assert train.inherited_from is None


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


# -- editing a run's config -----------------------------------------------


def _config_describe() -> utrain.container.schema.DescribeOutput:
    return utrain.container.schema.DescribeOutput(
        name="img",
        phases=[utrain.container.schema.PhaseInfo(name="pretrain", label="Train")],
        phase_order=["pretrain"],
        config_schema=utrain.container.schema.ConfigSchema(
            globals=utrain.container.schema.GlobalConfigSchema(
                groups=[
                    utrain.container.schema.FieldGroup(
                        name="model",
                        label="Model",
                        fields=[
                            utrain.container.schema.FieldSchema(
                                key="n_layer", label="Layers", type="int", default=4, min=1, max=48
                            ),
                            utrain.container.schema.FieldSchema(
                                key="dtype",
                                label="Precision",
                                type="enum",
                                default="fp32",
                                options=["fp32", "bf16"],
                            ),
                        ],
                    )
                ]
            ),
            phases={
                "pretrain": utrain.container.schema.PhaseConfigSchema(
                    groups=[
                        utrain.container.schema.FieldGroup(
                            name="optim",
                            label="Optimiser",
                            fields=[
                                utrain.container.schema.FieldSchema(
                                    key="lr", label="Learning rate", type="float", default=0.001
                                )
                            ],
                        )
                    ]
                )
            },
        ),
    )


def _configurable_run(
    session: sqlalchemy.orm.Session,
    tmp_path: pathlib.Path,
    status: str = "configuring",
) -> str:
    run_id = "c" * 32
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(
        f"run_id: {run_id}\ncompute: cpu\n"
        "globals:\n  model:\n    n_layer: 4\n    dtype: fp32\n"
        "phases:\n  pretrain:\n    lr: 0.001\n"
    )
    session.execute(
        sqlalchemy.insert(utrain.db.runs).values(
            id=run_id,
            name="draft",
            image="img",
            compute="cpu",
            run_dir=str(run_dir),
            status=status,
            config_hash=None,
            created_at=100.0,
        )
    )
    return run_id


def test_read_config_parses_the_file(
    session: sqlalchemy.orm.Session, tmp_path: pathlib.Path
) -> None:
    run_id = _configurable_run(session, tmp_path)
    assert utrain.runs.read_config(run_id, session)["globals"] == {
        "model": {"n_layer": 4, "dtype": "fp32"}
    }


def test_write_config_coerces_form_strings_to_the_declared_types(
    session: sqlalchemy.orm.Session, tmp_path: pathlib.Path
) -> None:
    """A form hands back strings; the schema says what they mean."""
    run_id = _configurable_run(session, tmp_path)
    utrain.runs.write_config(
        run_id,
        {
            "globals": {"model": {"n_layer": "12", "dtype": "bf16"}},
            "phases": {"pretrain": {"lr": "3e-4"}},
        },
        session,
        _config_describe(),
    )

    written = utrain.runs.read_config(run_id, session)
    assert written["globals"] == {"model": {"n_layer": 12, "dtype": "bf16"}}
    assert written["phases"] == {"pretrain": {"lr": 0.0003}}


def test_write_config_writes_the_first_config_of_a_run_that_has_none(
    session: sqlalchemy.orm.Session, tmp_path: pathlib.Path
) -> None:
    """`read_config` tolerates a missing file, so writing one must too."""
    run_id = _configurable_run(session, tmp_path)
    path = utrain.runs.config_path(run_id, session)
    path.unlink()

    utrain.runs.write_config(
        run_id, {"globals": {"model": {"n_layer": 8}}}, session, _config_describe()
    )

    written = utrain.runs.read_config(run_id, session)
    assert written["globals"] == {"model": {"n_layer": 8, "dtype": "fp32"}}


def test_write_config_keeps_the_run_id_and_compute(
    session: sqlalchemy.orm.Session, tmp_path: pathlib.Path
) -> None:
    """Neither is a config field, so neither is the form's to change."""
    run_id = _configurable_run(session, tmp_path)
    utrain.runs.write_config(
        run_id, {"run_id": "nope", "compute": "gpu0"}, session, _config_describe()
    )

    written = utrain.runs.read_config(run_id, session)
    assert written["run_id"] == run_id
    assert written["compute"] == "cpu"


def test_write_config_rejects_a_value_outside_the_declared_range(
    session: sqlalchemy.orm.Session, tmp_path: pathlib.Path
) -> None:
    run_id = _configurable_run(session, tmp_path)
    with pytest.raises(utrain.exceptions.UI, match="at most 48"):
        utrain.runs.write_config(
            run_id, {"globals": {"model": {"n_layer": 999}}}, session, _config_describe()
        )
    assert utrain.runs.read_config(run_id, session)["globals"] == {
        "model": {"n_layer": 4, "dtype": "fp32"}
    }


def test_write_config_rejects_a_value_outside_an_enum(
    session: sqlalchemy.orm.Session, tmp_path: pathlib.Path
) -> None:
    run_id = _configurable_run(session, tmp_path)
    with pytest.raises(utrain.exceptions.UI, match="must be one of"):
        utrain.runs.write_config(
            run_id, {"globals": {"model": {"dtype": "int4"}}}, session, _config_describe()
        )


def test_write_config_rejects_text_where_a_number_was_declared(
    session: sqlalchemy.orm.Session, tmp_path: pathlib.Path
) -> None:
    run_id = _configurable_run(session, tmp_path)
    with pytest.raises(utrain.exceptions.UI, match="must be an int"):
        utrain.runs.write_config(
            run_id, {"globals": {"model": {"n_layer": "many"}}}, session, _config_describe()
        )


def test_write_config_refuses_a_run_that_has_started(
    session: sqlalchemy.orm.Session, tmp_path: pathlib.Path
) -> None:
    """Once started, `runs.config_hash` records what the attempt ran with."""
    run_id = _configurable_run(session, tmp_path, status="running")
    with pytest.raises(utrain.exceptions.UI, match="only a configuring run"):
        utrain.runs.write_config(
            run_id, {"globals": {"model": {"n_layer": 8}}}, session, _config_describe()
        )


def test_write_config_puts_a_read_only_file_back_the_way_it_found_it(
    session: sqlalchemy.orm.Session, tmp_path: pathlib.Path
) -> None:
    """`start_run` chmods config.yaml to 0o444; `writable` must restore that."""
    run_id = _configurable_run(session, tmp_path)
    path = utrain.runs.config_path(run_id, session)
    os.chmod(path, 0o444)

    with utrain.runs.writable(path):
        path.write_text("run_id: x\n")

    assert path.stat().st_mode & 0o777 == 0o444
