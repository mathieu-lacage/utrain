"""Unit tests for the query layer.

These exist because the query layer no longer prints: `runs.list_runs` and
friends return dataclasses, so they can be asserted on directly instead of
through captured stdout. Nothing here needs podman, so unlike the cram suite
these do run in CI.
"""

import ast
import os
import pathlib
import random
import shutil
import subprocess
import sys
import tempfile
import typing

import pytest
import sqlalchemy
import sqlalchemy.orm

import utrain.cli.render
import utrain.config
import utrain.container.podman
import utrain.container.schema
import utrain.db
import utrain.exceptions
import utrain.lock
import utrain.logs
import utrain.orchestrator
import utrain.phases
import utrain.runs
import utrain.types

# A plausible podman image id, for seeds and frozen-id assertions.
FROZEN_ID = "ab" * 32


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
    image_id: str = FROZEN_ID,
) -> None:
    session.execute(
        sqlalchemy.insert(utrain.db.runs).values(
            id=run_id,
            name=name,
            image="img",
            image_id=image_id,
            compute="cpu",
            status=status,
            config_hash=None,
            created_at=created_at,
        )
    )


def _store_description(
    session: sqlalchemy.orm.Session, described: utrain.container.schema.DescribeOutput
) -> None:
    """What `runs.create_run` stores of the image, for runs inserted directly."""
    session.execute(
        sqlalchemy.insert(utrain.db.image_descriptions).values(
            image_id=FROZEN_ID, describe=described.model_dump_json()
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

    _store_description(session, _described())
    entries = utrain.phases.list_phases(run_id, session)

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
            image_id=None,
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
            image_id=FROZEN_ID,
            compute="cpu",
            status=status,
            config_hash=None,
            created_at=100.0,
        )
    )
    _store_description(session, _config_describe())
    return run_id


def test_read_config_parses_the_file(
    session: sqlalchemy.orm.Session, tmp_path: pathlib.Path
) -> None:
    run_id = _configurable_run(session, tmp_path)
    assert utrain.runs.read_config(run_id, session)["globals"] == {
        "model": {"n_layer": 4, "dtype": "fp32"}
    }


def test_run_survives_moving_the_whole_data_dir(tmp_path: pathlib.Path) -> None:
    """run_dir is derived from data_dir on every read rather than stored, so
    relocating db + runs together (as one data_dir tree) doesn't strand a run."""
    old_dir = tmp_path / "old"
    old_dir.mkdir()
    with utrain.db.with_db(utrain.config.Settings(data_dir=old_dir)) as session:
        run_id = _configurable_run(session, old_dir)

    new_dir = tmp_path / "new"
    shutil.move(str(old_dir), str(new_dir))

    with utrain.db.with_db(utrain.config.Settings(data_dir=new_dir)) as session:
        detail = utrain.runs.get_run_detail(run_id, session)
        assert detail.run_dir == new_dir / "runs" / run_id
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

    utrain.runs.write_config(run_id, {"globals": {"model": {"n_layer": 8}}}, session)

    written = utrain.runs.read_config(run_id, session)
    assert written["globals"] == {"model": {"n_layer": 8, "dtype": "fp32"}}


def test_write_config_keeps_the_run_id_and_compute(
    session: sqlalchemy.orm.Session, tmp_path: pathlib.Path
) -> None:
    """Neither is a config field, so neither is the form's to change."""
    run_id = _configurable_run(session, tmp_path)
    utrain.runs.write_config(run_id, {"run_id": "nope", "compute": "gpu0"}, session)

    written = utrain.runs.read_config(run_id, session)
    assert written["run_id"] == run_id
    assert written["compute"] == "cpu"


def test_write_config_rejects_a_value_outside_the_declared_range(
    session: sqlalchemy.orm.Session, tmp_path: pathlib.Path
) -> None:
    run_id = _configurable_run(session, tmp_path)
    with pytest.raises(utrain.exceptions.UI, match="at most 48"):
        utrain.runs.write_config(run_id, {"globals": {"model": {"n_layer": 999}}}, session)
    assert utrain.runs.read_config(run_id, session)["globals"] == {
        "model": {"n_layer": 4, "dtype": "fp32"}
    }


def test_write_config_rejects_a_value_outside_an_enum(
    session: sqlalchemy.orm.Session, tmp_path: pathlib.Path
) -> None:
    run_id = _configurable_run(session, tmp_path)
    with pytest.raises(utrain.exceptions.UI, match="must be one of"):
        utrain.runs.write_config(run_id, {"globals": {"model": {"dtype": "int4"}}}, session)


def test_write_config_rejects_text_where_a_number_was_declared(
    session: sqlalchemy.orm.Session, tmp_path: pathlib.Path
) -> None:
    run_id = _configurable_run(session, tmp_path)
    with pytest.raises(utrain.exceptions.UI, match="must be an int"):
        utrain.runs.write_config(run_id, {"globals": {"model": {"n_layer": "many"}}}, session)


def test_write_config_refuses_a_run_that_has_started(
    session: sqlalchemy.orm.Session, tmp_path: pathlib.Path
) -> None:
    """Once started, `runs.config_hash` records what the attempt ran with."""
    run_id = _configurable_run(session, tmp_path, status="running")
    with pytest.raises(utrain.exceptions.UI, match="only a configuring run"):
        utrain.runs.write_config(run_id, {"globals": {"model": {"n_layer": 8}}}, session)


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


def test_ensure_gpu_toolkit_refuses_a_gpu_compute_without_nvidia_ctk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A GPU run without the toolkit is a user error, not a crash in a log."""
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(utrain.exceptions.UI, match="nvidia-ctk not found"):
        utrain.orchestrator.ensure_gpu_toolkit("gpu0")


def test_ensure_gpu_toolkit_lets_cpu_runs_through_without_nvidia_ctk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU runs stay independent of the NVIDIA toolchain."""
    monkeypatch.setattr(shutil, "which", lambda name: None)
    utrain.orchestrator.ensure_gpu_toolkit("cpu")


def test_start_run_refuses_a_gpu_run_without_the_container_toolkit(
    session: sqlalchemy.orm.Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """start_run preflights the toolkit while it can still tell the user.

    The orchestrator it spawns is detached, so an error raised there would land
    in orchestrator.log and the run would go failed with no visible reason.
    """
    _insert_run(session, "5dc7917c", "toolkit-less", 1.0)
    session.execute(
        sqlalchemy.update(utrain.db.runs)
        .where(utrain.db.runs.c.id == "5dc7917c")
        .values(compute="gpu0")
    )
    monkeypatch.setattr(shutil, "which", lambda name: None)

    with pytest.raises(utrain.exceptions.UI, match="nvidia-ctk not found"):
        utrain.runs.start_run("5dc7917c", session)

    # Nothing was created: no attempt, and the run is still configurable.
    assert utrain.db.latest_attempt("5dc7917c", session) is None


# -- frozen image ids ------------------------------------------------------


def test_run_image_ref_returns_the_frozen_id(
    session: sqlalchemy.orm.Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run resolves to the id it was created with, never the name's target.

    The id is what makes a re-tag of the image name invisible to the run, so
    resolving must not even look at the preset list.
    """
    _insert_run(session, "f0ee" * 8, "pinned", 1.0, image_id=FROZEN_ID)
    monkeypatch.setattr(utrain.container.podman, "image_exists", lambda ref: True)
    monkeypatch.setattr(utrain.container.podman, "list_presets", typing.cast(typing.Any, None))

    ref = utrain.db.run_image_ref(utrain.db.get_run("f0ee" * 8, session))

    assert ref == FROZEN_ID


def test_run_image_ref_refuses_a_frozen_image_that_is_gone(
    session: sqlalchemy.orm.Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A frozen id that left the local store is an error, not a silent re-resolve.

    Falling back to the name would quietly run the run against different
    content -- exactly what freezing exists to prevent.
    """
    _insert_run(session, "d2d2" * 8, "gone", 1.0, image_id=FROZEN_ID)
    monkeypatch.setattr(utrain.container.podman, "image_exists", lambda ref: False)

    with pytest.raises(utrain.exceptions.UI, match="not in the local store"):
        utrain.db.run_image_ref(utrain.db.get_run("d2d2" * 8, session))


def test_create_run_freezes_the_image_id(
    tmp_path: pathlib.Path, session: sqlalchemy.orm.Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The id is captured at creation and described, so the run owns it."""
    monkeypatch.setattr(
        utrain.container.podman,
        "list_presets",
        lambda: {"utrain-fake": "localhost/utrain-fake:utrain"},
    )
    monkeypatch.setattr(utrain.container.podman, "image_id", lambda ref: FROZEN_ID)
    described = _described()
    described_calls: list[str] = []

    def fake_describe(ref: str) -> utrain.container.schema.DescribeOutput:
        described_calls.append(ref)
        return described

    monkeypatch.setattr(utrain.container.podman, "describe", fake_describe)

    settings = utrain.config.Settings(data_dir=tmp_path)
    run_id = utrain.runs.create_run("hello", "utrain-fake", "cpu", settings, session)

    row = utrain.db.get_run(run_id, session)
    assert row["image_id"] == FROZEN_ID
    # Describing went to the id, not the tag: whatever the tag does later, the
    # config this run was seeded from is the frozen image's.
    assert described_calls == [FROZEN_ID]


def test_an_image_is_described_once_whatever_is_created_from_it(
    tmp_path: pathlib.Path, session: sqlalchemy.orm.Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        utrain.container.podman, "list_presets", lambda: {"img": "localhost/utrain-img:utrain"}
    )
    monkeypatch.setattr(utrain.container.podman, "image_id", lambda ref: FROZEN_ID)
    calls: list[str] = []

    def describe(ref: str) -> utrain.container.schema.DescribeOutput:
        calls.append(ref)
        return _described()

    monkeypatch.setattr(utrain.container.podman, "describe", describe)
    settings = utrain.config.Settings(data_dir=tmp_path)
    utrain.runs.create_run("one", "img", "cpu", settings, session)
    utrain.runs.create_run("two", "img", "cpu", settings, session)

    assert calls == [FROZEN_ID]
    stored = session.execute(sqlalchemy.select(utrain.db.image_descriptions)).fetchall()
    assert len(stored) == 1


def test_reading_a_run_never_starts_a_container(
    session: sqlalchemy.orm.Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    def describe(ref: str) -> utrain.container.schema.DescribeOutput:
        raise AssertionError("a read started a container")

    monkeypatch.setattr(utrain.container.podman, "describe", describe)
    run_id = "e" * 32
    _insert_run(session, run_id, "busy", created_at=1.0, status="running")
    session.execute(
        sqlalchemy.insert(utrain.db.run_attempts).values(
            run_id=run_id, attempt=1, from_phase=None, status="running", pid=None, started_at=1.0
        )
    )
    for order, (phase, status) in enumerate([("prepare", "running"), ("train", "pending")]):
        session.execute(
            sqlalchemy.insert(utrain.db.run_phases).values(
                run_id=run_id, attempt=1, phase=phase, phase_order=order, status=status
            )
        )
    _store_description(session, _described())

    entries = utrain.phases.list_phases(run_id, session)
    assert [e.phase for e in entries] == ["prepare", "train"]
    assert utrain.phases.list_phase_ids(run_id, session) == [
        f"{run_id}/1/prepare",
        f"{run_id}/1/train",
    ]


def test_a_run_whose_image_description_is_missing_says_so(
    session: sqlalchemy.orm.Session,
) -> None:
    run_id = "e" * 32
    _insert_run(session, run_id, "orphan", created_at=1.0, status="running")
    session.execute(
        sqlalchemy.insert(utrain.db.run_attempts).values(
            run_id=run_id, attempt=1, from_phase=None, status="running", pid=None, started_at=1.0
        )
    )
    with pytest.raises(utrain.exceptions.UI, match="no description of image"):
        utrain.phases.list_phases(run_id, session)


def test_only_the_database_layer_asks_an_image_to_describe_itself() -> None:
    """Everything else reads the stored description, so no read starts a container."""
    src = pathlib.Path(utrain.db.__file__).parent
    callers = {
        path.relative_to(src).as_posix()
        for path in src.rglob("*.py")
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "describe"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "podman"
    }
    assert callers == {"db.py"}
