"""Unit tests for the dispatcher: which queued run it starts, and how it finalizes.

Rows are written straight into a fresh database. Starting a run is replaced by
a recorder, and an orchestrator's liveness by whatever the test says, so
nothing here spawns a process: what is under test is the queue's order, which
compute is free, and what an attempt is finalized to.
"""

import ast
import pathlib
import typing

import pytest
import sqlalchemy
import sqlalchemy.orm

import utrain.config
import utrain.container.podman
import utrain.container.schema
import utrain.db
import utrain.dispatcher
import utrain.exceptions
import utrain.lock
import utrain.orchestrator
import utrain.runs

S = utrain.container.schema


@pytest.fixture()
def settings(tmp_path: pathlib.Path) -> utrain.config.Settings:
    return utrain.config.Settings(data_dir=tmp_path / "data")


@pytest.fixture()
def session(settings: utrain.config.Settings) -> typing.Iterator[sqlalchemy.orm.Session]:
    with utrain.db.with_db(settings) as s:
        yield s


def _run(
    session: sqlalchemy.orm.Session,
    name: str,
    *,
    status: str = "queued",
    compute: str = "cpu",
    queued_at: float | None = None,
    sweep_id: str | None = None,
) -> str:
    run_id = name.ljust(32, "0")
    session.execute(
        sqlalchemy.insert(utrain.db.runs).values(
            id=run_id,
            name=name,
            image="img",
            image_id="ab" * 32,
            compute=compute,
            status=status,
            config_hash=None,
            created_at=float(len(name)),
            queued_at=queued_at,
            sweep_id=sweep_id,
        )
    )
    return run_id


def _sweep(session: sqlalchemy.orm.Session, name: str, state: str = "running") -> str:
    sweep_id = name.ljust(32, "0")
    session.execute(
        sqlalchemy.insert(utrain.db.sweeps).values(
            id=sweep_id,
            name=name,
            image="img",
            image_id="ab" * 32,
            spec="{}",
            state=state,
            created_at=1.0,
        )
    )
    return sweep_id


def _attempt(
    session: sqlalchemy.orm.Session, run_id: str, phases: list[str], attempt: int = 1
) -> None:
    """A running attempt whose phases are at `phases`, in order."""
    session.execute(
        sqlalchemy.insert(utrain.db.run_attempts).values(
            run_id=run_id, attempt=attempt, status="running", pid=None, started_at=1.0
        )
    )
    for order, status in enumerate(phases):
        session.execute(
            sqlalchemy.insert(utrain.db.run_phases).values(
                run_id=run_id,
                attempt=attempt,
                phase=f"p{order}",
                phase_order=order,
                status=status,
                started_at=None if status == "pending" else 10.0 + order,
                ended_at=20.0 + order if status in ("done", "failed", "stopped") else None,
            )
        )


class _Starts:
    """Records the runs a tick starts, and marks them running as a launch would."""

    def __init__(self) -> None:
        self.started: list[str] = []

    def __call__(self, run_id: str, session: sqlalchemy.orm.Session) -> None:
        self.started.append(run_id)
        session.execute(
            sqlalchemy.update(utrain.db.runs)
            .where(utrain.db.runs.c.id == run_id)
            .values(status="running")
        )


def _tick(session: sqlalchemy.orm.Session, starts: _Starts) -> bool:
    return utrain.dispatcher.tick(session, starts, log=lambda m: None)


def _status(session: sqlalchemy.orm.Session, run_id: str) -> str:
    return str(utrain.db.get_run(run_id, session)["status"])


@pytest.fixture()
def orchestrators_alive(monkeypatch: pytest.MonkeyPatch) -> set[str]:
    """Attempt dirs whose orchestrator the test says is alive; every other one is gone."""
    alive: set[str] = set()

    def is_held(
        directory: pathlib.Path, fallback_pid: int | None, name: str = utrain.lock.ORCHESTRATOR
    ) -> bool:
        return directory.parent.parent.name in alive

    monkeypatch.setattr(utrain.lock, "is_held", is_held)
    return alive


# -- the queue ------------------------------------------------------------


def test_two_sweeps_on_one_compute_get_one_run_at_a_time(
    session: sqlalchemy.orm.Session,
) -> None:
    a = _run(session, "a1", queued_at=1.0, sweep_id=_sweep(session, "sweep-a"))
    b = _run(session, "b1", queued_at=2.0, sweep_id=_sweep(session, "sweep-b"))
    starts = _Starts()

    assert _tick(session, starts)
    assert starts.started == [a]
    _tick(session, starts)
    assert starts.started == [a]
    assert _status(session, b) == "queued"


def test_runs_wait_in_the_order_they_were_queued(session: sqlalchemy.orm.Session) -> None:
    # Created first, queued last: a run retried after others were queued.
    late = _run(session, "late", queued_at=3.0)
    early = _run(session, "early-one", queued_at=1.0, compute="gpu0")
    first = _run(session, "first-one", queued_at=2.0)
    starts = _Starts()

    _tick(session, starts)
    assert starts.started == [early, first]
    assert _status(session, late) == "queued"


def test_a_held_back_sweep_does_not_block_a_run_queued_behind_it(
    session: sqlalchemy.orm.Session,
) -> None:
    paused = _run(session, "paused", queued_at=1.0, sweep_id=_sweep(session, "p", "paused"))
    draft = _run(session, "draft", queued_at=2.0, sweep_id=_sweep(session, "d", "draft"))
    mine = _run(session, "mine", queued_at=3.0)
    starts = _Starts()

    _tick(session, starts)
    assert starts.started == [mine]
    assert _status(session, paused) == _status(session, draft) == "queued"


def test_nothing_queued_and_nothing_running_is_nothing_to_do(
    session: sqlalchemy.orm.Session,
) -> None:
    _run(session, "done", status="done")
    _run(session, "held", queued_at=1.0, sweep_id=_sweep(session, "p", "paused"))
    assert not _tick(session, _Starts())


# -- finalizing -----------------------------------------------------------


@pytest.mark.parametrize(
    ("phases", "final"),
    [
        (["done", "done"], "done"),
        (["done", "failed"], "failed"),
        (["stopped", "stopped"], "stopped"),
        # Killed mid-phase: nothing recorded how it ended.
        (["done", "running"], "stopped"),
    ],
)
def test_an_attempt_is_finalized_from_its_phases_once_its_orchestrator_is_gone(
    session: sqlalchemy.orm.Session,
    orchestrators_alive: set[str],
    phases: list[str],
    final: str,
) -> None:
    run_id = _run(session, "gone", status="running")
    _attempt(session, run_id, phases)

    assert not _tick(session, _Starts())

    assert _status(session, run_id) == final
    attempt = session.execute(sqlalchemy.select(utrain.db.run_attempts)).mappings().one()
    assert attempt["status"] == final
    assert attempt["ended_at"] is not None
    assert (
        "running"
        not in session.execute(sqlalchemy.select(utrain.db.run_phases.c.status))
        .scalars()
        .fetchall()
    )


def test_an_attempt_ends_when_its_last_phase_did(
    session: sqlalchemy.orm.Session, orchestrators_alive: set[str]
) -> None:
    run_id = _run(session, "gone", status="running")
    _attempt(session, run_id, ["done", "done"])
    _tick(session, _Starts())
    ended = session.execute(sqlalchemy.select(utrain.db.run_attempts.c.ended_at)).scalar_one()
    assert ended == 21.0


def test_a_live_orchestrator_is_left_alone_and_keeps_its_compute(
    session: sqlalchemy.orm.Session, orchestrators_alive: set[str]
) -> None:
    run_id = _run(session, "live", status="running")
    _attempt(session, run_id, ["done", "running"])
    orchestrators_alive.add(run_id)
    waiting = _run(session, "waiting", queued_at=1.0)
    starts = _Starts()

    assert _tick(session, starts)
    assert _status(session, run_id) == "running"
    assert starts.started == []

    # Once it exits, the same tick that finalizes it hands its compute on.
    orchestrators_alive.clear()
    _tick(session, starts)
    assert _status(session, run_id) == "stopped"
    assert starts.started == [waiting]


def test_a_run_restarted_while_stopping_waits_for_its_last_attempt(
    session: sqlalchemy.orm.Session, orchestrators_alive: set[str]
) -> None:
    run_id = _run(session, "again", status="queued", queued_at=1.0)
    _attempt(session, run_id, ["running"])
    orchestrators_alive.add(run_id)
    starts = _Starts()

    _tick(session, starts)
    assert starts.started == []

    orchestrators_alive.clear()
    _tick(session, starts)
    # The old attempt ends as stopped, but the run was queued again: it is not
    # marked stopped, and it starts now that its compute is free.
    attempt = session.execute(sqlalchemy.select(utrain.db.run_attempts.c.status)).scalar_one()
    assert attempt == "stopped"
    assert starts.started == [run_id]


# -- starting -------------------------------------------------------------


def test_a_queued_restart_is_launched_from_its_phase(
    monkeypatch: pytest.MonkeyPatch, session: sqlalchemy.orm.Session
) -> None:
    launched: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        utrain.orchestrator,
        "launch",
        lambda run_id, from_phase, session: launched.append((run_id, from_phase)),
    )
    run_id = _run(session, "again", queued_at=1.0)
    session.execute(
        sqlalchemy.update(utrain.db.runs)
        .where(utrain.db.runs.c.id == run_id)
        .values(queued_from_phase="train")
    )
    utrain.dispatcher.start_queued(run_id, session)
    assert launched == [(run_id, "train")]


def test_a_run_that_cannot_start_is_failed_and_the_queue_goes_on(
    session: sqlalchemy.orm.Session,
) -> None:
    broken = _run(session, "broken", queued_at=1.0)
    other = _run(session, "other", queued_at=2.0, compute="gpu0")
    session.commit()
    started: list[str] = []
    messages: list[str] = []

    def start(run_id: str, session: sqlalchemy.orm.Session) -> None:
        if run_id == broken:
            raise utrain.exceptions.UI("image gone")
        started.append(run_id)

    utrain.dispatcher.tick(session, start, log=messages.append)
    assert _status(session, broken) == "failed"
    assert started == [other]
    assert any("image gone" in m for m in messages)


# -- ensure ---------------------------------------------------------------


def test_ensure_starts_a_dispatcher_only_for_work_nobody_is_doing(
    monkeypatch: pytest.MonkeyPatch,
    session: sqlalchemy.orm.Session,
    settings: utrain.config.Settings,
    _no_dispatcher: list[utrain.config.Settings],
) -> None:
    utrain.dispatcher.ensure(session)
    assert _no_dispatcher == []  # nothing to do

    _run(session, "queued", queued_at=1.0)
    monkeypatch.setattr(utrain.lock, "is_held", lambda *a: True)
    utrain.dispatcher.ensure(session)
    assert _no_dispatcher == []  # one is alive

    monkeypatch.setattr(utrain.lock, "is_held", lambda *a: False)
    utrain.dispatcher.ensure(session)
    assert [s.data_dir for s in _no_dispatcher] == [settings.data_dir]


# -- runs through the queue -----------------------------------------------


@pytest.fixture()
def configuring(session: sqlalchemy.orm.Session, settings: utrain.config.Settings) -> str:
    run_id = _run(session, "fresh", status="configuring")
    run_dir = settings.runs_dir / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text("run_id: fresh\n")
    return run_id


def test_run_start_freezes_the_config_and_queues_the_run(
    session: sqlalchemy.orm.Session,
    settings: utrain.config.Settings,
    configuring: str,
    _no_dispatcher: list[utrain.config.Settings],
) -> None:
    utrain.runs.start_run(configuring, session)

    row = utrain.db.get_run(configuring, session)
    assert row["status"] == "queued"
    assert row["queued_at"] is not None
    assert row["config_hash"] is not None
    assert len(_no_dispatcher) == 1
    with pytest.raises(utrain.exceptions.UI, match="already queued"):
        utrain.runs.start_run(configuring, session)


def test_stopping_a_queued_run_takes_it_off_the_queue(
    session: sqlalchemy.orm.Session, configuring: str
) -> None:
    utrain.runs.start_run(configuring, session)
    utrain.runs.stop_run(configuring, session)
    assert _status(session, configuring) == "stopped"
    assert not _tick(session, _Starts())


def test_stopping_a_running_run_leaves_its_status_to_the_dispatcher(
    session: sqlalchemy.orm.Session, settings: utrain.config.Settings
) -> None:
    run_id = _run(session, "going", status="running")
    _attempt(session, run_id, ["running"])
    utrain.orchestrator.init_mount_dir(
        settings.runs_dir / run_id / "attempt" / "1", _config(settings, run_id)
    )
    utrain.runs.stop_run(run_id, session)
    assert _status(session, run_id) == "running"
    control = settings.runs_dir / run_id / "attempt" / "1" / "mnt" / "control.json"
    assert "stop" in control.read_text()


def test_restart_from_a_phase_queues_it(
    monkeypatch: pytest.MonkeyPatch, session: sqlalchemy.orm.Session
) -> None:
    monkeypatch.setattr(utrain.container.podman, "image_exists", lambda ref: True)
    monkeypatch.setattr(
        utrain.container.podman,
        "describe",
        lambda ref: S.DescribeOutput(
            name="img",
            phases=[S.PhaseInfo(name="prep", label="Prep"), S.PhaseInfo(name="train", label="T")],
            phase_order=["prep", "train"],
            config_schema=S.ConfigSchema(),
        ),
    )
    run_id = _run(session, "again", status="failed")
    utrain.runs.restart_run(run_id, "train", session)
    row = utrain.db.get_run(run_id, session)
    assert (row["status"], row["queued_from_phase"]) == ("queued", "train")
    with pytest.raises(utrain.exceptions.UI, match="not found in image"):
        utrain.runs.restart_run(run_id, "nope", session)


def _config(settings: utrain.config.Settings, run_id: str) -> pathlib.Path:
    path = settings.runs_dir / run_id / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"run_id: {run_id}\n")
    return path


def test_only_the_dispatcher_finalizes_runs() -> None:
    """Reading a run writes nothing, so nothing but the dispatcher imports `reconcile`."""
    src = pathlib.Path(utrain.dispatcher.__file__).parent
    importers = {
        path.relative_to(src).as_posix()
        for path in src.rglob("*.py")
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.ImportFrom)
        and any(alias.name == "reconcile" for alias in node.names)
    }
    assert importers == {"dispatcher.py"}


def test_a_queued_run_is_waiting_only_on_something_besides_the_next_tick(
    session: sqlalchemy.orm.Session,
) -> None:
    first = _run(session, "first", queued_at=1.0)
    second = _run(session, "second", queued_at=2.0)
    held = _run(
        session, "held", queued_at=0.5, compute="gpu0", sweep_id=_sweep(session, "p", "paused")
    )
    assert not utrain.dispatcher.is_waiting(first, session)
    assert utrain.dispatcher.is_waiting(second, session)  # behind `first`
    assert utrain.dispatcher.is_waiting(held, session)  # its sweep is paused

    _run(session, "busy", status="running", compute="gpu1")
    alone = _run(session, "alone", queued_at=3.0, compute="gpu1")
    assert utrain.dispatcher.is_waiting(alone, session)


def test_launching_a_run_taken_off_the_queue_does_nothing(
    monkeypatch: pytest.MonkeyPatch,
    session: sqlalchemy.orm.Session,
    settings: utrain.config.Settings,
) -> None:
    monkeypatch.setattr(utrain.container.podman, "image_exists", lambda ref: True)
    monkeypatch.setattr(
        utrain.container.podman,
        "describe",
        lambda ref: S.DescribeOutput(
            name="img",
            phases=[S.PhaseInfo(name="train", label="Train")],
            phase_order=["train"],
            config_schema=S.ConfigSchema(),
        ),
    )
    run_id = _run(session, "stopped", status="stopped")
    _config(settings, run_id)
    assert utrain.orchestrator.launch(run_id, None, session) is None
    assert session.execute(sqlalchemy.select(utrain.db.run_attempts)).fetchall() == []
    assert _status(session, run_id) == "stopped"


def test_an_orchestrator_reads_a_stop_before_each_phase(tmp_path: pathlib.Path) -> None:
    attempt_dir = tmp_path / "attempt" / "1"
    config = tmp_path / "config.yaml"
    config.write_text("run_id: x\n")
    utrain.orchestrator.init_mount_dir(attempt_dir, config)
    assert not utrain.orchestrator.stop_requested(attempt_dir)
    utrain.orchestrator.write_control(attempt_dir, "stop")
    assert utrain.orchestrator.stop_requested(attempt_dir)
