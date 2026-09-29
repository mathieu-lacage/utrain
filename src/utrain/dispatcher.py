"""The dispatcher: the one process that starts runs and records how they end.

A run is never started by the command that asks for it. `run start`, `run
restart` and a sweep's creation, retry or extension mark the run `queued`, and
the dispatcher starts it once its compute is free -- where free means no run at
all is running on it. It is the only process that starts runs, so two of them
can never be given one GPU at once, however many sweeps and hand-started runs
share it. Runs wait in the order they were queued (`runs.queued_at`); a sweep's
runs are only started while the sweep is `running`.

It is also the only process that records how an attempt, and so a run, ended.
The orchestrator writes its phases' statuses and nothing more. Once its lock is
free the dispatcher finalizes the attempt from them (`reconcile`), which is the
same step whether the orchestrator finished, failed, was stopped or was killed.

There is one dispatcher per data dir. It is detached, like the orchestrator: it
holds a lock for as long as it lives, ticks every couple of seconds, and exits
once nothing is running and nothing it could start is queued. Everything it
knows is in the database, so one that dies loses nothing: every command that
reads or queues runs calls `ensure`, which starts another one if there is work
and nobody to do it.
"""

import os
import subprocess
import sys
import time
import typing

import sqlalchemy
import sqlalchemy.orm

from . import config, exceptions, lock, orchestrator, reconcile
from . import db as dbmod

# How often the dispatcher looks for a free compute and a finished attempt. A
# tick is a few queries and a lock probe per running attempt, so it is cheap;
# it is short because `run start` and `run stop` wait for it.
_TICK_SECONDS = 1.0

# What starts a queued run; its result, an orchestrator's process, is the
# caller's to keep.
Start = typing.Callable[[str, sqlalchemy.orm.Session], object]


def _running_attempts(session: sqlalchemy.orm.Session) -> list[tuple[str, int]]:
    return [
        (str(run_id), int(attempt))
        for run_id, attempt in session.execute(
            sqlalchemy.select(dbmod.run_attempts.c.run_id, dbmod.run_attempts.c.attempt).where(
                dbmod.run_attempts.c.status == "running"
            )
        ).fetchall()
    ]


def _busy(session: sqlalchemy.orm.Session) -> set[str]:
    """Computes something is running on.

    A run's own status is not enough: a run restarted while its last attempt was
    still stopping is `queued`, and its container holds the compute until that
    attempt is finalized.
    """
    running_attempt = sqlalchemy.select(dbmod.run_attempts.c.run_id).where(
        dbmod.run_attempts.c.status == "running"
    )
    return {
        str(c)
        for c in session.execute(
            sqlalchemy.select(dbmod.runs.c.compute).where(
                (dbmod.runs.c.status == "running") | dbmod.runs.c.id.in_(running_attempt)
            )
        )
        .scalars()
        .fetchall()
    }


def _startable(session: sqlalchemy.orm.Session) -> list[tuple[str, str]]:
    """Queued runs that may start once their compute is free, first in line first.

    A run of no sweep may always start; a sweep's run only while the sweep is
    running, not while it is a draft, paused or cancelled.
    """
    return [
        (str(run_id), str(compute))
        for run_id, compute in session.execute(
            sqlalchemy.select(dbmod.runs.c.id, dbmod.runs.c.compute)
            .select_from(
                dbmod.runs.outerjoin(dbmod.sweeps, dbmod.runs.c.sweep_id == dbmod.sweeps.c.id)
            )
            .where(
                (dbmod.runs.c.status == "queued")
                & (dbmod.runs.c.sweep_id.is_(None) | (dbmod.sweeps.c.state == "running"))
            )
            .order_by(dbmod.runs.c.queued_at, dbmod.runs.c.created_at)
        ).fetchall()
    ]


def has_work(session: sqlalchemy.orm.Session) -> bool:
    """Whether a dispatcher has anything to do: an attempt to watch or a run to start."""
    return bool(_running_attempts(session)) or bool(_startable(session))


def is_waiting(run_id: str, session: sqlalchemy.orm.Session) -> bool:
    """Whether a queued run is held back by more than the next tick.

    It is if its sweep is not running, if another run holds its compute, or if
    another run is ahead of it in line for that compute. Otherwise the next
    tick starts it.
    """
    line = _startable(session)
    if run_id not in {r for r, _ in line}:
        return True
    compute = str(dbmod.get_run(run_id, session)["compute"])
    ahead = next(r for r, c in line if c == compute) != run_id
    running_attempt = sqlalchemy.select(dbmod.run_attempts.c.run_id).where(
        dbmod.run_attempts.c.status == "running"
    )
    other = session.execute(
        sqlalchemy.select(dbmod.runs.c.id)
        .where(
            (dbmod.runs.c.compute == compute)
            & (dbmod.runs.c.id != run_id)
            & ((dbmod.runs.c.status == "running") | dbmod.runs.c.id.in_(running_attempt))
        )
        .limit(1)
    ).first()
    return ahead or other is not None


def start_queued(run_id: str, session: sqlalchemy.orm.Session) -> subprocess.Popen[bytes] | None:
    """Start one queued run as its next attempt, from the phase it was queued from.

    None if the run left the queue before it could be started.
    """
    from_phase = dbmod.get_run(run_id, session)["queued_from_phase"]
    return orchestrator.launch(run_id, str(from_phase) if from_phase else None, session)


def tick(
    session: sqlalchemy.orm.Session,
    start: Start = start_queued,
    log: typing.Callable[[str], None] = print,
) -> bool:
    """One tick of the dispatcher. Returns whether there is anything left to do.

    First finalize every attempt whose orchestrator is gone, which frees its
    compute. Then, for each compute that is free, start the first run queued
    for it. A run that cannot be started (its image gone, the GPU toolkit
    missing) is marked failed rather than retried on every tick, and the reason
    goes to the dispatcher's log.
    """
    for run_id, attempt in _running_attempts(session):
        status = reconcile.reconcile_attempt(run_id, attempt, session)
        if status is not None:
            log(f"dispatcher: {run_id} attempt {attempt} {status}")
    session.commit()

    busy = _busy(session)
    for run_id, compute in _startable(session):
        if compute in busy:
            continue
        try:
            start(run_id, session)
            log(f"dispatcher: started {run_id} on {compute}")
        except Exception as e:
            # Anything, not only a UI error: a dispatcher that dies here is
            # restarted by the next read, to fail on the same run again.
            # Failing the run instead lets the rest of the queue go on, and the
            # log says why.
            session.rollback()
            log(f"dispatcher: could not start {run_id} on {compute}: {e!r}")
            session.execute(
                sqlalchemy.update(dbmod.runs)
                .where(dbmod.runs.c.id == run_id)
                .values(status="failed", queued_from_phase=None)
            )
        busy.add(compute)
        session.commit()
    return has_work(session)


def ensure(session: sqlalchemy.orm.Session) -> None:
    """Start a dispatcher if there is work and none is alive.

    Called by every command that reads or queues runs. It is a lock probe while
    the dispatcher lives, and starts one that died, or that exited for lack of
    work before this command gave it more. The session is committed first: the
    dispatcher is another process and must see what this one just wrote -- and
    a dispatcher about to exit looks for work once more after letting go of its
    lock, so anything committed before this probe is seen by one or the other.
    """
    session.commit()
    settings: config.Settings = session.info["settings"]
    if lock.is_held(settings.dispatcher_dir, None, lock.DISPATCHER):
        return
    if not has_work(session):
        return
    _spawn(settings)


def _spawn(settings: config.Settings) -> None:
    directory = settings.dispatcher_dir
    directory.mkdir(parents=True, exist_ok=True)
    # The data dir travels explicitly: the dispatcher, and the orchestrators it
    # starts, must work on the database this session read, whatever utrain.yaml
    # or the environment would have them pick.
    env = {**os.environ, "UTRAIN_DATA_DIR": str(settings.data_dir)}
    # The child keeps its own copy of the descriptor; the parent's is closed.
    with open(directory / "dispatcher.log", "ab") as log:
        subprocess.Popen(
            [sys.executable, "-m", "utrain.cli.main", "_dispatch"],
            stdout=log,
            stderr=log,
            start_new_session=True,
            env=env,
        )


def run(settings: config.Settings) -> None:
    """The dispatcher process: tick until nothing is running or startable."""
    directory = settings.dispatcher_dir
    directory.mkdir(parents=True, exist_ok=True)
    engine = dbmod.create_engine(settings)
    # The orchestrators this process started, kept to be reaped as they exit.
    # Whether one has exited is still read from its lock, since a dispatcher
    # started after a crash is not the parent of the orchestrators it watches.
    children: list[subprocess.Popen[bytes]] = []

    def start(run_id: str, session: sqlalchemy.orm.Session) -> None:
        proc = start_queued(run_id, session)
        if proc is not None:
            children.append(proc)

    while True:
        # Held while ticking; the kernel releases it however this process
        # ends, which is what `ensure` reads.
        try:
            held = lock.hold(directory, lock.DISPATCHER)
        except exceptions.UI:
            _log("dispatcher: another dispatcher is running")
            return
        try:
            while True:
                with sqlalchemy.orm.Session(engine) as session:
                    session.info["settings"] = settings
                    more = tick(session, start, log=_log)
                    session.commit()
                children[:] = [c for c in children if c.poll() is None]
                if not more:
                    break
                time.sleep(_TICK_SECONDS)
        finally:
            held.close()
        # A command that queued a run after the last tick, and found this
        # process still holding the lock, started no dispatcher of its own.
        # Look once more now that the lock is free, so its run is not left
        # waiting for the next command.
        with sqlalchemy.orm.Session(engine) as session:
            if not has_work(session):
                _log("dispatcher: nothing left to do")
                return


def _log(message: str) -> None:
    print(message)
    sys.stdout.flush()
