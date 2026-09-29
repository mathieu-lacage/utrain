"""Creating and migrating the database, from many sessions at once.

The TUI opens sessions from several worker threads the moment it starts, and
the orchestrator and the dispatcher are processes of their own, so the
first ones to reach a fresh or older database race to create and migrate it.
"""

import pathlib
import sqlite3
import threading

import sqlalchemy

import utrain.config
import utrain.db

_THREADS = 16


def _race(settings: utrain.config.Settings) -> list[BaseException]:
    """Open a session from each of many threads, released together."""
    barrier = threading.Barrier(_THREADS)
    errors: list[BaseException] = []

    def open_one() -> None:
        barrier.wait()
        try:
            with utrain.db.with_db(settings) as session:
                session.execute(sqlalchemy.select(utrain.db.runs.c.id)).fetchall()
        except BaseException as e:  # noqa: BLE001 -- collected for the assert
            errors.append(e)

    threads = [threading.Thread(target=open_one) for _ in range(_THREADS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return errors


def _columns(path: pathlib.Path, table: str) -> set[str]:
    with sqlite3.connect(path) as conn:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def test_many_sessions_on_a_fresh_database_create_it_once(tmp_path: pathlib.Path) -> None:
    settings = utrain.config.Settings(data_dir=tmp_path)
    assert _race(settings) == []
    assert "sweep_id" in _columns(settings.db_path, "runs")


def test_many_sessions_on_an_older_database_migrate_it_once(tmp_path: pathlib.Path) -> None:
    """A database from before sweeps: no sweep columns on `runs`, no new tables."""
    settings = utrain.config.Settings(data_dir=tmp_path)
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(settings.db_path) as conn:
        conn.execute(
            "CREATE TABLE runs (id TEXT PRIMARY KEY, name TEXT NOT NULL, image TEXT NOT NULL,"
            " image_id TEXT NOT NULL, compute TEXT NOT NULL, status TEXT NOT NULL,"
            " config_hash TEXT, created_at FLOAT NOT NULL)"
        )
        conn.execute("INSERT INTO runs VALUES ('r1', 'old', 'img', 'x', 'cpu', 'done', NULL, 1.0)")

    assert _race(settings) == []
    assert {"sweep_id", "sweep_point"} <= _columns(settings.db_path, "runs")
    with utrain.db.with_db(settings) as session:
        names = session.execute(sqlalchemy.select(utrain.db.runs.c.name)).scalars().all()
        assert list(names) == ["old"]
        tables = sqlalchemy.inspect(session.connection()).get_table_names()
        assert {"sweeps", "tui_state"} <= set(tables)


def test_runs_queued_before_the_queue_had_an_order_wait_in_creation_order(
    tmp_path: pathlib.Path,
) -> None:
    settings = utrain.config.Settings(data_dir=tmp_path)
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(settings.db_path) as conn:
        conn.execute(
            "CREATE TABLE runs (id TEXT PRIMARY KEY, name TEXT NOT NULL, image TEXT NOT NULL,"
            " image_id TEXT NOT NULL, compute TEXT NOT NULL, status TEXT NOT NULL,"
            " config_hash TEXT, created_at FLOAT NOT NULL, sweep_id TEXT, sweep_point TEXT)"
        )
        conn.execute(
            "INSERT INTO runs VALUES ('q', 'q', 'img', 'x', 'cpu', 'queued', NULL, 5.0, NULL, NULL)"
        )
        conn.execute(
            "INSERT INTO runs VALUES ('d', 'd', 'img', 'x', 'cpu', 'done', NULL, 6.0, NULL, NULL)"
        )

    with utrain.db.with_db(settings) as session:
        queued_at = dict(
            session.execute(sqlalchemy.select(utrain.db.runs.c.id, utrain.db.runs.c.queued_at))
            .tuples()
            .all()
        )
    assert queued_at == {"q": 5.0, "d": None}
