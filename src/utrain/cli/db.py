import collections.abc
import contextlib
import sqlite3

import sqlalchemy
import sqlalchemy.event
import sqlalchemy.orm

from .. import config
from . import exceptions

metadata = sqlalchemy.MetaData()

runs = sqlalchemy.Table(
    "runs",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Text, primary_key=True),
    sqlalchemy.Column("name", sqlalchemy.Text, nullable=False),
    sqlalchemy.Column("image", sqlalchemy.Text, nullable=False),
    sqlalchemy.Column("compute", sqlalchemy.Text, nullable=False),
    sqlalchemy.Column("run_dir", sqlalchemy.Text, nullable=False),
    sqlalchemy.Column("status", sqlalchemy.Text, nullable=False, default="configuring"),
    sqlalchemy.Column("config_hash", sqlalchemy.Text, nullable=True),
    sqlalchemy.Column("created_at", sqlalchemy.Float, nullable=False),
)

run_attempts = sqlalchemy.Table(
    "run_attempts",
    metadata,
    sqlalchemy.Column("run_id", sqlalchemy.Text, nullable=False),
    sqlalchemy.Column("attempt", sqlalchemy.Integer, nullable=False),
    sqlalchemy.Column("from_phase", sqlalchemy.Text, nullable=True),
    sqlalchemy.Column("status", sqlalchemy.Text, nullable=False),
    sqlalchemy.Column("pid", sqlalchemy.Integer, nullable=True),
    sqlalchemy.Column("started_at", sqlalchemy.Float, nullable=False),
    sqlalchemy.Column("ended_at", sqlalchemy.Float, nullable=True),
    sqlalchemy.PrimaryKeyConstraint("run_id", "attempt"),
)

run_phases = sqlalchemy.Table(
    "run_phases",
    metadata,
    sqlalchemy.Column("run_id", sqlalchemy.Text, nullable=False),
    sqlalchemy.Column("attempt", sqlalchemy.Integer, nullable=False),
    sqlalchemy.Column("phase", sqlalchemy.Text, nullable=False),
    sqlalchemy.Column("phase_order", sqlalchemy.Integer, nullable=False),
    sqlalchemy.Column("status", sqlalchemy.Text, nullable=False),
    sqlalchemy.Column("started_at", sqlalchemy.Float, nullable=True),
    sqlalchemy.Column("ended_at", sqlalchemy.Float, nullable=True),
    sqlalchemy.PrimaryKeyConstraint("run_id", "attempt", "phase"),
)


def _set_sqlite_pragmas(dbapi_conn: sqlite3.Connection, _record: object) -> None:
    # A detached orchestrator writes the DB concurrently with foreground read
    # commands (notably `run show --wait`). Wait up to 5s for a held lock instead
    # of failing immediately with "database is locked".
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()


def create_engine(settings: config.Settings) -> sqlalchemy.Engine:
    url = f"sqlite:///{settings.db_path}"
    engine = sqlalchemy.create_engine(url, connect_args={"check_same_thread": False})
    sqlalchemy.event.listen(engine, "connect", _set_sqlite_pragmas)
    return engine


def init_db(engine: sqlalchemy.Engine) -> None:
    metadata.create_all(engine)
    _migrate(engine)


def _migrate(engine: sqlalchemy.Engine) -> None:
    with engine.begin() as conn:
        inspector = sqlalchemy.inspect(engine)
        existing = {
            t: {c["name"] for c in inspector.get_columns(t)} for t in inspector.get_table_names()
        }

        # Rename gpu column to compute if it exists and compute doesn't
        if "runs" in existing and "gpu" in existing["runs"] and "compute" not in existing["runs"]:
            conn.execute(sqlalchemy.text("ALTER TABLE runs RENAME COLUMN gpu TO compute"))
            # Refresh existing set after the rename
            existing["runs"].discard("gpu")
            existing["runs"].add("compute")

        # Add new columns to legacy 'runs' table if it came from the old schema
        if "runs" in existing:
            for col, ddl in [
                ("name", "TEXT NOT NULL DEFAULT ''"),
                ("image", "TEXT NOT NULL DEFAULT ''"),
                ("compute", "TEXT NOT NULL DEFAULT 'cpu'"),
                ("config_hash", "TEXT"),
                ("created_at", "REAL NOT NULL DEFAULT 0"),
            ]:
                if col not in existing["runs"]:
                    conn.execute(sqlalchemy.text(f"ALTER TABLE runs ADD COLUMN {col} {ddl}"))

        # Drop columns that don't belong in the new schema (SQLite can't DROP columns
        # before 3.35; skip silently — the extra columns are harmless).


@contextlib.contextmanager
def with_db(
    settings: config.Settings,
) -> collections.abc.Generator[sqlalchemy.orm.Session, None, None]:
    engine = create_engine(settings)
    init_db(engine)
    with sqlalchemy.orm.Session(engine) as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise


def resolve_run_id(prefix: str, session: sqlalchemy.orm.Session) -> str:
    rows = (
        session.execute(sqlalchemy.select(runs.c.id).where(runs.c.id.like(f"{prefix}%")))
        .scalars()
        .fetchall()
    )
    if not rows:
        raise exceptions.UI(f"abort: run '{prefix}' not found")
    if len(rows) > 1:
        matches = ", ".join(str(r) for r in rows[:4])
        raise exceptions.UI(f"abort: id prefix '{prefix}' is ambiguous (matches: {matches})")
    return str(rows[0])


def short_run_id(run_id: str, session: sqlalchemy.orm.Session) -> str:
    """Shortest prefix of run_id that is unique across the runs table.

    Uses run_id's lexicographic neighbors (PK-indexed), so it avoids scanning the
    whole table. The result may be one char shorter than the uniform width used by
    'run list', but still resolves unambiguously through resolve_run_id.
    """
    pred = session.execute(
        sqlalchemy.select(runs.c.id).where(runs.c.id < run_id).order_by(runs.c.id.desc()).limit(1)
    ).scalar_one_or_none()
    succ = session.execute(
        sqlalchemy.select(runs.c.id).where(runs.c.id > run_id).order_by(runs.c.id.asc()).limit(1)
    ).scalar_one_or_none()

    def _lcp(a: str, b: str | None) -> int:
        if b is None:
            return 0
        n = 0
        for ca, cb in zip(a, b):
            if ca != cb:
                break
            n += 1
        return n

    plen = 1 + max(
        _lcp(run_id, str(pred) if pred is not None else None),
        _lcp(run_id, str(succ) if succ is not None else None),
    )
    return run_id[:plen]


def latest_attempt(run_id: str, session: sqlalchemy.orm.Session) -> int | None:
    result = session.execute(
        sqlalchemy.select(sqlalchemy.func.max(run_attempts.c.attempt)).where(
            run_attempts.c.run_id == run_id
        )
    ).scalar_one_or_none()
    return int(result) if result is not None else None


def get_run(run_id: str, session: sqlalchemy.orm.Session) -> sqlalchemy.engine.RowMapping:
    row = session.execute(sqlalchemy.select(runs).where(runs.c.id == run_id)).mappings().fetchone()
    if row is None:
        raise exceptions.UI(f"abort: run '{run_id}' not found")
    return row
