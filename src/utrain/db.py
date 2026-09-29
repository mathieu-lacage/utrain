import collections.abc
import contextlib
import pathlib
import sqlite3

import sqlalchemy
import sqlalchemy.dialects.sqlite
import sqlalchemy.event
import sqlalchemy.exc
import sqlalchemy.orm

from . import config, container, exceptions

metadata = sqlalchemy.MetaData()

runs = sqlalchemy.Table(
    "runs",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Text, primary_key=True),
    sqlalchemy.Column("name", sqlalchemy.Text, nullable=False),
    sqlalchemy.Column("image", sqlalchemy.Text, nullable=False),
    # The podman image id frozen when the run was created, so a later re-tag of
    # the name cannot change what the run sees (config schema, phase order) or
    # runs. Resolved through db.run_image_ref.
    sqlalchemy.Column("image_id", sqlalchemy.Text, nullable=False),
    sqlalchemy.Column("compute", sqlalchemy.Text, nullable=False),
    sqlalchemy.Column("status", sqlalchemy.Text, nullable=False, default="configuring"),
    sqlalchemy.Column("config_hash", sqlalchemy.Text, nullable=True),
    sqlalchemy.Column("created_at", sqlalchemy.Float, nullable=False),
    # The sweep that generated this run, and the point of its grid the run
    # stands for: a JSON object of axis path to value. Both null for a run
    # created on its own. See `utrain.sweeps`.
    sqlalchemy.Column("sweep_id", sqlalchemy.Text, nullable=True),
    sqlalchemy.Column("sweep_point", sqlalchemy.Text, nullable=True),
    # When the run last became `queued`, which orders the dispatcher's queue:
    # a run requeued by a retry goes behind the runs already waiting. Null for
    # a run that was never queued.
    sqlalchemy.Column("queued_at", sqlalchemy.Float, nullable=True),
    # The phase a queued restart starts from, for the dispatcher to hand the
    # orchestrator. Null for a restart from the first phase, and for a first
    # start.
    sqlalchemy.Column("queued_from_phase", sqlalchemy.Text, nullable=True),
)

sweeps = sqlalchemy.Table(
    "sweeps",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Text, primary_key=True),
    # Unique, unlike a run's: a sweep is addressed by name (`@lr-depth`) as
    # often as by id, so two of the same name would make one unreachable.
    sqlalchemy.Column("name", sqlalchemy.Text, nullable=False, unique=True),
    sqlalchemy.Column("image", sqlalchemy.Text, nullable=False),
    # Frozen once for the whole sweep, as a run freezes its own: every point
    # runs the same content, which is what makes them comparable.
    sqlalchemy.Column("image_id", sqlalchemy.Text, nullable=False),
    # The spec as JSON: axes, replicate axes, computes and base run. See
    # `sweeps.Spec`.
    sqlalchemy.Column("spec", sqlalchemy.Text, nullable=False),
    # What the user last asked of the sweep: draft, running, paused or
    # cancelled. The dispatcher starts a sweep's queued runs only while it is
    # running. Whether the sweep is *done* is derived from its runs.
    sqlalchemy.Column("state", sqlalchemy.Text, nullable=False),
    sqlalchemy.Column("created_at", sqlalchemy.Float, nullable=False),
)

# Saved state of the TUI, one row per key: the run tray, the comparison
# Compare is showing, and named comparisons. JSON values, so that what the
# TUI keeps can change without a migration each time.
tui_state = sqlalchemy.Table(
    "tui_state",
    metadata,
    sqlalchemy.Column("key", sqlalchemy.Text, primary_key=True),
    sqlalchemy.Column("value", sqlalchemy.Text, nullable=False),
)

# What each image said about itself -- its phases, their labels and plots, its
# config schema -- as `container.schema.DescribeOutput` JSON. Asking means
# starting a container, which for an image that imports its training stack
# takes seconds, and the answer cannot change: an image id is immutable, and
# runs and sweeps are frozen to one. So it is asked once, when a run or sweep
# is first created from the image, and read from here ever after.
image_descriptions = sqlalchemy.Table(
    "image_descriptions",
    metadata,
    sqlalchemy.Column("image_id", sqlalchemy.Text, primary_key=True),
    sqlalchemy.Column("describe", sqlalchemy.Text, nullable=False),
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
    # SQLite creates the database file on first connect, but not the parent
    # directories -- a fresh data_dir would otherwise fail with
    # "unable to open database file" before anything could create it.
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"sqlite:///{settings.db_path}"
    engine = sqlalchemy.create_engine(url, connect_args={"check_same_thread": False})
    sqlalchemy.event.listen(engine, "connect", _set_sqlite_pragmas)
    return engine


def init_db(engine: sqlalchemy.Engine) -> None:
    """Create the tables and bring an older schema up to date.

    Safe to race. The TUI opens sessions from several threads at once, and the
    orchestrator and the dispatcher are processes of their own: two of
    them each checking that a table is missing and then creating it would have
    the second fail with "table already exists". So the check-and-create runs
    under `BEGIN IMMEDIATE`, SQLite's write lock, which the others wait on (see
    the busy timeout) and after which they find nothing left to do.

    The lock is only taken when something is missing: an up-to-date database,
    which is every call but the first, is answered by a read.
    """
    with engine.connect() as conn:
        if _up_to_date(_existing(conn)):
            return
    with engine.connect() as conn:
        conn.exec_driver_sql("BEGIN IMMEDIATE")
        metadata.create_all(conn)
        _migrate(conn)
        conn.commit()


def _existing(conn: sqlalchemy.Connection) -> dict[str, set[str]]:
    """Every table in the database, with its columns."""
    inspector = sqlalchemy.inspect(conn)
    return {t: {c["name"] for c in inspector.get_columns(t)} for t in inspector.get_table_names()}


# Columns `runs` has gained since its first schema, with the DDL that adds each
# to a database that predates it.
_RUNS_COLUMNS = [
    ("name", "TEXT NOT NULL DEFAULT ''"),
    ("image", "TEXT NOT NULL DEFAULT ''"),
    ("compute", "TEXT NOT NULL DEFAULT 'cpu'"),
    ("config_hash", "TEXT"),
    ("created_at", "REAL NOT NULL DEFAULT 0"),
    ("sweep_id", "TEXT"),
    ("sweep_point", "TEXT"),
    ("queued_at", "REAL"),
    ("queued_from_phase", "TEXT"),
]


def _up_to_date(existing: dict[str, set[str]]) -> bool:
    if any(table not in existing for table in metadata.tables):
        return False
    columns = existing["runs"]
    return (
        "gpu" not in columns
        and "run_dir" not in columns
        and all(col in columns for col, _ in _RUNS_COLUMNS)
    )


def _migrate(conn: sqlalchemy.Connection) -> None:
    existing = _existing(conn)

    # Rename gpu column to compute if it exists and compute doesn't
    if "runs" in existing and "gpu" in existing["runs"] and "compute" not in existing["runs"]:
        conn.execute(sqlalchemy.text("ALTER TABLE runs RENAME COLUMN gpu TO compute"))
        # Refresh existing set after the rename
        existing["runs"].discard("gpu")
        existing["runs"].add("compute")

    # Add new columns to legacy 'runs' table if it came from the old schema
    if "runs" in existing:
        for col, ddl in _RUNS_COLUMNS:
            if col not in existing["runs"]:
                conn.execute(sqlalchemy.text(f"ALTER TABLE runs ADD COLUMN {col} {ddl}"))
        # Runs queued before `queued_at` existed wait in the order they were
        # created, as they did then.
        if "queued_at" not in existing["runs"]:
            conn.execute(
                sqlalchemy.text("UPDATE runs SET queued_at = created_at WHERE status = 'queued'")
            )

    # run_dir used to be stored as an absolute path, frozen at creation time,
    # which broke once the data directory was moved elsewhere. It's now
    # recomputed on every read from data_dir, so drop the stale column.
    if "runs" in existing and "run_dir" in existing["runs"]:
        try:
            conn.execute(sqlalchemy.text("ALTER TABLE runs DROP COLUMN run_dir"))
        except sqlalchemy.exc.OperationalError:
            pass  # SQLite < 3.35 can't drop columns; leave it, it's harmless.


def open_engine(settings: config.Settings) -> sqlalchemy.Engine:
    """An engine on the settings' database, with its schema brought up to date.

    A process that reads the database more than once -- the TUI, every second
    -- keeps one: each engine has its own pool and compiled-statement cache,
    and checks the schema when it is opened.
    """
    engine = create_engine(settings)
    init_db(engine)
    return engine


@contextlib.contextmanager
def session(
    engine: sqlalchemy.Engine, settings: config.Settings
) -> collections.abc.Generator[sqlalchemy.orm.Session, None, None]:
    """A session committed on success and rolled back on error."""
    with sqlalchemy.orm.Session(engine) as s:
        s.info["settings"] = settings
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise


@contextlib.contextmanager
def with_db(
    settings: config.Settings,
) -> collections.abc.Generator[sqlalchemy.orm.Session, None, None]:
    """A session on an engine of its own, for a process that opens just one."""
    with session(open_engine(settings), settings) as s:
        yield s


def resolve_run_id(prefix: str, session: sqlalchemy.orm.Session) -> str:
    rows = (
        session.execute(sqlalchemy.select(runs.c.id).where(runs.c.id.like(f"{prefix}%")))
        .scalars()
        .fetchall()
    )
    if not rows:
        raise exceptions.UI(f"run '{prefix}' not found")
    if len(rows) > 1:
        matches = ", ".join(str(r) for r in rows[:4])
        raise exceptions.UI(f"id prefix '{prefix}' is ambiguous (matches: {matches})")
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
        raise exceptions.UI(f"run '{run_id}' not found")
    return row


def run_image_ref(row: sqlalchemy.engine.RowMapping) -> str:
    """The podman image id a run is frozen to, checked against the store.

    A run stores the id of the image it was created against, so that re-tagging
    the image name later cannot silently move the run to different content:
    the config UI, the phase list and every container the run starts all
    resolve through here. An id that has left the local store is an error
    rather than a fall back to the name, which would quietly run the run
    against different content -- exactly what freezing exists to prevent.

    `podman run <bare hex id>` works anywhere `podman run <ref>` does, so the
    id can stand in for a reference in every container invocation.
    """
    run_id = str(row["id"])
    name = str(row["image"])
    image_id = str(row["image_id"])
    if not container.podman.image_exists(image_id):
        raise exceptions.UI(
            f"image '{name}' ({image_id[:12]}) frozen for run '{run_id}' is not in the local store"
        )
    return image_id


def describe_image(
    image_id: str, session: sqlalchemy.orm.Session
) -> container.schema.DescribeOutput:
    """An image's description: the stored one, or asked of the image and stored.

    For the paths that create runs and sweeps from an image, and so may be the
    first to meet it. Asking starts a container; every later reader takes the
    stored answer through `description`.
    """
    stored = _stored_description(image_id, session)
    if stored is not None:
        return stored
    described = container.podman.describe(image_id)
    session.execute(
        sqlalchemy.dialects.sqlite.insert(image_descriptions)
        .values(image_id=image_id, describe=described.model_dump_json())
        .on_conflict_do_nothing()
    )
    return described


def _stored_description(
    image_id: str, session: sqlalchemy.orm.Session
) -> container.schema.DescribeOutput | None:
    stored = session.execute(
        sqlalchemy.select(image_descriptions.c.describe).where(
            image_descriptions.c.image_id == image_id
        )
    ).scalar_one_or_none()
    if stored is None:
        return None
    return container.schema.DescribeOutput.model_validate_json(str(stored))


def description(image_id: str, session: sqlalchemy.orm.Session) -> container.schema.DescribeOutput:
    """An image's description, as stored when a run or sweep was created from it."""
    stored = _stored_description(image_id, session)
    if stored is None:
        raise exceptions.UI(f"no description of image {image_id[:12]} is stored; recreate the run")
    return stored


def run_description(
    row: sqlalchemy.engine.RowMapping, session: sqlalchemy.orm.Session
) -> container.schema.DescribeOutput:
    """The description of the image a run is frozen to."""
    return description(str(row["image_id"]), session)


def run_dir(run_id: str, session: sqlalchemy.orm.Session) -> pathlib.Path:
    """A run's directory, computed from the current data_dir rather than stored.

    Storing this as an absolute path used to break once the data directory
    was moved elsewhere; deriving it fresh keeps it correct after a move.
    """
    settings: config.Settings = session.info["settings"]
    return settings.runs_dir / run_id
