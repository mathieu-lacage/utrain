"""What the TUI keeps between sessions: the marked runs, comparisons, where you were.

One table of JSON values by key (`db.tui_state`), so that what is kept can
change without a migration each time. It is in the database rather than in a
file beside it because it refers to runs and sweeps by id, and the database is
where those are deleted: reading the tray drops ids that are gone, so a mark
never outlives its run.
"""

import json

import sqlalchemy
import sqlalchemy.orm

from . import db as dbmod

# The runs marked for comparison, in the order they were marked.
TRAY = "tray"
# The comparison Compare is showing: where its runs come from and how.
COMPARE = "compare"
# A named comparison is kept under this prefix and its name.
SAVED_PREFIX = "comparison:"
# Where the viewer was: the workspace, and each workspace's cursor.
SESSION = "session"


def get(session: sqlalchemy.orm.Session, key: str) -> object:
    """The value stored under `key`, or None."""
    row = session.execute(
        sqlalchemy.select(dbmod.tui_state.c.value).where(dbmod.tui_state.c.key == key)
    ).first()
    if row is None:
        return None
    try:
        return json.loads(str(row[0]))
    except ValueError:
        return None


def put(session: sqlalchemy.orm.Session, key: str, value: object) -> None:
    text = json.dumps(value)
    exists = session.execute(
        sqlalchemy.select(dbmod.tui_state.c.key).where(dbmod.tui_state.c.key == key)
    ).first()
    if exists is None:
        session.execute(sqlalchemy.insert(dbmod.tui_state).values(key=key, value=text))
    else:
        session.execute(
            sqlalchemy.update(dbmod.tui_state)
            .where(dbmod.tui_state.c.key == key)
            .values(value=text)
        )


def delete(session: sqlalchemy.orm.Session, key: str) -> None:
    session.execute(sqlalchemy.delete(dbmod.tui_state).where(dbmod.tui_state.c.key == key))


def _existing(session: sqlalchemy.orm.Session, run_ids: list[str]) -> list[str]:
    if not run_ids:
        return []
    found = {
        str(i)
        for i in session.execute(
            sqlalchemy.select(dbmod.runs.c.id).where(dbmod.runs.c.id.in_(run_ids))
        )
        .scalars()
        .fetchall()
    }
    return [r for r in run_ids if r in found]


def tray(session: sqlalchemy.orm.Session) -> list[str]:
    """The marked runs that still exist, in the order they were marked."""
    value = get(session, TRAY)
    ids = [str(v) for v in value] if isinstance(value, list) else []
    return _existing(session, ids)


def set_tray(session: sqlalchemy.orm.Session, run_ids: list[str]) -> None:
    put(session, TRAY, list(dict.fromkeys(run_ids)))


def toggle_mark(session: sqlalchemy.orm.Session, run_id: str) -> bool:
    """Mark a run, or unmark it if it was. Whether it is marked now."""
    ids = tray(session)
    if run_id in ids:
        ids.remove(run_id)
        marked = False
    else:
        ids.append(run_id)
        marked = True
    set_tray(session, ids)
    return marked


def saved_names(session: sqlalchemy.orm.Session) -> list[str]:
    """The names comparisons were saved under, alphabetically."""
    keys = (
        session.execute(
            sqlalchemy.select(dbmod.tui_state.c.key).where(
                dbmod.tui_state.c.key.like(f"{SAVED_PREFIX}%")
            )
        )
        .scalars()
        .fetchall()
    )
    return sorted(str(k).removeprefix(SAVED_PREFIX) for k in keys)
