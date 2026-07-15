import pathlib

import pydantic
import sqlalchemy
import sqlalchemy.exc


class PhaseEvent(pydantic.BaseModel):
    phase: str
    event: str  # started | completed | failed
    timestamp: float


class Metric(pydantic.BaseModel):
    rowid: int
    step: int
    timestamp: float
    phase: str
    name: str
    value: float


_metadata = sqlalchemy.MetaData()

_phase_events = sqlalchemy.Table(
    "phase_events",
    _metadata,
    sqlalchemy.Column("phase", sqlalchemy.Text),
    sqlalchemy.Column("event", sqlalchemy.Text),
    sqlalchemy.Column("timestamp", sqlalchemy.Float),
)

_metrics = sqlalchemy.Table(
    "metrics",
    _metadata,
    sqlalchemy.Column("rowid", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("step", sqlalchemy.Integer),
    sqlalchemy.Column("timestamp", sqlalchemy.Float),
    sqlalchemy.Column("phase", sqlalchemy.Text),
    sqlalchemy.Column("name", sqlalchemy.Text),
    sqlalchemy.Column("value", sqlalchemy.Float),
)


def _engine(metrics_db: pathlib.Path) -> sqlalchemy.Engine:
    return sqlalchemy.create_engine(f"sqlite+pysqlite:///file:{metrics_db}?mode=ro&uri=true")


def read_phase_events(metrics_db: pathlib.Path) -> list[PhaseEvent]:
    if not metrics_db.exists():
        return []
    try:
        with _engine(metrics_db).connect() as conn:
            rows = (
                conn.execute(sqlalchemy.select(_phase_events).order_by(_phase_events.c.timestamp))
                .mappings()
                .fetchall()
            )
        return [
            PhaseEvent(phase=r["phase"], event=r["event"], timestamp=r["timestamp"]) for r in rows
        ]
    except sqlalchemy.exc.OperationalError:
        return []


def read_metrics(
    metrics_db: pathlib.Path,
    phase: str | None = None,
    name: str | None = None,
    since_rowid: int = 0,
) -> list[Metric]:
    if not metrics_db.exists():
        return []
    try:
        with _engine(metrics_db).connect() as conn:
            query = sqlalchemy.select(_metrics).where(_metrics.c.rowid > since_rowid)
            if phase is not None:
                query = query.where(_metrics.c.phase == phase)
            if name is not None:
                query = query.where(_metrics.c.name == name)
            query = query.order_by(_metrics.c.rowid)
            rows = conn.execute(query).mappings().fetchall()
        return [
            Metric(
                rowid=r["rowid"],
                step=r["step"],
                timestamp=r["timestamp"],
                phase=r["phase"],
                name=r["name"],
                value=r["value"],
            )
            for r in rows
        ]
    except sqlalchemy.exc.OperationalError:
        return []
