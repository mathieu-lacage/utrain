import pathlib

import baw.rtsdb
import pydantic


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


def _find_rtsdb(run_dir: pathlib.Path) -> pathlib.Path | None:
    wandb_dir = run_dir / "wandb"
    if not wandb_dir.exists():
        return None
    files = list(wandb_dir.rglob("*.rtsdb"))
    return files[0] if files else None


def read_phase_events(run_dir: pathlib.Path) -> list[PhaseEvent]:
    path = _find_rtsdb(run_dir)
    if path is None or not path.exists():
        return []
    try:
        reader = baw.rtsdb.Reader(str(path))
        try:
            events: list[PhaseEvent] = []
            for row in reader.read_rows(0):
                col_names = [c.name for c in row.schema.columns]
                if "_phase_event" not in col_names or "_timestamp" not in col_names:
                    continue
                event_raw = row.values[col_names.index("_phase_event")]
                ts_raw = row.values[col_names.index("_timestamp")]
                if not isinstance(event_raw, str) or not isinstance(ts_raw, (int, float)):
                    continue
                phase, _, event = event_raw.partition("/")
                if not event:
                    continue
                events.append(PhaseEvent(phase=phase, event=event, timestamp=float(ts_raw)))
            return events
        finally:
            reader.close()
    except Exception:
        return []


def read_metrics(
    run_dir: pathlib.Path,
    phase: str | None = None,
    name: str | None = None,
    since_rowid: int = 0,
) -> list[Metric]:
    path = _find_rtsdb(run_dir)
    if path is None or not path.exists():
        return []
    try:
        reader = baw.rtsdb.Reader(str(path))
        try:
            metrics: list[Metric] = []
            for row in reader.read_rows(0):
                if row.end_offset <= since_rowid:
                    continue
                col_names = [c.name for c in row.schema.columns]
                if "_phase_event" in col_names or "_step" not in col_names:
                    continue
                if "_timestamp" not in col_names:
                    continue
                step_raw = row.values[col_names.index("_step")]
                ts_raw = row.values[col_names.index("_timestamp")]
                if not isinstance(step_raw, int) or not isinstance(ts_raw, (int, float)):
                    continue
                step = step_raw
                ts = float(ts_raw)
                for col in row.schema.columns:
                    if col.name.startswith("_"):
                        continue
                    col_phase, _, col_name = col.name.partition("/")
                    if not col_name:
                        continue
                    if phase is not None and col_phase != phase:
                        continue
                    if name is not None and col_name != name:
                        continue
                    val_raw = row.values[col_names.index(col.name)]
                    if not isinstance(val_raw, (int, float)):
                        continue
                    metrics.append(
                        Metric(
                            rowid=row.end_offset,
                            step=step,
                            timestamp=ts,
                            phase=col_phase,
                            name=col_name,
                            value=float(val_raw),
                        )
                    )
            return metrics
        finally:
            reader.close()
    except Exception:
        return []
