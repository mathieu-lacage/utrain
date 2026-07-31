import pathlib

import naw.rtsdb
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


def _find_rtsdb(run_dir: pathlib.Path, phase: str) -> pathlib.Path | None:
    phase_dir = run_dir / "wandb" / phase
    if not phase_dir.exists():
        return None
    files = list(phase_dir.glob("*.rtsdb"))
    return files[0] if files else None


def read_phase_events(run_dir: pathlib.Path, phase: str) -> list[PhaseEvent]:
    path = _find_rtsdb(run_dir, phase)
    if path is None or not path.exists():
        return []
    try:
        reader = naw.rtsdb.Reader(str(path))
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
                phase_part, _, event = event_raw.partition("/")
                if not event:
                    continue
                events.append(PhaseEvent(phase=phase_part, event=event, timestamp=float(ts_raw)))
            return events
        finally:
            reader.close()
    except Exception:
        return []


def read_metrics(
    run_dir: pathlib.Path,
    phase: str,
    name: str | None = None,
    since_rowid: int = 0,
) -> list[Metric]:
    path = _find_rtsdb(run_dir, phase)
    if path is None or not path.exists():
        return []
    try:
        reader = naw.rtsdb.Reader(str(path))
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
                    col_name = col.name
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
                            phase=phase,
                            name=col_name,
                            value=float(val_raw),
                        )
                    )
            return metrics
        finally:
            reader.close()
    except Exception:
        return []
