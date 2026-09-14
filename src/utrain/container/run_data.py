import pathlib

import naw.rtsdb
import pydantic

from .. import metrics


class Metric(pydantic.BaseModel):
    rowid: int
    step: int
    timestamp: float
    phase: str
    name: str
    value: float


def read_metrics(
    run_dir: pathlib.Path,
    phase: str,
    name: str | None = None,
    since_rowid: int = 0,
) -> list[Metric]:
    path = metrics.find_rtsdb(run_dir, phase)
    if path is None or not path.exists():
        return []
    try:
        reader = naw.rtsdb.Reader(str(path))
        try:
            found: list[Metric] = []
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
                    found.append(
                        Metric(
                            rowid=row.end_offset,
                            step=step,
                            timestamp=ts,
                            phase=phase,
                            name=col_name,
                            value=float(val_raw),
                        )
                    )
            return found
        finally:
            reader.close()
    except Exception:
        return []
