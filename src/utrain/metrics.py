"""Reads a phase's metrics straight out of its ``.rtsdb`` file.

Metrics never reach the database. The orchestrator shadows the image's ``wandb``
with a shim backed by ``naw`` (see ``orchestrator._wandb_mount_args``), so a
phase's time series lands on disk at
``<attempt_dir>/wandb/<phase>/<run_id>.rtsdb`` -- one row per commit, columns
``_step`` and ``_timestamp`` plus one per metric name.

Reading that file directly, rather than through
``container.run_data.read_metrics``, buys two things a dashboard needs:

- **Resumption.** A :class:`Tail` picks up where the last read stopped, so a
  live view appends new points instead of re-parsing the whole file every
  second. ``read_metrics`` always scans from zero and filters afterwards.
- **The column list.** ``row.schema.columns`` names every metric the phase has
  logged, which is what a metric picker offers the user. Getting it otherwise
  means a scan whose only purpose is to collect names.

Resuming needs the ``naw`` reader itself, not merely a byte offset: row schemas
are declared once, near the start of the file, and the reader accumulates them
as it goes. Handing a fresh reader a mid-file offset leaves it unable to decode
anything. So a :class:`Tail` holds the reader open for as long as the caller
wants to follow the phase -- which is exactly how ``naw watch`` does it.
"""

import dataclasses
import pathlib
import types
import typing

import naw.rtsdb

# The row schema `naw.wandb` writes metrics under; the same file also carries
# metadata and bookkeeping rows, which are not metrics.
_METRICS_SCHEMA = "metrics"

# Written for every row by `naw.wandb.Run`, and the two axes a caller can plot
# against without choosing a metric. Every other underscore-prefixed column is
# naw-internal.
_STEP = "_step"
_TIMESTAMP = "_timestamp"


@dataclasses.dataclass(frozen=True)
class MetricPoint:
    """One logged value, carrying both axes it can be plotted against."""

    step: int
    timestamp: float
    value: float


@dataclasses.dataclass(frozen=True)
class MetricUpdate:
    """What one read of a :class:`Tail` turned up.

    ``points`` is keyed by metric name and holds only what this batch saw, so a
    caller polling a live phase extends its own series rather than replacing
    them. Rows are sparse -- a phase may log ``loss`` every step and validation
    metrics every hundredth -- so each metric carries its own points and the
    lists are not row-aligned with each other.
    """

    # Metric names seen for the first time in this batch. The running total is
    # `Tail.columns`.
    columns: list[str]
    points: dict[str, list[MetricPoint]]

    @property
    def empty(self) -> bool:
        return not self.points


def find_rtsdb(attempt_dir: pathlib.Path, phase: str) -> pathlib.Path | None:
    """The ``.rtsdb`` a phase logs to, or None before it has written one."""
    phase_dir = attempt_dir / "wandb" / phase
    if not phase_dir.is_dir():
        return None
    # A phase re-run under a different wandb id leaves more than one file; the
    # newest is the one it is still writing to. The name breaks a tie, so that
    # two files of the same age resolve the same way from one refresh to the
    # next.
    files = sorted(phase_dir.glob("*.rtsdb"), key=lambda p: (p.stat().st_mtime, p.name))
    return files[-1] if files else None


class Tail:
    """An open cursor over one phase's metrics file.

    Each :meth:`read` returns only what has been appended since the last one.
    The file is opened lazily, so a Tail may be created for a phase that has not
    logged anything yet and will start producing points once it does.

    A partially written row is the normal case, not an error: ``naw``'s writer
    wraps an ordinary buffered file, so rows reach disk in buffer-sized batches
    and the tail of a live file is routinely a fragment. Such a row ends the
    batch and is re-read, whole, on a later call.

    There is deliberately no "the writer has finished" flag here. ``naw``
    settles ``Reader.closed_at`` when the reader is constructed and never
    revisits it, so a Tail opened while the phase was running would report
    ``False`` forever, and re-opening the reader to refresh it would throw away
    the accumulated schema state that makes resumption work at all. Whether a
    phase is still running is what ``types.PhaseDetail.status`` says.
    """

    def __init__(self, path: pathlib.Path) -> None:
        self._path = path
        self._reader: naw.rtsdb.Reader | None = None
        self._cursor = 0
        self._columns: list[str] = []

    @property
    def path(self) -> pathlib.Path:
        return self._path

    @property
    def columns(self) -> list[str]:
        """Every metric name seen so far, in the order first encountered."""
        return list(self._columns)

    def _open(self) -> naw.rtsdb.Reader | None:
        if self._reader is not None:
            return self._reader
        if not self._path.exists():
            return None
        try:
            self._reader = naw.rtsdb.Reader(str(self._path))
        except Exception:
            # Created but not yet a valid file: the writer has the header in its
            # buffer still. Nothing to do but try again next time.
            return None
        return self._reader

    def read(self) -> MetricUpdate:
        """Metric points appended since the previous read."""
        reader = self._open()
        if reader is None:
            return MetricUpdate(columns=[], points={})

        new_columns: list[str] = []
        points: dict[str, list[MetricPoint]] = {}

        try:
            for row in reader.read_rows(self._cursor):
                self._cursor = row.end_offset
                if row.schema.name != _METRICS_SCHEMA:
                    continue

                values = {col.name: val for col, val in zip(row.schema.columns, row.values)}
                step = values.get(_STEP)
                timestamp = values.get(_TIMESTAMP)
                if not isinstance(step, int) or not isinstance(timestamp, (int, float)):
                    continue

                for name, value in values.items():
                    if name.startswith("_") or not isinstance(value, (int, float)):
                        continue
                    if name not in self._columns:
                        self._columns.append(name)
                        new_columns.append(name)
                    points.setdefault(name, []).append(
                        MetricPoint(step=step, timestamp=float(timestamp), value=float(value))
                    )
        except Exception:
            # A half-written trailing row. naw raises FileTruncated when the
            # row's bytes are short, but a chunk boundary landing mid-header
            # trips an assert or a decode error deeper in instead, so the net
            # has to be wide.
            #
            # Deliberately not `run_data.read_metrics`'s blanket
            # `except: return []`: the points already read are good and the
            # cursor still marks the end of the last whole row. Discarding them
            # would make every live phase look empty.
            pass

        return MetricUpdate(columns=new_columns, points=points)

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def __enter__(self) -> "Tail":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: types.TracebackType | None,
    ) -> typing.Literal[False]:
        self.close()
        return False


def read_all(path: pathlib.Path) -> dict[str, list[MetricPoint]]:
    """Every point in a metrics file, for a caller that is not following it."""
    with Tail(path) as tail:
        return tail.read().points
