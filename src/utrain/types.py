"""Result types returned by the query layer.

These exist so that a caller -- the CLI today, a TUI next -- gets typed data
rather than a SQLAlchemy ``RowMapping``, whose ``row["name"]`` subscripts are
all typed ``Any``.

Two conventions matter for anything rendering these:

- timestamps stay raw epoch floats. Formatting is the caller's job, because a
  table wants "2026-08-22 14:03" and a TUI wants "3m ago".
- run ids stay full length. Truncating to a unique prefix depends on the set
  being displayed, so it belongs to whoever displays it.
"""

import dataclasses
import pathlib


@dataclasses.dataclass(frozen=True)
class RunRow:
    """One run, as it appears in a list."""

    id: str
    name: str
    image: str
    # The podman image id frozen when the run was created; what the run keeps
    # pointing at however the image's name is re-tagged later.
    image_id: str
    compute: str
    status: str
    created_at: float
    attempt: int | None
    # The first phase of the latest attempt that is not done/stopped/failed,
    # i.e. what the run is currently working on. None when nothing is pending.
    phase: str | None
    # The sweep that generated the run, and the point of its grid the run
    # stands for (axis path to value). None and empty for a run of its own.
    sweep_id: str | None = None
    point: dict[str, object] = dataclasses.field(default_factory=dict[str, object])


@dataclasses.dataclass(frozen=True)
class SweepCounts:
    """How a sweep's runs stand, by status."""

    total: int
    queued: int
    running: int
    done: int
    failed: int
    stopped: int

    @property
    def finished(self) -> int:
        """Runs that will not run again unless asked: done, failed or stopped."""
        return self.done + self.failed + self.stopped


@dataclasses.dataclass(frozen=True)
class SweepRow:
    """One sweep, as it appears in a list."""

    id: str
    name: str
    image: str
    image_id: str
    # draft, running, paused, done or cancelled. `done` is derived: a sweep
    # that was started and has no run left queued or running.
    status: str
    created_at: float
    # Axis path to the values it takes, in grid order.
    axes: dict[str, list[object]]
    # The axes whose runs are aggregated rather than told apart when compared.
    replicate: list[str]
    compute: list[str]
    # The run the sweep's config was copied from, if any.
    base: str | None
    counts: SweepCounts


@dataclasses.dataclass(frozen=True)
class SweepDetail:
    sweep: SweepRow
    # In grid order, which is creation order.
    runs: list[RunRow]


@dataclasses.dataclass(frozen=True)
class PhaseRow:
    phase: str
    phase_order: int
    status: str
    started_at: float | None
    ended_at: float | None
    # ``<short run id>/<attempt>/<phase>``: how this phase is addressed on the
    # command line (`phase show`, `phase restart`, ...). Kept beside the bare
    # name because the short form depends on the other runs in the database,
    # and the attempt because a `--from-phase` restart leaves earlier phases'
    # outputs under earlier attempts' directories.
    address: str


@dataclasses.dataclass(frozen=True)
class RunDetail:
    run: RunRow
    run_dir: pathlib.Path
    n_attempts: int
    # Status of the newest attempt, which is not the same as the run's own
    # status; falls back to the run status when there are no attempts.
    latest_attempt_status: str
    phases: list[PhaseRow]

    @property
    def config_path(self) -> pathlib.Path:
        return self.run_dir / "config.yaml"

    @property
    def logs_dir(self) -> pathlib.Path | None:
        if self.run.attempt is None:
            return None
        return self.run_dir / "attempt" / str(self.run.attempt) / "logs"


@dataclasses.dataclass(frozen=True)
class AttemptRow:
    run_id: str
    attempt: int
    # ``<short run id>/<attempt>``: how this attempt is named on the command
    # line. Kept beside the full run_id because the short form depends on the
    # other runs in the database.
    address: str
    from_phase: str | None
    status: str
    started_at: float | None
    ended_at: float | None


@dataclasses.dataclass(frozen=True)
class AttemptDetail:
    run_id: str
    run_name: str
    attempt: AttemptRow
    phases: list[PhaseRow]
    logs_dir: pathlib.Path
    data_dir: pathlib.Path


@dataclasses.dataclass(frozen=True)
class PhaseListEntry:
    """A phase of an attempt, including ones inherited from an earlier attempt.

    A ``--from-phase`` restart does not re-run the phases before it, so those
    carry the status they finished with in the attempt that did run them, and
    ``inherited_from`` names that attempt. It is what a caller reading the
    phase's log or metrics has to address them under: this attempt's directory
    has nothing in it for a phase it skipped.
    """

    phase: str
    phase_order: int
    address: str
    status: str | None
    started_at: float | None
    ended_at: float | None
    inherited_from: int | None


@dataclasses.dataclass(frozen=True)
class Metric:
    name: str
    step: int
    value: float


@dataclasses.dataclass(frozen=True)
class PhaseDetail:
    run_id: str
    run_name: str
    attempt: int
    phase: str
    # ``<phase> (<label>)`` when the image describes a label for it.
    phase_label: str
    status: str
    started_at: float | None
    ended_at: float | None
    # Last value seen for each metric name, sorted by name.
    last_metrics: list[Metric]
    log_file: pathlib.Path | None


@dataclasses.dataclass(frozen=True)
class StoreSummary:
    """The content-addressed store's size, and what `store gc` would reclaim."""

    files: int
    bytes: int
    orphaned: int
    orphaned_bytes: int


@dataclasses.dataclass(frozen=True)
class GcResult:
    removed: int
    reclaimed_bytes: int


@dataclasses.dataclass(frozen=True)
class StoreProblem:
    """One way a phase's data diverges from the content-addressed store.

    ``path`` is relative to the data directory: ``store/<sha>`` for a store
    file, ``runs/<run_id>/attempt/<n>/data/...`` for a phase's data file.
    """

    path: pathlib.Path
    problem: str


@dataclasses.dataclass(frozen=True)
class StoreLink:
    """One checked data file and the store file it is hardlinked to.

    Both paths are relative to the data directory, as in ``StoreProblem``.
    """

    data: pathlib.Path
    store: pathlib.Path


@dataclasses.dataclass(frozen=True)
class StoreCheckResult:
    """What `utrain store check` found.

    Only attempts recorded as done are checked: deduplication runs once the
    whole run completes, so anything still going has legitimately unlinked
    data.
    """

    # Done attempts whose data dir was examined.
    attempts: int
    data_files: int
    store_files: int
    # Store files nothing hardlinks any more; `utrain store gc` reclaims them.
    orphaned: int
    problems: list[StoreProblem]
    # Every checked file beside its store file; empty unless the caller asked
    # for verbose output, because it is one entry per data file.
    links: list[StoreLink] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class ChatMessage:
    """One turn of a chat conversation.

    The wire format is OpenAI's ``{"role": ..., "content": ...}``; this exists
    so that a caller holding a conversation is holding typed turns rather than
    dicts whose every subscript is ``Any``. ``utrain.chat.Client`` is what
    converts between the two.
    """

    role: str
    content: str
