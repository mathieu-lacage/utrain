"""The TUI's single point of contact with the query layer.

Two things live here that the widgets must not have to think about.

**Sessions are per-fetch.** Every method opens a `db.with_db` session, uses it,
and closes it, the way `cli/main._cmd_run_delete` does. A long-lived session
would hold a SQLite connection for the life of the app against a detached
orchestrator that is writing, and they share a 5s busy timeout.

**Describe is cached.** `phases.list_phases` and `phases.show_phase` both call
`container.podman.describe`, which is `podman run --rm <image> describe` -- it
starts a container -- and `container.podman.list_presets`, which shells out to
`podman images`. Neither is cached in the query layer, which is right for a CLI
that runs one command and exits, and unusable for a view that refreshes every
second: it would launch a container per frame. An image's phase list does not
change under a running app, so it is fetched once per image and kept.

The cache is held here rather than in `container.podman` deliberately. A global
memo there would change what the CLI does and would leave a long-lived process
believing a stale answer after an image was rebuilt; here it is scoped to one
app, and `refresh()` drops it.
"""

import dataclasses
import pathlib
import time

from .. import compute, config, exceptions, images, metrics, phases, runs, serve, types
from .. import container as containermod
from .. import db as dbmod

# How much of a phase's output each refresh reads. `logs.tail_lines` seeks
# backwards from the end in blocks, so a deep tail costs no more to read than a
# shallow one. The log pane keeps what it has been shown -- it writes only the
# lines that grew onto the previous tail -- so this is the history a viewer has
# the moment a phase is selected; from then on the pane accumulates.
_LOG_LINES = 5000

# `podman images` is cheap next to `describe`, but it still forks. The preset
# list only changes when someone adds or removes an image, so a few seconds of
# staleness costs nothing and takes it off the refresh path.
_PRESETS_TTL_SECONDS = 5.0


@dataclasses.dataclass(frozen=True)
class Snapshot:
    """One tick of the whole view.

    The layout shows runs, phases, metrics, a plot and a log at once, so the
    alternative is five workers refreshing five panes on their own schedules and
    a screen that is briefly describing two different moments. One fetch, one
    apply, one moment.
    """

    now: float
    # The run and the phase this tick was fetched for. The cursor can move
    # between the fetch and the apply -- and a thread worker cannot be
    # interrupted, so `exclusive` stops the next fetch starting, not this one
    # landing -- and neither the panes of one run nor the points of one phase
    # may be filed under another.
    run_id: str | None
    address: str | None
    runs: list[types.RunRow]
    run: types.RunDetail | None
    phases: list[types.PhaseListEntry]
    # The selected run's config schema and the values currently on disk. Read
    # every tick because the file is a few hundred bytes; `ConfigPane` decides
    # for itself when to re-seed the widgets from them.
    config_schema: containermod.schema.ConfigSchema | None
    config: dict[str, object]
    phase_order: list[str]
    # Status and timings of the selected phase, taken from `phases`; there is
    # deliberately no `show_phase` here, because it rescans the whole metrics
    # file to recompute last values and this view already has them from the
    # Tail.
    phase: types.PhaseListEntry | None
    phase_label: str
    # The plots the selected phase declares in `describe`, in the order it
    # names them. Empty when it names none, which is the all-metrics dashboard.
    phase_plots: list[containermod.schema.PlotSpec]
    log: list[str]
    update: metrics.MetricUpdate
    # Opened by the fetch when the phase had not logged anything yet, so the
    # screen can adopt it; the same object it was handed otherwise.
    tail: metrics.Tail | None
    # The metrics file `update` was read from, or None when the phase has not
    # logged one (yet). The screen compares it with the tick before to notice
    # the file moving under a held Tail -- a restart starts a new attempt
    # directory, and the address the screen keys its series by follows the
    # latest attempt -- and starts those series over.
    metrics_path: pathlib.Path | None


def _address(run_id: str, phase: str | None, attempt: int | None) -> str | None:
    """A phase address, as the query layer parses it.

    Three-part when an attempt is named; the two-part form means the latest
    attempt. The screen names the phase the same way, and compares the two to
    tell a snapshot that is still about the selected phase from one that is not.
    """
    if phase is None:
        return None
    if attempt is None:
        return f"{run_id}/{phase}"
    return f"{run_id}/{attempt}/{phase}"


def current_phase(phases: list[types.PhaseListEntry]) -> str | None:
    """Where a run has got to: the phase running, or the last one that ran.

    What the content column is about while the cursor is still on the run
    itself, one level up from the phases. It is what keeps the runs list a live
    monitor -- scanning it shows each run's current curve and current log --
    rather than a menu on the way to them.
    """
    if not phases:
        return None
    for entry in phases:
        if entry.status == "running":
            return entry.phase
    started = [entry for entry in phases if entry.status not in (None, "pending")]
    return (started[-1] if started else phases[0]).phase


def phase_plots(
    described: containermod.schema.DescribeOutput | None, phase: str | None
) -> list[containermod.schema.PlotSpec]:
    """The plots an image asks for on one phase, or none.

    Takes the `DescribeOutput` rather than an image name so that the caller says
    where it came from: the snapshot has just fetched one, and the screen has
    only what `Data.described` already knows -- and must not start a container
    on the message loop to learn more.

    Not checked against what the phase has actually logged: a live phase may not
    have reached the metric yet, and a plot with nothing in it draws as empty
    rather than as an error.
    """
    if described is None or phase is None:
        return []
    for info in described.phases:
        if info.name == phase:
            return list(info.plots)
    return []


def _snapshot_phase(
    entries: list[types.PhaseListEntry], phase: str | None
) -> types.PhaseListEntry | None:
    for entry in entries:
        if entry.phase == phase:
            return entry
    return None


@dataclasses.dataclass(frozen=True)
class NewRunChoices:
    """The two lists the new-run dialog picks from.

    `compute` is handed over whole rather than as strings: which specs are valid
    is `runs._resolve_compute`'s business, and how they are labelled is
    `render.compute_options`'.
    """

    images: list[str]
    compute: compute.ComputeInfo


class Data:
    """Typed reads for the TUI, with the podman work cached.

    ``describe_cache`` is injectable so tests can seed it: with it primed,
    nothing in the phase or run screens shells out, which is what lets the TUI
    tests run in CI. The cram suite cannot -- CI has no podman at all.
    """

    def __init__(
        self,
        settings: config.Settings | None = None,
        describe_cache: dict[str, containermod.schema.DescribeOutput] | None = None,
    ) -> None:
        self._settings = settings if settings is not None else config.Settings()
        self._describe: dict[str, containermod.schema.DescribeOutput] = (
            describe_cache if describe_cache is not None else {}
        )
        self._presets: dict[str, str] = {}
        self._presets_at = 0.0

    @property
    def settings(self) -> config.Settings:
        return self._settings

    def refresh(self) -> None:
        """Drop the caches, so the next read reflects a rebuilt image."""
        self._describe.clear()
        self._presets = {}
        self._presets_at = 0.0

    def presets(self) -> dict[str, str]:
        now = time.monotonic()
        if not self._presets or now - self._presets_at > _PRESETS_TTL_SECONDS:
            self._presets = containermod.podman.list_presets()
            self._presets_at = now
        return self._presets

    def described(self, ref: str) -> containermod.schema.DescribeOutput | None:
        """What is already known about an image, or None -- never a container.

        `describe` shells out on a miss, which is right everywhere it is called
        from a worker and wrong on the message loop: `check_action` runs there,
        once per key and once per footer rebuild, and a `podman run` from it
        would freeze the app. The cache is warmed by every snapshot fetch, so
        the answer is at worst one tick late.
        """
        return self._describe.get(ref)

    def describe(self, ref: str) -> containermod.schema.DescribeOutput:
        """An image's phase list, fetched at most once per image per app."""
        cached = self._describe.get(ref)
        if cached is not None:
            return cached
        described = phases.describe_image(ref)
        self._describe[ref] = described
        return described

    def image_ref(self, run: types.RunRow) -> str:
        """What a run's image resolves to, without touching podman or the db.

        The frozen id when the run has one, the bare preset key otherwise --
        `phases.describe_image` resolves a bare key through the preset list,
        exactly as every run did before ids were frozen, so a legacy run is
        described (and cached) under its name as it always was.
        """
        if run.image_id is not None:
            return run.image_id
        return run.image

    def phase_order(self, ref: str) -> list[str]:
        return list(self.describe(ref).phase_order)

    def phase_label(self, ref: str, phase: str) -> str:
        for info in self.describe(ref).phases:
            if info.name == phase:
                return f"{phase} ({info.label})"
        return phase

    # -- runs -------------------------------------------------------------

    def list_runs(self) -> list[types.RunRow]:
        with dbmod.with_db(self._settings) as session:
            return runs.list_runs(session)

    def run_detail(self, run_id: str) -> types.RunDetail:
        with dbmod.with_db(self._settings) as session:
            return runs.get_run_detail(run_id, session)

    # -- phases -----------------------------------------------------------

    def list_phases(self, run_id: str) -> list[types.PhaseListEntry]:
        with dbmod.with_db(self._settings) as session:
            run = runs.get_run(run_id, session)
            return phases.list_phases(run_id, session, self.describe(self.image_ref(run)))

    def phase_detail(self, addr: str) -> types.PhaseDetail:
        with dbmod.with_db(self._settings) as session:
            run = runs.get_run(addr.split("/")[0], session)
            return phases.show_phase(addr, session, self.describe(self.image_ref(run)))

    def phase_log_tail(self, detail: types.PhaseDetail, n: int) -> list[str]:
        return phases.read_log_tail(detail, n)

    def write_run_config(self, run_id: str, values: dict[str, object]) -> None:
        with dbmod.with_db(self._settings) as session:
            run = runs.get_run(run_id, session)
            runs.write_config(run_id, values, session, self.describe(self.image_ref(run)))
            session.commit()

    # -- lifecycle --------------------------------------------------------
    #
    # The only writes here besides the config. Both take a full run id: the TUI
    # always has one, and resolving a prefix is a CLI concern.

    def start_run(self, run_id: str) -> None:
        with dbmod.with_db(self._settings) as session:
            runs.start_run(run_id, session)

    def stop_run(self, run_id: str) -> None:
        with dbmod.with_db(self._settings) as session:
            runs.stop_run(run_id, session)

    def restart_run(self, run_id: str, from_phase: str | None) -> None:
        with dbmod.with_db(self._settings) as session:
            runs.restart_run(run_id, from_phase, session)

    def delete_run(self, run_id: str) -> None:
        """Never forced: the TUI greys `d` out on a running run rather than
        killing an attempt behind a delete prompt. Stopping is `S`, and it asks."""
        with dbmod.with_db(self._settings) as session:
            runs.delete_run(run_id, force=False, session=session)

    def start_server(self, run_id: str, phase: str | None = None) -> serve.Server:
        """Bring up a serve container for one phase's snapshot of a run.

        The session is closed on the way out, and `serve.start` commits before
        it returns: a chat lasts as long as the viewer keeps typing, and
        neither the screen underneath nor a detached orchestrator can be made
        to wait that long for the SQLite write lock.

        The container is spawned but not yet listening -- `Server.wait_for_port`
        is the second half, and is the caller's to wait on.
        """
        with dbmod.with_db(self._settings) as session:
            return serve.start(run_id, session, phase)

    def create_run(self, name: str, image: str, compute_spec: str) -> str:
        with dbmod.with_db(self._settings) as session:
            return runs.create_run(name, image, compute_spec, self._settings, session)

    def new_run_choices(self) -> NewRunChoices:
        """What the new-run dialog offers. Both halves fork -- `podman images`
        and `nvidia-smi` -- so this is read in a worker before the modal opens,
        not while it is composing."""
        return NewRunChoices(images=sorted(self.presets()), compute=compute.collect_compute())

    def metrics_tail(self, addr: str) -> metrics.Tail | None:
        """An open cursor over a phase's metrics, or None if it has logged none.

        The Tail is handed to the caller to keep: following a live phase means
        one reader held open across refreshes, not a fresh read each time.
        """
        with dbmod.with_db(self._settings) as session:
            path = phases.metrics_path(addr, session)
        return None if path is None else metrics.Tail(path)

    def snapshot(
        self,
        run_id: str | None,
        phase: str | None,
        attempt: int | None = None,
        tail: metrics.Tail | None = None,
        log_lines: int = _LOG_LINES,
    ) -> Snapshot:
        """Everything the main screen shows, read through one session.

        Tolerant by section: a run with no attempts yet has no phases, and a
        pending phase has no log file. Neither is an error the viewer needs to
        be told about, and raising would blank the panes that did have an
        answer.

        `attempt` names where the phase's output is, for a phase a
        `--from-phase` restart inherited; None means the run's current attempt.
        """
        now = time.time()
        with dbmod.with_db(self._settings) as session:
            rows = runs.list_runs(session)
            if run_id is None:
                return Snapshot(
                    now=now,
                    run_id=None,
                    address=None,
                    runs=rows,
                    run=None,
                    phases=[],
                    config_schema=None,
                    config={},
                    phase_order=[],
                    phase=None,
                    phase_label="",
                    phase_plots=[],
                    log=[],
                    update=metrics.MetricUpdate(columns=[], points={}),
                    tail=tail,
                    metrics_path=None,
                )

            detail = runs.get_run_detail(run_id, session)
            described = self.describe(self.image_ref(detail.run))
            config = runs.read_config(run_id, session)
            try:
                entries = phases.list_phases(run_id, session, described)
            except exceptions.UI:
                entries = []

            log_attempt = attempt if attempt is not None else detail.run.attempt
            address = _address(run_id, phase, attempt)

            log: list[str] = []
            if phase is not None and log_attempt is not None:
                try:
                    log = runs.read_log_tail(run_id, session, log_attempt, phase, tail=log_lines)
                except exceptions.UI:
                    log = []

            metrics_path: pathlib.Path | None = None
            if address is not None:
                try:
                    metrics_path = phases.metrics_path(address, session)
                except exceptions.UI:
                    metrics_path = None
                if tail is not None and tail.path != metrics_path:
                    # The tail follows a file that is no longer this phase's:
                    # a restart has put the phase's output in a new attempt
                    # directory, or this one has not written a metrics file
                    # yet. The held tail would read the old file to the end of
                    # time -- appending nothing, so the plots would freeze at
                    # whatever the previous attempt logged -- so it is dropped
                    # here and a fresh one opened on the current path. The
                    # screen closes the tail it holds when it adopts the
                    # replacement, and `Snapshot.metrics_path` is what tells it
                    # to start its series over.
                    tail = None
                if tail is None and metrics_path is not None:
                    tail = metrics.Tail(metrics_path)

        update = tail.read() if tail is not None else metrics.MetricUpdate(columns=[], points={})
        return Snapshot(
            now=now,
            run_id=run_id,
            address=address,
            runs=rows,
            run=detail,
            phases=entries,
            config_schema=described.config_schema,
            config=config,
            phase_order=list(described.phase_order),
            phase=_snapshot_phase(entries, phase),
            phase_label=""
            if phase is None
            else self.phase_label(self.image_ref(detail.run), phase),
            phase_plots=phase_plots(described, phase),
            log=log,
            update=update,
            tail=tail,
            metrics_path=metrics_path,
        )

    # -- the rest of the CLI's read surface -------------------------------

    def list_images(self) -> list[images.ImageInfo]:
        with dbmod.with_db(self._settings) as session:
            return images.list_images(session)

    def add_image(self, url: str) -> str:
        """Pull an image and tag it as a preset, returning the preset's name.

        Quiet, because podman would otherwise pull onto the screen the app is
        drawing on. The caches go with it: `presets` has a few seconds of TTL
        and would come round on its own, but the whole point of pressing the
        key is to see the image appear, and `refresh` is what the screen's own
        `r` does.
        """
        name = images.add_image(url, quiet=True)
        self.refresh()
        return name

    def compute(self) -> compute.ComputeInfo:
        return compute.collect_compute()
