"""Pilot tests: the TUI driven by keypresses, against a seeded database.

The describe cache is seeded, so nothing here shells out to podman. That is
what lets these run in CI -- every cram test skips there for want of podman, so
this is the only end-to-end coverage the UI layer gets on that machine.

Assertions are on widget state rather than on rendered output. uniplot
rasterises to a character grid sized from the widget, so a snapshot would
encode the terminal size and break on any layout change.
"""

import collections.abc
import dataclasses
import os
import pathlib
import subprocess
import time
import typing

import naw.wandb
import pytest
import sqlalchemy
import textual.coordinate
import textual.widgets
import textual.worker
import uniplot

import utrain.compute
import utrain.config
import utrain.container.podman
import utrain.container.schema
import utrain.db
import utrain.exceptions
import utrain.images
import utrain.metrics
import utrain.runs
import utrain.serve
import utrain.tui.app
import utrain.tui.data
import utrain.tui.render
import utrain.tui.screens
import utrain.tui.widgets

# Driven through anyio's pytest plugin rather than pytest-asyncio: anyio is
# already a runtime dependency, so this needs nothing new installed.
pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


def _write_metrics(
    attempt_dir: pathlib.Path,
    phase: str,
    run_id: str,
    rows: list[dict[str, float]],
) -> None:
    """A phase's metrics, laid out the way the per-phase mount leaves them.

    `naw.wandb` always inserts a literal ``wandb`` segment; under utrain the
    orchestrator's bind mount makes that segment the phase's own directory. The
    rename stands in for the mount. The project name is not the phase, because
    utrain neither chooses nor reads it.
    """
    staging = attempt_dir / f".staging-{phase}"
    run = naw.wandb.init(project="a-project", id=run_id, dir=str(staging))
    for step, row in enumerate(rows):
        run.log(dict(row), step=step, commit=True)
    run.finish()

    phase_dir = attempt_dir / "wandb" / phase
    phase_dir.parent.mkdir(parents=True, exist_ok=True)
    (staging / "wandb").rename(phase_dir)
    staging.rmdir()


# The real `refresh_soon`, kept because the fixture below replaces it and the
# tests that are about the settling put it back.
_REFRESH_SOON = utrain.tui.screens.MainScreen.refresh_soon


@pytest.fixture(autouse=True)
def no_settle_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fetch on the cursor move rather than 150ms after it stops.

    `_settle` waits for workers to finish, and a timer is not a worker, so the
    real delay would leave every test here asserting on a fetch that had not
    started. Zeroing `_SETTLE_SECONDS` is not the way -- Textual's `Timer`
    divides by its interval -- so the deferral is removed instead of shortened.
    """
    monkeypatch.setattr(
        utrain.tui.screens.MainScreen,
        "refresh_soon",
        utrain.tui.screens.MainScreen.refresh_data,
    )


RUN_ID = "a" * 32
DRAFT_ID = "b" * 32
LIVE_ID = "c" * 32
IMAGE = "utrain-fake"
# The podman image id every seeded run is frozen to; the describe cache is keyed
# by it, since that is what a run's image resolves to.
IMAGE_ID = "f" * 64

# Big enough that the sidebar, the plots and the log all lay out.
SIZE = (140, 45)


def _describe() -> utrain.container.schema.DescribeOutput:
    return utrain.container.schema.DescribeOutput(
        name=IMAGE,
        phases=[
            utrain.container.schema.PhaseInfo(name="tokenizer", label="Build Vocabulary"),
            # `MainScreen.chattable` reads `can_serve` to decide whether `t` is
            # a key at all. Only `pretrain` leaves a model behind, as in the
            # real container, so `t` is greyed on `tokenizer`.
            utrain.container.schema.PhaseInfo(
                name="pretrain", label="Train Character LM", can_serve=True
            ),
        ],
        phase_order=["tokenizer", "pretrain"],
        config_schema=utrain.container.schema.ConfigSchema(
            globals=utrain.container.schema.GlobalConfigSchema(
                groups=[
                    utrain.container.schema.FieldGroup(
                        name="model",
                        label="Model",
                        fields=[
                            utrain.container.schema.FieldSchema(
                                key="n_layer", label="Layers", type="int", default=4, min=1, max=48
                            ),
                            utrain.container.schema.FieldSchema(
                                key="dtype",
                                label="Precision",
                                type="enum",
                                default="fp32",
                                options=["fp32", "bf16"],
                            ),
                        ],
                    )
                ]
            ),
            phases={
                "pretrain": utrain.container.schema.PhaseConfigSchema(
                    groups=[
                        utrain.container.schema.FieldGroup(
                            name="optim",
                            label="Optimiser",
                            fields=[
                                utrain.container.schema.FieldSchema(
                                    key="lr", label="Learning rate", type="float", default=0.001
                                ),
                                utrain.container.schema.FieldSchema(
                                    key="resume", label="Resume", type="bool", default=False
                                ),
                            ],
                        )
                    ]
                )
            },
        ),
    )


def _seed(data_dir: pathlib.Path) -> None:
    """A finished run with two phases, the second having logged metrics."""
    run_dir = data_dir / "runs" / RUN_ID
    attempt_dir = run_dir / "attempt" / "1"
    (attempt_dir / "logs").mkdir(parents=True)
    (attempt_dir / "logs" / "pretrain_output.log").write_text(
        "".join(f"step {i}\n" for i in range(50))
    )

    _write_metrics(
        attempt_dir,
        "pretrain",
        RUN_ID,
        [{"loss": 3.0 - step * 0.1, "mfu": 0.01 * step} for step in range(20)],
    )

    settings = utrain.config.Settings(data_dir=data_dir)
    with utrain.db.with_db(settings) as session:
        session.execute(
            sqlalchemy.insert(utrain.db.runs).values(
                id=RUN_ID,
                name="tiny-shakespeare",
                image=IMAGE,
                image_id=IMAGE_ID,
                compute="cpu",
                status="done",
                config_hash=None,
                created_at=100.0,
            )
        )
        session.execute(
            sqlalchemy.insert(utrain.db.run_attempts).values(
                run_id=RUN_ID,
                attempt=1,
                from_phase=None,
                status="done",
                pid=None,
                started_at=100.0,
                ended_at=200.0,
            )
        )
        for order, phase in enumerate(["tokenizer", "pretrain"]):
            session.execute(
                sqlalchemy.insert(utrain.db.run_phases).values(
                    run_id=RUN_ID,
                    attempt=1,
                    phase=phase,
                    phase_order=order,
                    status="done",
                    started_at=100.0,
                    ended_at=200.0,
                )
            )


def _seed_draft(data_dir: pathlib.Path) -> None:
    """A run that has been created but not started, so its config is editable.

    Newer than the finished one, so `list_runs`'s created_at ordering puts it
    first and the app opens on it.
    """
    run_dir = data_dir / "runs" / DRAFT_ID
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(
        f"run_id: {DRAFT_ID}\ncompute: cpu\n"
        "globals:\n  model:\n    n_layer: 4\n    dtype: fp32\n"
        "phases:\n  pretrain:\n    lr: 0.001\n"
    )
    with utrain.db.with_db(utrain.config.Settings(data_dir=data_dir)) as session:
        session.execute(
            sqlalchemy.insert(utrain.db.runs).values(
                id=DRAFT_ID,
                name="draft",
                image=IMAGE,
                image_id=IMAGE_ID,
                compute="cpu",
                status="configuring",
                config_hash=None,
                created_at=300.0,
            )
        )


def _seed_running(data_dir: pathlib.Path) -> None:
    """A run whose orchestrator is alive, so `S` applies to it.

    No lock file is written and the attempt records this process's pid, which
    is the case `lock.is_held` falls back on: reconcile then leaves the attempt
    alone and the run stays `running` across the refreshes these tests do.
    Newest, so the app opens on it.
    """
    run_dir = data_dir / "runs" / LIVE_ID
    (run_dir / "attempt" / "1" / "logs").mkdir(parents=True)
    with utrain.db.with_db(utrain.config.Settings(data_dir=data_dir)) as session:
        session.execute(
            sqlalchemy.insert(utrain.db.runs).values(
                id=LIVE_ID,
                name="live",
                image=IMAGE,
                image_id=IMAGE_ID,
                compute="cpu",
                status="running",
                config_hash="deadbeef",
                created_at=400.0,
            )
        )
        session.execute(
            sqlalchemy.insert(utrain.db.run_attempts).values(
                run_id=LIVE_ID,
                attempt=1,
                from_phase=None,
                status="running",
                pid=os.getpid(),
                started_at=400.0,
                ended_at=None,
            )
        )
        for order, (phase, status) in enumerate([("tokenizer", "done"), ("pretrain", "running")]):
            session.execute(
                sqlalchemy.insert(utrain.db.run_phases).values(
                    run_id=LIVE_ID,
                    attempt=1,
                    phase=phase,
                    phase_order=order,
                    status=status,
                    started_at=400.0,
                    ended_at=None,
                )
            )


def _seed_restart_attempt(data_dir: pathlib.Path) -> None:
    """A second attempt for the seeded run, which has logged nothing yet.

    What `runs.restart_run` leaves behind: a new attempt directory, database
    rows naming it the latest, and no output in it at all.
    """
    (data_dir / "runs" / RUN_ID / "attempt" / "2" / "logs").mkdir(parents=True)
    with utrain.db.with_db(utrain.config.Settings(data_dir=data_dir)) as session:
        session.execute(
            sqlalchemy.insert(utrain.db.run_attempts).values(
                run_id=RUN_ID,
                attempt=2,
                from_phase=None,
                status="done",
                pid=None,
                started_at=300.0,
                ended_at=400.0,
            )
        )
        session.execute(
            sqlalchemy.insert(utrain.db.run_phases).values(
                run_id=RUN_ID,
                attempt=2,
                phase="pretrain",
                phase_order=1,
                status="done",
                started_at=300.0,
                ended_at=400.0,
            )
        )


RESTART_ID = "e" * 32


def _seed_restarted(data_dir: pathlib.Path) -> None:
    """A run restarted `--from-phase pretrain`, so tokenizer is inherited.

    Attempt 1 ran both phases and holds tokenizer's log and metrics; attempt 2
    ran pretrain alone, and its directory has nothing about tokenizer in it at
    all. Newest, so the app opens on it.
    """
    run_dir = data_dir / "runs" / RESTART_ID
    first = run_dir / "attempt" / "1"
    (first / "logs").mkdir(parents=True)
    (first / "logs" / "tokenizer_stdout.log").write_text("counting\nvocab built\n")
    _write_metrics(first, "tokenizer", RESTART_ID, [{"chars": 65.0}])

    (run_dir / "attempt" / "2" / "logs").mkdir(parents=True)

    with utrain.db.with_db(utrain.config.Settings(data_dir=data_dir)) as session:
        session.execute(
            sqlalchemy.insert(utrain.db.runs).values(
                id=RESTART_ID,
                name="restarted",
                image=IMAGE,
                image_id=IMAGE_ID,
                compute="cpu",
                status="done",
                config_hash=None,
                created_at=500.0,
            )
        )
        for attempt, from_phase in ((1, None), (2, "pretrain")):
            session.execute(
                sqlalchemy.insert(utrain.db.run_attempts).values(
                    run_id=RESTART_ID,
                    attempt=attempt,
                    from_phase=from_phase,
                    status="done",
                    pid=None,
                    started_at=500.0,
                    ended_at=600.0,
                )
            )
        for order, phase in enumerate(["tokenizer", "pretrain"]):
            session.execute(
                sqlalchemy.insert(utrain.db.run_phases).values(
                    run_id=RESTART_ID,
                    attempt=1,
                    phase=phase,
                    phase_order=order,
                    status="done",
                    started_at=500.0,
                    ended_at=600.0,
                )
            )
        session.execute(
            sqlalchemy.insert(utrain.db.run_phases).values(
                run_id=RESTART_ID,
                attempt=2,
                phase="pretrain",
                phase_order=1,
                status="done",
                started_at=700.0,
                ended_at=800.0,
            )
        )


NEW_ID = "d" * 32

# What `_RecordingData` offers the new-run dialog. Two images so that picking
# one is a choice, and one GPU so the compute list is not the cpu alone.
CHOICE_IMAGES = [IMAGE, "utrain-other"]


def _compute_info() -> utrain.compute.ComputeInfo:
    return utrain.compute.ComputeInfo(
        cpu=utrain.compute.CpuInfo(
            name="Fake CPU", cores=16, mem_total_gb=64.0, mem_available_gb=32.0
        ),
        gpus=[
            utrain.compute.GpuInfo(
                index=0,
                name="Fake GPU",
                power_draw=10.0,
                power_limit=350.0,
                util=0,
                mem_used_mb=0.0,
                mem_total_mb=24576.0,
            )
        ],
    )


class _RecordingData(utrain.tui.data.Data):
    """`Data` with the lifecycle calls recorded instead of made.

    `runs.start_run` runs `podman describe` and spawns an orchestrator, neither
    of which belongs in a test that is about which key does what -- and
    `new_run_choices` shells out to `podman images` and nvidia-smi. Subclassing
    the seam the TUI already goes through is the same trick as seeding the
    describe cache.
    """

    def __init__(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        super().__init__(*args, **kwargs)
        self.started: list[str] = []
        self.stopped: list[str] = []
        self.restarted: list[tuple[str, str | None]] = []
        self.deleted: list[str] = []
        self.created: list[tuple[str, str, str]] = []
        # The run of every `snapshot` call, in order.
        self.fetched: list[str | None] = []
        # Emptied by the test that wants the dialog to find nothing to offer,
        # and appended to by `add_image` -- an image added is an image the
        # new-run dialog can then pick, which is true of the real one too.
        self.images = list(CHOICE_IMAGES)
        self.added: list[str] = []
        # Set by the test that wants to see a failure marshalled back.
        self.fail = ""

    def snapshot(
        self,
        run_id: str | None,
        phase: str | None,
        attempt: int | None = None,
        tail: utrain.metrics.Tail | None = None,
        log_lines: int = utrain.tui.data._LOG_LINES,
    ) -> utrain.tui.data.Snapshot:
        """The real fetch, with the run it was for written down.

        What the settling is about is how many of these a cursor move costs, so
        counting them is the only way to assert on it.
        """
        self.fetched.append(run_id)
        return super().snapshot(run_id, phase, attempt, tail, log_lines)

    def start_run(self, run_id: str) -> None:
        if self.fail:
            raise utrain.exceptions.UI(self.fail)
        self.started.append(run_id)

    def stop_run(self, run_id: str) -> None:
        self.stopped.append(run_id)

    def restart_run(self, run_id: str, from_phase: str | None) -> None:
        if self.fail:
            raise utrain.exceptions.UI(self.fail)
        self.restarted.append((run_id, from_phase))

    def delete_run(self, run_id: str) -> None:
        """Recorded, and the row removed for real.

        Unlike start and stop, what the screen does next depends on the run
        being gone: the rows below it shift up under a stationary cursor, and
        the selection has to follow.
        """
        if self.fail:
            raise utrain.exceptions.UI(self.fail)
        self.deleted.append(run_id)
        with utrain.db.with_db(self.settings) as session:
            for table in (utrain.db.run_phases, utrain.db.run_attempts):
                session.execute(sqlalchemy.delete(table).where(table.c.run_id == run_id))
            session.execute(sqlalchemy.delete(utrain.db.runs).where(utrain.db.runs.c.id == run_id))

    def new_run_choices(self) -> utrain.tui.data.NewRunChoices:
        return utrain.tui.data.NewRunChoices(images=self.images, compute=_compute_info())

    def list_images(self) -> list[utrain.images.ImageInfo]:
        """The real one shells out to `podman images` and `podman image inspect`."""
        return [
            utrain.images.ImageInfo(name=name, size_str="10M", run_count=0) for name in self.images
        ]

    def add_image(self, url: str) -> str:
        """Recorded, and added to the list for real.

        Not just recorded, because what the screen does next is refresh and
        show the new row; an add that recorded and did nothing else could not
        be told from one that got lost.
        """
        if self.fail:
            raise utrain.exceptions.UI(self.fail)
        self.added.append(url)
        name = utrain.container.podman.preset_key(url.split("://")[-1].split("/")[-1].split(":")[0])
        self.images.append(name)
        return name

    def create_run(self, name: str, image: str, compute_spec: str) -> str:
        """Recorded, and inserted for real.

        The row has to land in the database because what the screen does next is
        wait for the fetch that lists it and put the cursor there; a create that
        recorded and did nothing else could not be told from one that got lost.
        """
        if self.fail:
            raise utrain.exceptions.UI(self.fail)
        self.created.append((name, image, compute_spec))
        run_dir = self.settings.runs_dir / NEW_ID
        run_dir.mkdir(parents=True)
        (run_dir / "config.yaml").write_text(
            f"run_id: {NEW_ID}\ncompute: {compute_spec}\n"
            "globals:\n  model:\n    n_layer: 4\n    dtype: fp32\n"
            "phases:\n  pretrain:\n    lr: 0.001\n"
        )
        with utrain.db.with_db(self.settings) as session:
            session.execute(
                sqlalchemy.insert(utrain.db.runs).values(
                    id=NEW_ID,
                    name=name,
                    image=IMAGE,
                    image_id=IMAGE_ID,
                    compute=compute_spec,
                    status="configuring",
                    config_hash=None,
                    created_at=500.0,
                )
            )
        return NEW_ID


def _recording(app: utrain.tui.app.UtrainApp) -> _RecordingData:
    source = app.data
    assert isinstance(source, _RecordingData)
    return source


def _described_with_plots() -> utrain.container.schema.DescribeOutput:
    """The same image, but `pretrain` names the plot it wants opened on.

    `tokenizer` names none, so one image covers both paths -- and `_describe`
    stays plotless, which is what keeps every dashboard test above about the
    dashboard.
    """
    described = _describe()
    for info in described.phases:
        if info.name == "pretrain":
            info.plots = [utrain.container.schema.PlotSpec(x="step", y="loss")]
    return described


def _app(
    data_dir: pathlib.Path,
    described: utrain.container.schema.DescribeOutput | None = None,
    charset: str = "block",
) -> utrain.tui.app.UtrainApp:
    """The app under test.

    `charset` is pinned rather than left on `auto`, whose answer depends on the
    `TERM` the suite happens to run under. What `auto` decides is
    `render.default_charset`'s business and is tested there; these tests are
    about what the screen does with the answer.
    """
    source = _RecordingData(
        settings=utrain.config.Settings(data_dir=data_dir, tui_charset=charset),
        describe_cache={IMAGE_ID: described if described is not None else _describe()},
    )
    return utrain.tui.app.UtrainApp(source)


@pytest.fixture()
def app(tmp_path: pathlib.Path) -> utrain.tui.app.UtrainApp:
    _seed(tmp_path)
    return _app(tmp_path)


@pytest.fixture()
def plotted_app(tmp_path: pathlib.Path) -> utrain.tui.app.UtrainApp:
    """The same run, against an image whose `pretrain` names its own plot."""
    _seed(tmp_path)
    return _app(tmp_path, _described_with_plots())


@pytest.fixture()
def draft_app(tmp_path: pathlib.Path) -> utrain.tui.app.UtrainApp:
    _seed(tmp_path)
    _seed_draft(tmp_path)
    return _app(tmp_path)


@pytest.fixture()
def restarted_app(tmp_path: pathlib.Path) -> utrain.tui.app.UtrainApp:
    """The restarted run first, the finished one under it."""
    _seed(tmp_path)
    _seed_restarted(tmp_path)
    return _app(tmp_path)


@pytest.fixture()
def live_app(tmp_path: pathlib.Path) -> utrain.tui.app.UtrainApp:
    """A running run first, the configuring one under it, the finished one last."""
    _seed(tmp_path)
    _seed_draft(tmp_path)
    _seed_running(tmp_path)
    return _app(tmp_path)


def _footer(app: utrain.tui.app.UtrainApp) -> dict[str, bool]:
    """The keys the footer is actually showing, mapped to whether they are live.

    Read off the widget rather than off `Screen.active_bindings`, because those
    two are exactly what came apart: `check_action` is re-read whenever anyone
    asks, so `active_bindings` is always right, while the footer is composed
    once and recomposed only when the screen says its bindings changed. Only
    this one sees a stale footer.

    `FooterKey` lives in a private module, so it is identified by what it
    carries -- a `key`, and the `-disabled` class Textual greys it with.
    """
    footer = app.screen.query(textual.widgets.Footer).first()
    return {
        key: not widget.has_class("-disabled")
        for widget in footer.query("*")
        if isinstance(key := getattr(widget, "key", None), str)
    }


def _main(app: utrain.tui.app.UtrainApp) -> utrain.tui.screens.MainScreen:
    screen = app.screen
    assert isinstance(screen, utrain.tui.screens.MainScreen)
    return screen


async def _settle(app: utrain.tui.app.UtrainApp, pilot: typing.Any) -> None:
    """Let the thread workers finish and their results reach the screen.

    The refresh timer keeps firing while this waits, and the fetch workers are
    exclusive, so a tick can cancel the one being waited on. That is a normal
    race rather than a failure -- wait again, and the replacement is the one
    that lands.
    """
    for _ in range(5):
        try:
            await app.workers.wait_for_complete()
            break
        except textual.worker.WorkerCancelled:
            continue
    for _ in range(4):
        await pilot.pause()


async def _pause_until(
    pilot: typing.Any, predicate: collections.abc.Callable[[], bool], what: str
) -> None:
    """Pump the message loop until `predicate` holds.

    One `pilot.pause()` is a single cycle of it, which is all an idle machine
    needs. Scrolling to a highlight and rebuilding an option list are both
    scheduled after a refresh, so on a loaded one an assertion placed after a
    single pause can run while the widget is still catching up.
    """
    deadline = time.monotonic() + 5.0
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        await pilot.pause()


async def _select_pretrain(app: utrain.tui.app.UtrainApp, pilot: typing.Any) -> None:
    """Drill into the run, whose current phase is the one that logged metrics.

    Both of the seeded phases are done, so the run's current phase is the last
    of them -- which is why this no longer has to move the cursor.
    """
    await _settle(app, pilot)
    await pilot.press("enter")
    await _settle(app, pilot)
    # The metrics arrive on the tick after the selection, since the fetch that
    # was in flight was for the previously selected phase.
    await _settle(app, pilot)


# -- selection ------------------------------------------------------------


async def test_the_app_opens_on_the_first_run_and_where_it_has_got_to(
    app: utrain.tui.app.UtrainApp,
) -> None:
    """The current phase, not the first: the runs list is a live monitor, so
    standing on a run shows what it is doing rather than what it did first."""
    async with app.run_test(size=SIZE) as pilot:
        await _settle(app, pilot)
        screen = _main(app)

        assert [r.name for r in screen.runs] == ["tiny-shakespeare"]
        assert screen.selected_run == RUN_ID
        assert screen.level == "runs"
        assert [p.phase for p in screen.phases] == ["tokenizer", "pretrain"]
        assert screen.selected_phase == "pretrain"


async def test_moving_the_phase_cursor_selects_that_phase(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)

        # The full run id, not the short form the table displays.
        assert _main(app).address() == f"{RUN_ID}/pretrain"

        await pilot.press("up")
        await _settle(app, pilot)

        assert _main(app).address() == f"{RUN_ID}/tokenizer"


async def test_a_highlight_for_a_row_the_cursor_has_left_is_not_a_selection(
    app: utrain.tui.app.UtrainApp,
) -> None:
    """A queued highlight for another row is stale, not a second selection.

    Filling the phases table for the first time highlights row 0 as the cursor
    becomes valid, and the same pass moves the cursor to the phase the run has
    got to. Both are messages; a snapshot lands from a thread rather than
    through the queue, so it can arrive between them -- and if the first were
    read as a selection, it would put the cursor back on row 0 and leave the
    screen on a phase the viewer never picked.
    """
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        screen = _main(app)
        table = screen.table("#phases")
        assert table.cursor_row == 1

        table.post_message(
            textual.widgets.DataTable.RowHighlighted(table, 0, table.ordered_rows[0].key)
        )
        await _settle(app, pilot)

        assert screen.selected_phase == "pretrain"
        assert table.cursor_row == 1


async def test_enter_drills_into_the_run_and_escape_comes_back(
    app: utrain.tui.app.UtrainApp,
) -> None:
    """One list, two levels: `enter` is down and `escape` is up, and the list
    says which of the two it is showing."""
    async with app.run_test(size=SIZE) as pilot:
        await _settle(app, pilot)
        screen = _main(app)
        assert screen.level == "runs"
        assert screen.query_one("#runs").display
        assert not screen.query_one("#phases").display

        await pilot.press("enter")
        await _settle(app, pilot)
        assert screen.level == "phases"
        assert screen.query_one("#phases").display
        assert not screen.query_one("#runs").display
        assert "tiny-shakespeare" in str(screen.query_one("#phases").border_title)

        await pilot.press("escape")
        await _settle(app, pilot)
        assert screen.level == "runs"
        assert app.focused is not None and app.focused.id == "runs"


async def test_coming_back_to_a_run_lands_on_the_phase_it_was_left_at(
    app: utrain.tui.app.UtrainApp,
) -> None:
    """Otherwise the drill is lossy, and stepping up to compare two runs costs
    the place in each of them."""
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        screen = _main(app)
        await pilot.press("up")
        await _settle(app, pilot)
        assert screen.selected_phase == "tokenizer"

        await pilot.press("escape")
        await _settle(app, pilot)
        await pilot.press("enter")
        await _settle(app, pilot)

        assert screen.selected_phase == "tokenizer"


async def test_i_and_c_open_panels_over_the_run(app: utrain.tui.app.UtrainApp) -> None:
    """The images and the compute are panels over whatever the viewer was
    looking at, not places of their own: `i` or `c` pushes one, `escape`
    closes it, and the screen is exactly what it was before it went up."""
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        await pilot.press("3")
        await _settle(app, pilot)

        await pilot.press("i")
        await _settle(app, pilot)
        assert isinstance(app.screen, utrain.tui.screens.ImagesScreen)

        await pilot.press("escape")
        await _settle(app, pilot)
        assert isinstance(app.screen, utrain.tui.screens.MainScreen)
        assert _main(app).address() == f"{RUN_ID}/pretrain"
        assert _main(app).tab == "plots"

        await pilot.press("c")
        await _settle(app, pilot)
        assert isinstance(app.screen, utrain.tui.screens.ComputeScreen)


async def test_a_panel_leaves_the_screen_beneath_where_it_was(
    app: utrain.tui.app.UtrainApp,
) -> None:
    """The screen under a modal is suspended, not thrown away: going to look
    at an image must not throw away the phase the viewer had open."""
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        await pilot.press("3")
        await _settle(app, pilot)
        before = _main(app)
        assert before.address() == f"{RUN_ID}/pretrain"

        await pilot.press("i")
        await _settle(app, pilot)
        await pilot.press("escape")
        await _settle(app, pilot)

        assert _main(app) is before
        assert before.address() == f"{RUN_ID}/pretrain"
        assert before.tab == "plots"


async def test_a_panel_dims_the_screen_beneath_it(app: utrain.tui.app.UtrainApp) -> None:
    """A modal is only a modal if what is under it shows through: the panel's
    background is translucent, and the run being watched is still composed
    behind it, dimmed.

    `_Popover` reaches `ModalScreen` through the MRO but not through the walk
    Textual uses to gather DEFAULT_CSS, which follows the first DOMNode base
    and lands on Screen's opaque background instead -- so the translucent
    colour is restated in the app's CSS, and this is the test that notices if
    it is dropped again.
    """
    async with app.run_test(size=SIZE) as pilot:
        await _settle(app, pilot)
        await pilot.press("i")
        await _settle(app, pilot)

        assert app.screen.styles.background.a < 1
        # And what that alpha buys: the runs browser, drawn behind the panel.
        assert "tiny-shakespeare" in app.export_screenshot()


async def test_a_panel_keeps_the_keys_of_the_screen_beneath_from_firing(
    app: utrain.tui.app.UtrainApp,
) -> None:
    """A modal intercepts the keys: `2` aimed at the content tabs stays inside
    the panel, and the main screen does not move under it."""
    async with app.run_test(size=SIZE) as pilot:
        await _settle(app, pilot)
        main = _main(app)

        await pilot.press("i")
        await _settle(app, pilot)
        await pilot.press("2")
        await _settle(app, pilot)

        assert isinstance(app.screen, utrain.tui.screens.ImagesScreen)
        assert main.tab == "config"


async def test_q_does_not_quit_through_a_panel(app: utrain.tui.app.UtrainApp) -> None:
    """`q` is off while a modal is up, panels included: they are left with
    escape, the way every dialog is."""
    async with app.run_test(size=SIZE) as pilot:
        await _settle(app, pilot)
        await pilot.press("i")
        await _settle(app, pilot)
        await pilot.press("q")
        await _settle(app, pilot)

        assert app.is_running
        assert isinstance(app.screen, utrain.tui.screens.ImagesScreen)


# -- the metrics pane -----------------------------------------------------


async def test_the_metrics_pane_lists_what_the_phase_logged(
    app: utrain.tui.app.UtrainApp,
) -> None:
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        metrics = app.screen.query_one("#metrics", utrain.tui.widgets.MetricList)

        assert sorted(metrics.columns) == ["loss", "mfu"]
        # Everything is plotted until the viewer says otherwise.
        assert sorted(metrics.selected) == ["loss", "mfu"]


async def test_the_metrics_pane_shows_each_metric_s_last_value(
    app: utrain.tui.app.UtrainApp,
) -> None:
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        lines = [
            option.prompt for option in app.screen.query_one(utrain.tui.widgets.MetricList).options
        ]

        # 3.0 - 19 * 0.1, to four significant figures.
        assert any("loss" in str(line) and "1.1" in str(line) for line in lines)


def _marked_rows(metrics: utrain.tui.widgets.MetricList) -> list[int]:
    """The rows drawn as part of the range, read back off the prompts."""
    marked: list[int] = []
    for index in range(metrics.option_count):
        prompt = metrics.get_option_at_index(index).prompt
        spans = getattr(prompt, "spans", [])
        if any("reverse" in str(span.style) for span in spans):
            marked.append(index)
    return marked


async def test_shift_arrows_extend_the_range_space_acts_on(
    app: utrain.tui.app.UtrainApp,
) -> None:
    """Both metrics start checked, so one `space` over both unchecks both."""
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        metrics = app.screen.query_one("#metrics", utrain.tui.widgets.MetricList)
        await pilot.press("m")
        await _settle(app, pilot)
        assert sorted(metrics.selected) == ["loss", "mfu"]

        await pilot.press("shift+down")
        await _settle(app, pilot)
        assert metrics.range_indices == [0, 1]

        await pilot.press("space")
        await _settle(app, pilot)

        assert metrics.selected == []
        assert not [name for name, p in _main(app).plots.items() if p.display]


async def test_space_checks_a_range_that_is_not_all_checked(
    app: utrain.tui.app.UtrainApp,
) -> None:
    """A mixed range goes one way, rather than each row flipping on its own."""
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        metrics = app.screen.query_one("#metrics", utrain.tui.widgets.MetricList)
        await pilot.press("m")
        await pilot.press("space")
        await _settle(app, pilot)
        assert len(metrics.selected) == 1

        await pilot.press("shift+down")
        await pilot.press("space")
        await _settle(app, pilot)

        assert sorted(metrics.selected) == ["loss", "mfu"]


async def test_a_plain_arrow_rubs_out_the_range_it_drops(
    app: utrain.tui.app.UtrainApp,
) -> None:
    """The marks are drawn into the prompts, so dropping the range has to redraw.

    Asserted on the prompts rather than on `range_indices`: what went wrong
    before was that the range was gone and still on screen.
    """
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        metrics = app.screen.query_one("#metrics", utrain.tui.widgets.MetricList)
        await pilot.press("m")
        await pilot.press("shift+down")
        await _settle(app, pilot)
        # The cursor's own row wears the cursor bar instead, so the range shows
        # as the one row it has grown past.
        assert _marked_rows(metrics) == [0]

        await pilot.press("up")
        await _settle(app, pilot)

        assert metrics.range_indices == [0]
        assert _marked_rows(metrics) == []


async def test_a_plain_arrow_drops_the_range(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        metrics = app.screen.query_one("#metrics", utrain.tui.widgets.MetricList)
        await pilot.press("m")
        await pilot.press("shift+down")
        await _settle(app, pilot)
        assert metrics.range_indices == [0, 1]

        await pilot.press("up")
        await _settle(app, pilot)
        assert metrics.range_indices == [0]

        await pilot.press("space")
        await _settle(app, pilot)

        # Only the row the cursor is on, so the other metric is still drawn.
        assert len(metrics.selected) == 1


async def test_extending_stops_at_the_end_of_the_list(
    app: utrain.tui.app.UtrainApp,
) -> None:
    """Rather than wrapping the way the plain arrows do."""
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        metrics = app.screen.query_one("#metrics", utrain.tui.widgets.MetricList)
        await pilot.press("m")
        for _ in range(4):
            await pilot.press("shift+down")
        await _settle(app, pilot)

        assert metrics.highlighted == 1
        assert metrics.range_indices == [0, 1]


async def test_a_refresh_leaves_the_metrics_cursor_and_scroll_alone(
    app: utrain.tui.app.UtrainApp,
) -> None:
    """A live phase relogs every value each tick; the viewer's place must hold.

    Driven through the widget rather than through a fetch, because the pane has
    to be longer than the picker for a scroll offset to exist at all, and the
    seeded run logs two metrics.
    """

    def _rows(offset: float) -> list[utrain.tui.render.MetricRow]:
        return [
            utrain.tui.render.MetricRow(mark="*", name=f"m{i:02d}", value=f"{i + offset:g}")
            for i in range(80)
        ]

    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        await pilot.press("m")
        await _settle(app, pilot)
        # The tick would otherwise repaint the pane with the run's own two
        # metrics halfway through.
        timer = _main(app).timer
        assert timer is not None
        timer.stop()
        metrics = app.screen.query_one("#metrics", utrain.tui.widgets.MetricList)
        metrics.show(_rows(0.0))
        await _pause_until(pilot, lambda: metrics.option_count == 80, "the rows to be listed")
        metrics.highlighted = 70
        await _pause_until(pilot, lambda: metrics.scroll_y > 0, "the cursor to scroll into view")
        # Scrolled away from the cursor, which is what reading further up the
        # list looks like and what a rebuild cannot put back: restoring the
        # highlight scrolls it into view, and that is not where this is.
        metrics.scroll_to(y=0, animate=False, immediate=True)
        await _pause_until(pilot, lambda: metrics.scroll_y == 0, "the pane to scroll back up")

        metrics.show(_rows(0.5))
        # Waiting on the new values rather than on a fixed number of cycles: the
        # point of the test is where the cursor and the offset are once the
        # rebuild has landed, so the rebuild has to be observably done first.
        await _pause_until(
            pilot,
            lambda: "70.5" in str(metrics.get_option_at_index(70).prompt),
            "the relogged values to land",
        )

        assert metrics.highlighted == 70
        assert metrics.scroll_y == 0


async def test_the_dashboard_reads_the_metric_points(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        points = _main(app).points[f"{RUN_ID}/pretrain"]

        assert len(points["loss"]) == 20
        assert points["loss"][0].value == pytest.approx(3.0)


async def test_a_restart_starts_the_plots_over(
    app: utrain.tui.app.UtrainApp, tmp_path: pathlib.Path
) -> None:
    """A restart's attempt is a new run of the phase, not a continuation.

    The series the dashboard accumulates are keyed by `RUN/PHASE`, which keeps
    meaning the latest attempt across a restart, and the Tail it holds was
    opened on the previous attempt's file -- which nothing appends to any more,
    so the curves would freeze there forever. The restart has to be seen: the
    plots go empty until the new attempt logs, and what it logs is all there
    is.
    """
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        screen = _main(app)
        addr = f"{RUN_ID}/pretrain"
        assert len(screen.points[addr]["loss"]) == 20

        _seed_restart_attempt(tmp_path)
        screen.refresh_data()
        await _settle(app, pilot)

        assert screen.points.get(addr) == {}
        metrics = app.screen.query_one("#metrics", utrain.tui.widgets.MetricList)
        assert metrics.columns == []

        _write_metrics(
            tmp_path / "runs" / RUN_ID / "attempt" / "2",
            "pretrain",
            RUN_ID,
            [{"loss": 1.5, "mfu": 0.5}],
        )
        screen.refresh_data()
        await _settle(app, pilot)
        await _settle(app, pilot)

        assert [p.value for p in screen.points[addr]["loss"]] == [1.5]
        plot = screen.plots[("loss", "step")].plot
        assert plot is not None
        assert plot.ys == [1.5]


async def test_every_selected_metric_gets_its_own_plot(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        screen = _main(app)

        assert sorted(screen.plots) == [("loss", "step"), ("mfu", "step")]
        assert all(p.display for p in screen.plots.values())
        loss = screen.plots[("loss", "step")].plot
        assert loss is not None and loss.title == "loss"


async def test_toggling_a_metric_hides_its_plot(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        screen = _main(app)
        await pilot.press("m")
        await pilot.press("space")
        await _settle(app, pilot)

        metrics = screen.query_one("#metrics", utrain.tui.widgets.MetricList)
        assert metrics.selected == ["mfu"]
        assert {key for key, p in screen.plots.items() if p.display} == {("mfu", "step")}


async def test_toggling_a_metric_back_shows_it_again(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        await pilot.press("m")
        await pilot.press("space")
        await _settle(app, pilot)
        await pilot.press("space")
        await _settle(app, pilot)

        metrics = app.screen.query_one("#metrics", utrain.tui.widgets.MetricList)
        assert sorted(metrics.selected) == ["loss", "mfu"]


async def test_y_solos_one_metric(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        screen = _main(app)
        await pilot.press("m")
        await pilot.press("y")
        await _settle(app, pilot)

        assert screen.solo == "loss"
        assert {key for key, p in screen.plots.items() if p.display} == {("loss", "step")}


async def test_y_again_stops_soloing(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        screen = _main(app)
        await pilot.press("m")
        await pilot.press("y")
        await _settle(app, pilot)
        await pilot.press("y")
        await _settle(app, pilot)

        assert screen.solo is None
        assert {key for key, p in screen.plots.items() if p.display} == {
            ("loss", "step"),
            ("mfu", "step"),
        }


async def test_x_cycles_the_x_axis(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        screen = _main(app)
        assert screen.x_axis == utrain.tui.render.X_STEP

        await pilot.press("m")
        await pilot.press("x")
        await _settle(app, pilot)
        assert screen.x_axis == utrain.tui.render.X_ELAPSED

        # ...and on into the metrics, so one can be plotted against another.
        await pilot.press("x")
        await _settle(app, pilot)
        assert screen.x_axis in ("loss", "mfu")


async def test_the_x_axis_reaches_the_plots(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        await pilot.press("m")
        await pilot.press("x")
        await _settle(app, pilot)

        plot = _main(app).plots[("loss", utrain.tui.render.X_ELAPSED)].plot
        assert plot is not None
        assert plot.x_label == "elapsed (s)"


async def test_l_toggles_log_scale_on_every_plot(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        await pilot.press("3")
        await pilot.press("l")
        await _settle(app, pilot)

        assert all(p.log_y for p in _main(app).plots.values())


async def test_coming_back_to_a_phase_still_shows_its_metrics(
    app: utrain.tui.app.UtrainApp,
) -> None:
    """The Tail hands each column name over once, and it has already done so.

    So leaving a phase and returning must re-seed the pane from what was read
    the first time, not from an exhausted reader.
    """
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        await pilot.press("up")
        await _settle(app, pilot)
        await pilot.press("down")
        await _settle(app, pilot)

        metrics = app.screen.query_one("#metrics", utrain.tui.widgets.MetricList)
        assert sorted(metrics.columns) == ["loss", "mfu"]
        assert sorted(_main(app).plots) == [("loss", "step"), ("mfu", "step")]


# -- plots the container asked for ----------------------------------------


async def test_a_phase_that_names_a_plot_gets_that_one_only(
    plotted_app: utrain.tui.app.UtrainApp,
) -> None:
    app = plotted_app
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        screen = _main(app)

        assert screen.pinned == [("loss", "step")]
        assert {key for key, p in screen.plots.items() if p.display} == {("loss", "step")}


async def test_a_phase_that_names_no_plot_still_gets_the_dashboard(
    plotted_app: utrain.tui.app.UtrainApp,
) -> None:
    """`tokenizer` names none."""
    app = plotted_app
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        await pilot.press("up")
        await _settle(app, pilot)

        assert _main(app).selected_phase == "tokenizer"
        assert _main(app).pinned == []


async def test_only_the_named_metrics_start_checked(
    plotted_app: utrain.tui.app.UtrainApp,
) -> None:
    app = plotted_app
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)

        metrics = app.screen.query_one("#metrics", utrain.tui.widgets.MetricList)
        assert sorted(metrics.columns) == ["loss", "mfu"]
        assert metrics.selected == ["loss"]


async def test_checking_a_metric_takes_the_dashboard_back(
    plotted_app: utrain.tui.app.UtrainApp,
) -> None:
    app = plotted_app
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        screen = _main(app)
        await pilot.press("m")
        await pilot.press("down")
        await pilot.press("space")
        await _settle(app, pilot)

        assert screen.pinned == []
        assert {key for key, p in screen.plots.items() if p.display} == {
            ("loss", "step"),
            ("mfu", "step"),
        }


async def test_cycling_the_x_axis_takes_the_dashboard_back(
    plotted_app: utrain.tui.app.UtrainApp,
) -> None:
    """A named plot carries its own x, so one screen-wide x cannot show it."""
    app = plotted_app
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        screen = _main(app)
        await pilot.press("m")
        await pilot.press("x")
        await _settle(app, pilot)

        assert screen.pinned == []
        assert {key for key, p in screen.plots.items() if p.display} == {
            ("loss", utrain.tui.render.X_ELAPSED)
        }


async def test_soloing_and_back_returns_to_the_named_plot(
    plotted_app: utrain.tui.app.UtrainApp,
) -> None:
    """`y` says "this curve for a moment", not "give me the dashboard"."""
    app = plotted_app
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        screen = _main(app)
        await pilot.press("m")
        await pilot.press("down")
        await pilot.press("y")
        await _settle(app, pilot)
        assert {key for key, p in screen.plots.items() if p.display} == {("mfu", "step")}

        await pilot.press("y")
        await _settle(app, pilot)
        assert screen.pinned == [("loss", "step")]
        assert {key for key, p in screen.plots.items() if p.display} == {("loss", "step")}


async def test_leaving_the_phase_forgets_that_the_dashboard_was_asked_for(
    plotted_app: utrain.tui.app.UtrainApp,
) -> None:
    """Which plots a phase names is the phase's answer, not the run's."""
    app = plotted_app
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        screen = _main(app)
        await pilot.press("m")
        await pilot.press("down")
        await pilot.press("space")
        await _settle(app, pilot)
        assert screen.pinned == []

        # Off the phase and back onto it, which is what forgets the override.
        await pilot.press("1")
        await pilot.press("up")
        await _settle(app, pilot)
        await pilot.press("down")
        await _settle(app, pilot)
        await _settle(app, pilot)

        assert screen.pinned == [("loss", "step")]
        metrics = app.screen.query_one("#metrics", utrain.tui.widgets.MetricList)
        assert metrics.selected == ["loss"]


async def test_the_log_pane_shows_the_tail(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        log = app.screen.query_one("#log", utrain.tui.widgets.LogTail)
        assert log.shown[-1] == "step 49"


async def _settled_log_pane(
    app: utrain.tui.app.UtrainApp, pilot: typing.Any
) -> utrain.tui.widgets.LogTail:
    """The seeded run's log pane, with the refresh timer stopped.

    The tests below drive the widget directly, and a tick repainting the pane
    halfway through would race them.
    """
    await _select_pretrain(app, pilot)
    timer = _main(app).timer
    assert timer is not None
    timer.stop()
    return app.screen.query_one("#log", utrain.tui.widgets.LogTail)


async def test_a_growing_log_is_appended_not_rewritten(
    app: utrain.tui.app.UtrainApp,
) -> None:
    """The pane writes only the lines that grew onto the tail it has shown.

    Clearing and rewriting on each refresh used to throw the pane's scrollback
    away with every tick, which left the viewer unable to read anything past
    the fetched tail -- the log showed only its end.
    """
    async with app.run_test(size=SIZE) as pilot:
        log = await _settled_log_pane(app, pilot)
        assert log.shown == [f"step {i}" for i in range(50)]

        log.show([f"step {i}" for i in range(60)])
        await _pause_until(pilot, lambda: len(log.lines) == 60, "the new lines to land")

        assert log.shown == [f"step {i}" for i in range(60)]
        # Sixty lines in the pane: the fifty it had, plus the ten that grew
        # onto them, none of the old ones cleared away.
        assert len(log.lines) == 60


async def test_an_unchanged_log_writes_nothing(app: utrain.tui.app.UtrainApp) -> None:
    """A re-read tail identical to what the pane has writes no lines.

    The log is re-read every refresh; rewriting it would scroll the pane out
    from under the viewer once a second.
    """
    async with app.run_test(size=SIZE) as pilot:
        log = await _settled_log_pane(app, pilot)
        assert len(log.lines) == 50

        log.show(log.shown)
        await pilot.pause()

        assert log.shown == [f"step {i}" for i in range(50)]
        assert len(log.lines) == 50


async def test_a_log_the_pane_has_not_seen_starts_over(
    app: utrain.tui.app.UtrainApp,
) -> None:
    """A tail the pane's content does not extend cannot be appended to.

    A different phase's output, or a new attempt's, arrives as the tail of a
    file the pane has never seen; writing it onto the old lines would
    interleave two logs.
    """
    async with app.run_test(size=SIZE) as pilot:
        log = await _settled_log_pane(app, pilot)
        assert len(log.lines) == 50

        log.show(["vocab built"])
        await _pause_until(pilot, lambda: len(log.lines) == 1, "the new log to replace the old")

        assert log.shown == ["vocab built"]
        assert len(log.lines) == 1


async def test_log_text_can_be_selected_and_copied(
    app: utrain.tui.app.UtrainApp,
) -> None:
    """Dragging across the log selects the text under it, and ctrl+c copies it.

    The pane is a `Log` rather than a `RichLog` for exactly this: a RichLog
    accepts the drag but can extract no text from its rendered strips, so the
    selection it made copied nothing.
    """
    async with app.run_test(size=SIZE) as pilot:
        log = await _settled_log_pane(app, pilot)
        # The pane is only on screen while its tab is; the widget-level tests
        # above drive it hidden, but a drag needs it displayed.
        await pilot.press("4")
        await pilot.pause()
        log.scroll_to(y=0, animate=False, immediate=True)
        await _pause_until(pilot, lambda: log.scroll_y == 0, "the pane to scroll to the top")

        # Drag across the first line's text. The border is the outer cell, so
        # the first content cell sits at offset (1, 1) and the offsets below
        # cover the whole of `step 0`.
        await pilot.mouse_down(log, offset=(1, 1))
        await pilot.hover(log, offset=(1 + len("step 0"), 1))
        await pilot.mouse_up(log, offset=(1 + len("step 0"), 1))
        await pilot.pause()

        assert app.screen.get_selected_text() == "step 0"

        await pilot.press("ctrl+c")
        await pilot.pause()
        assert app._clipboard == "step 0"


async def test_growth_leaves_a_viewer_who_scrolled_up_where_they_are(
    app: utrain.tui.app.UtrainApp,
) -> None:
    """Reading further back is not undone by the phase writing more output.

    A viewer scrolled up holds their place while the pane grows under them;
    one sitting at the bottom keeps following.
    """
    async with app.run_test(size=SIZE) as pilot:
        log = await _settled_log_pane(app, pilot)
        log.show([f"step {i}" for i in range(200)])
        await _pause_until(pilot, lambda: len(log.lines) == 200, "the tail to be written")
        log.scroll_to(y=0, animate=False, immediate=True)
        await _pause_until(pilot, lambda: log.scroll_y == 0, "the pane to scroll to the top")

        log.show([f"step {i}" for i in range(210)])
        await _pause_until(pilot, lambda: len(log.lines) == 210, "the new lines to land")

        assert log.scroll_y == 0

        log.scroll_end(animate=False)
        await _pause_until(
            pilot, lambda: log.is_vertical_scroll_end, "the pane to reach the bottom"
        )
        log.show([f"step {i}" for i in range(220)])
        await _pause_until(
            pilot,
            lambda: log.is_vertical_scroll_end and log.scroll_y > 0,
            "the pane to follow the growth",
        )

        assert log.is_vertical_scroll_end


async def test_a_growing_log_file_extends_the_pane(
    app: utrain.tui.app.UtrainApp, tmp_path: pathlib.Path
) -> None:
    """A snapshot of a phase that wrote more output grows what the pane shows."""
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        screen = _main(app)
        log = screen.query_one("#log", utrain.tui.widgets.LogTail)
        assert log.shown[-1] == "step 49"

        log_file = tmp_path / "runs" / RUN_ID / "attempt" / "1" / "logs" / "pretrain_output.log"
        with log_file.open("a") as f:
            f.write("".join(f"step {i}\n" for i in range(50, 60)))
        screen.apply(app.data.snapshot(RUN_ID, "pretrain", None))
        await pilot.pause()

        assert log.shown == [f"step {i}" for i in range(60)]


async def test_an_inherited_phase_shows_the_attempt_that_ran_it(
    restarted_app: utrain.tui.app.UtrainApp,
) -> None:
    """A `--from-phase` restart leaves the phases before it in the old attempt.

    They are listed with the status and the timings they finished with there,
    and the log and plot panes read the same attempt: the current one has no
    directory for a phase it skipped.
    """
    async with restarted_app.run_test(size=SIZE) as pilot:
        await _settle(restarted_app, pilot)
        screen = _main(restarted_app)
        assert screen.selected_run == RESTART_ID
        await pilot.press("enter")
        await pilot.press("up")
        await _settle(restarted_app, pilot)
        assert screen.selected_phase == "tokenizer"
        # The list itself: inherited from attempt 1, and addressed under it.
        entry = screen.selected_entry()
        assert entry is not None and entry.inherited_from == 1
        await _settle(restarted_app, pilot)

        assert screen.address() == f"{RESTART_ID}/1/tokenizer"
        log = restarted_app.screen.query_one("#log", utrain.tui.widgets.LogTail)
        assert log.shown[-1] == "vocab built"
        metrics = restarted_app.screen.query_one("#metrics", utrain.tui.widgets.MetricList)
        assert metrics.columns == ["chars"]


async def test_a_phase_that_logged_nothing_is_empty_not_broken(
    app: utrain.tui.app.UtrainApp,
) -> None:
    """tokenizer has no rtsdb and no log file here."""
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        await pilot.press("up")
        await _settle(app, pilot)
        screen = _main(app)

        assert screen.address() == f"{RUN_ID}/tokenizer"
        assert screen.points.get(f"{RUN_ID}/tokenizer", {}) == {}
        assert screen.query_one("#metrics", utrain.tui.widgets.MetricList).columns == []
        assert screen.query_one("#log", utrain.tui.widgets.LogTail).shown == []


# -- the config pane, reading ---------------------------------------------


def _form(app: utrain.tui.app.UtrainApp) -> utrain.tui.widgets.ConfigPane:
    return _main(app).query_one("#config", utrain.tui.widgets.ConfigPane)


def _rows(app: utrain.tui.app.UtrainApp) -> list[utrain.tui.widgets.FieldRow]:
    return _form(app).rows()


def _written(app: utrain.tui.app.UtrainApp, run_id: str = DRAFT_ID) -> dict[str, object]:
    with utrain.db.with_db(_main(app).data.settings) as session:
        return utrain.runs.read_config(run_id, session)


async def test_a_configuring_run_reads_the_same_as_any_other(
    draft_app: utrain.tui.app.UtrainApp,
) -> None:
    """The pane is a list of values; editing is one field at a time, in place.

    A pane full of `Input`s mounted for a run you have only put the cursor on
    says "type here" before anyone has asked to, which is what made this pane
    confusing in the first place.
    """
    async with draft_app.run_test(size=SIZE) as pilot:
        await _settle(draft_app, pilot)
        screen = _main(draft_app)

        assert screen.selected_run == DRAFT_ID
        assert screen.tab == "config"
        assert _form(draft_app).editable
        assert not any(r.editing for r in _rows(draft_app))
        assert [str(r.value) for r in _rows(draft_app)] == ["4", "fp32", "0.001", "false"]


async def test_a_started_run_shows_its_config_read_only(
    draft_app: utrain.tui.app.UtrainApp,
) -> None:
    """What a finished run was configured with is worth seeing, not editing."""
    async with draft_app.run_test(size=SIZE) as pilot:
        await _settle(draft_app, pilot)
        await pilot.press("down")
        await _settle(draft_app, pilot)
        screen = _main(draft_app)

        assert screen.selected_run == RUN_ID
        assert screen.tab == "config"
        assert not _form(draft_app).editable
        assert [str(r.value) for r in _rows(draft_app)] == ["4", "fp32", "0.001", "false"]


async def test_the_config_pane_names_the_run(draft_app: utrain.tui.app.UtrainApp) -> None:
    async with draft_app.run_test(size=SIZE) as pilot:
        await _settle(draft_app, pilot)
        form = _form(draft_app)
        summary = {row.label: row.value for row in form.query(utrain.tui.widgets.FieldRow)}

        assert summary["id"] == DRAFT_ID
        assert summary["image"] == IMAGE
        assert summary["status"] == "configuring"


async def test_the_summary_rows_are_not_cursor_stops(
    draft_app: utrain.tui.app.UtrainApp,
) -> None:
    """There is nothing to do to a run id but read it."""
    async with draft_app.run_test(size=SIZE) as pilot:
        await _settle(draft_app, pilot)
        rows = _form(draft_app).query(utrain.tui.widgets.FieldRow)

        assert [r.label for r in rows if r.can_focus] == [
            "Layers",
            "Precision",
            "Learning rate",
            "Resume",
        ]
        assert not any(r.can_focus for r in rows if r.row is None)


# -- navigation: enter goes in, escape comes out --------------------------


async def test_enter_opens_the_tab_that_is_up(app: utrain.tui.app.UtrainApp) -> None:
    """From either list, and without changing which tab that is.

    Moving the cursor has been feeding that pane all along, so `enter` is "let
    me at it" rather than "show me something else".
    """
    async with app.run_test(size=SIZE) as pilot:
        await _settle(app, pilot)

        # Runs, then phases, then the tab that is up.
        await pilot.press("enter")
        await pilot.press("enter")
        await _settle(app, pilot)
        assert app.focused is not None and app.focused.id == "config"

        await pilot.press("3")
        await pilot.press("1")
        await pilot.press("enter")
        await _settle(app, pilot)
        assert app.focused is not None and app.focused.id == "plots"


async def test_enter_on_the_metrics_list_still_toggles(app: utrain.tui.app.UtrainApp) -> None:
    """The one pane where enter already meant something."""
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        metrics = _main(app).query_one("#metrics", utrain.tui.widgets.MetricList)
        await pilot.press("m")
        await _settle(app, pilot)
        # The cursor starts on the first metric rather than nowhere, which is
        # what gives the list's own `enter` something to act on.
        assert metrics.highlighted == 0

        await pilot.press("enter")
        await _settle(app, pilot)

        assert metrics.selected == ["mfu"]


async def test_escape_returns_to_the_list_the_pane_follows(
    app: utrain.tui.app.UtrainApp,
) -> None:
    async with app.run_test(size=SIZE) as pilot:
        await _settle(app, pilot)

        await pilot.press("enter")
        await pilot.press("escape")
        await _settle(app, pilot)
        assert app.focused is not None and app.focused.id == "runs"

        await pilot.press("enter")
        await pilot.press("enter")
        await pilot.press("escape")
        await _settle(app, pilot)
        assert app.focused is not None and app.focused.id == "phases"

        await pilot.press("4")
        await pilot.press("escape")
        await _settle(app, pilot)
        assert app.focused is not None and app.focused.id == "phases"


async def test_escape_on_a_sidebar_pane_leaves_the_screen_alone(
    app: utrain.tui.app.UtrainApp,
) -> None:
    """It used to pop the dashboard and leave the app's blank default screen."""
    async with app.run_test(size=SIZE) as pilot:
        await _settle(app, pilot)
        screen = _main(app)

        await pilot.press("escape")
        await _settle(app, pilot)

        assert app.screen is screen
        assert app.focused is not None and app.focused.id == "runs"


# -- the config pane, editing one field -----------------------------------


async def test_e_opens_an_editor_on_that_row_only(draft_app: utrain.tui.app.UtrainApp) -> None:
    """In place: the row keeps its height, so nothing below it moves."""
    async with draft_app.run_test(size=SIZE) as pilot:
        await _settle(draft_app, pilot)
        before = len(_rows(draft_app))

        await pilot.press("e")
        await _settle(draft_app, pilot)

        rows = _rows(draft_app)
        assert len(rows) == before
        assert [r.editing for r in rows] == [True, False, False, False]
        assert isinstance(draft_app.focused, textual.widgets.Input)
        assert rows[0].outer_size.height == 1
        assert _main(draft_app).editing()


async def test_e_edits_the_field_under_the_row_cursor(
    draft_app: utrain.tui.app.UtrainApp,
) -> None:
    """Otherwise a long config means walking down from the top every time."""
    async with draft_app.run_test(size=SIZE) as pilot:
        await _settle(draft_app, pilot)

        await pilot.press("2")
        await pilot.press("down")
        await pilot.press("down")
        await pilot.press("down")
        await _settle(draft_app, pilot)
        assert draft_app.focused is _rows(draft_app)[2]

        await pilot.press("e")
        await _settle(draft_app, pilot)

        assert [r.editing for r in _rows(draft_app)] == [False, False, True, False]


async def test_enter_goes_one_level_deeper_each_press(
    draft_app: utrain.tui.app.UtrainApp,
) -> None:
    """Run, its phases, its config, then the first field: `e` skips to the last."""
    async with draft_app.run_test(size=SIZE) as pilot:
        await _settle(draft_app, pilot)

        await pilot.press("enter")
        await _settle(draft_app, pilot)
        assert draft_app.focused is not None and draft_app.focused.id == "phases"

        await pilot.press("enter")
        await _settle(draft_app, pilot)
        assert draft_app.focused is _form(draft_app)

        await pilot.press("enter")
        await _settle(draft_app, pilot)

        assert _rows(draft_app)[0].editing
        assert isinstance(draft_app.focused, textual.widgets.Input)


async def test_typing_a_value_and_committing_writes_it(
    draft_app: utrain.tui.app.UtrainApp,
) -> None:
    """The whole path by keypress: cursor, edit, type, enter."""
    async with draft_app.run_test(size=SIZE) as pilot:
        await _settle(draft_app, pilot)

        await pilot.press("e")
        await _settle(draft_app, pilot)
        # An `Input` selects its content when focused, so this replaces the 4.
        await pilot.press("8")
        await pilot.press("enter")
        await _settle(draft_app, pilot)

        assert _written(draft_app)["globals"] == {"model": {"n_layer": 8, "dtype": "fp32"}}
        rows = _rows(draft_app)
        assert not any(r.editing for r in rows)
        assert str(rows[0].value) == "8"
        assert draft_app.focused is rows[0]


async def test_two_edits_in_a_row_both_land(draft_app: utrain.tui.app.UtrainApp) -> None:
    """The second write carries the first: the rows are what it is built from.

    Without `applied`, the row would still hold the file's old value until the
    next refresh, and the second write would undo the first.
    """
    async with draft_app.run_test(size=SIZE) as pilot:
        await _settle(draft_app, pilot)

        await pilot.press("e")
        await pilot.press("8")
        await pilot.press("enter")
        await pilot.press("down")
        await pilot.press("down")
        await pilot.press("e")
        await pilot.press("0", ".", "5")
        await pilot.press("enter")
        await _settle(draft_app, pilot)

        written = _written(draft_app)
        assert written["globals"] == {"model": {"n_layer": 8, "dtype": "fp32"}}
        assert written["phases"] == {"pretrain": {"lr": 0.5, "resume": False}}


async def test_escape_closes_the_editor_and_writes_nothing(
    draft_app: utrain.tui.app.UtrainApp,
) -> None:
    async with draft_app.run_test(size=SIZE) as pilot:
        await _settle(draft_app, pilot)
        screen = _main(draft_app)

        await pilot.press("e")
        await pilot.press("8")
        await pilot.press("escape")
        await _settle(draft_app, pilot)

        assert draft_app.screen is screen
        rows = _rows(draft_app)
        assert not any(r.editing for r in rows)
        assert str(rows[0].value) == "4"
        assert draft_app.focused is rows[0]
        assert _written(draft_app)["globals"] == {"model": {"n_layer": 4, "dtype": "fp32"}}


async def test_an_out_of_range_value_is_reported_not_written(
    draft_app: utrain.tui.app.UtrainApp,
) -> None:
    """The editor stays open, because the thing to do about it is fix it."""
    async with draft_app.run_test(size=SIZE) as pilot:
        await _settle(draft_app, pilot)
        screen = _main(draft_app)

        await pilot.press("e")
        await pilot.press("9", "9", "9")
        await pilot.press("enter")
        await _settle(draft_app, pilot)

        assert "at most 48" in screen.error
        assert _rows(draft_app)[0].editing
        assert _written(draft_app)["globals"] == {"model": {"n_layer": 4, "dtype": "fp32"}}


async def test_a_bool_flips_on_enter(draft_app: utrain.tui.app.UtrainApp) -> None:
    """Two states do not need a widget to pick between them."""
    async with draft_app.run_test(size=SIZE) as pilot:
        await _settle(draft_app, pilot)

        await pilot.press("2")
        await pilot.press("up")
        await _settle(draft_app, pilot)
        assert draft_app.focused is _rows(draft_app)[3]

        await pilot.press("e")
        await _settle(draft_app, pilot)

        assert not any(r.editing for r in _rows(draft_app))
        assert str(_rows(draft_app)[3].value) == "true"
        assert _written(draft_app)["phases"] == {"pretrain": {"lr": 0.001, "resume": True}}


async def test_an_enum_offers_its_options(draft_app: utrain.tui.app.UtrainApp) -> None:
    async with draft_app.run_test(size=SIZE) as pilot:
        await _settle(draft_app, pilot)

        await pilot.press("2")
        await pilot.press("down")
        await pilot.press("down")
        await pilot.press("e")
        await _settle(draft_app, pilot)
        select = _rows(draft_app)[1].query_one(textual.widgets.Select)
        assert select.expanded

        await pilot.press("down")
        await pilot.press("enter")
        await _settle(draft_app, pilot)

        assert not any(r.editing for r in _rows(draft_app))
        assert _written(draft_app)["globals"] == {"model": {"n_layer": 4, "dtype": "bf16"}}


async def test_a_refresh_does_not_disturb_an_open_editor(
    draft_app: utrain.tui.app.UtrainApp,
) -> None:
    """The rows are mounted once and their values updated in place."""
    async with draft_app.run_test(size=SIZE) as pilot:
        await _settle(draft_app, pilot)
        screen = _main(draft_app)
        await pilot.press("e")
        await pilot.press("8")
        await _settle(draft_app, pilot)
        focused = draft_app.focused

        screen.refresh_data()
        await _settle(draft_app, pilot)

        assert draft_app.focused is focused
        assert _rows(draft_app)[0].editor_value() == "8"


async def test_a_message_outlives_the_refresh_that_would_clear_it(
    draft_app: utrain.tui.app.UtrainApp,
) -> None:
    """The refresh clears the error line, which is how a fetch error heals.

    A message that answers a keypress has to survive that, or it is gone before
    it has been read.
    """
    async with draft_app.run_test(size=SIZE) as pilot:
        await _settle(draft_app, pilot)
        screen = _main(draft_app)
        await pilot.press("e")
        await pilot.press("9", "9", "9")
        await pilot.press("enter")
        await _settle(draft_app, pilot)
        assert "at most 48" in screen.error

        screen.refresh_data()
        await _settle(draft_app, pilot)

        assert "at most 48" in screen.error


async def test_a_single_key_shortcut_cannot_interrupt_an_edit(
    draft_app: utrain.tui.app.UtrainApp,
) -> None:
    """The option list a `Select` drops does not swallow printable keys.

    Without the guard, `q` typed at an open enum would quit the app instead of
    picking an option.
    """
    async with draft_app.run_test(size=SIZE) as pilot:
        await _settle(draft_app, pilot)
        screen = _main(draft_app)
        await pilot.press("2")
        await pilot.press("down")
        await pilot.press("down")
        await pilot.press("e")
        await _settle(draft_app, pilot)
        assert _rows(draft_app)[1].editing

        await pilot.press("q")
        await pilot.press("1")
        await pilot.press("i")
        await _settle(draft_app, pilot)

        assert draft_app.is_running
        assert "q" not in draft_app.screen.active_bindings
        assert draft_app.screen is screen
        assert screen.pane == "config"
        assert _rows(draft_app)[1].editing


async def test_e_is_offered_only_where_it_works(draft_app: utrain.tui.app.UtrainApp) -> None:
    """`e` is live on a configuring run, and greyed in the footer on any other."""
    async with draft_app.run_test(size=SIZE) as pilot:
        await _settle(draft_app, pilot)
        screen = _main(draft_app)
        assert screen.active_bindings["e"].enabled

        await pilot.press("down")
        await _settle(draft_app, pilot)
        assert screen.selected_run == RUN_ID
        # Still in the footer, greyed: that is the answer to "why can I not
        # edit this one".
        assert not screen.active_bindings["e"].enabled

        await pilot.press("e")
        await _settle(draft_app, pilot)
        assert not screen.editing()


async def test_tab_leaves_the_config_pane_for_the_sidebar(
    draft_app: utrain.tui.app.UtrainApp,
) -> None:
    """`tab` is one cycle over every pane, so the last one wraps to the first."""
    async with draft_app.run_test(size=SIZE) as pilot:
        await _settle(draft_app, pilot)
        screen = _main(draft_app)
        await pilot.press("2")
        await _settle(draft_app, pilot)
        assert screen.pane == "config"

        await pilot.press("tab")
        await _settle(draft_app, pilot)

        assert screen.pane == "runs"


# -- the podman cache -----------------------------------------------------


async def test_polling_does_not_shell_out_to_podman(
    app: utrain.tui.app.UtrainApp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reason the describe cache exists.

    `podman.describe` starts a container. Without the cache a dashboard
    refreshing once a second would start one per second, so this pins that a
    seeded cache means no calls at all.
    """

    def explode(*args: object, **kwargs: object) -> object:
        raise AssertionError("the TUI shelled out to podman while polling")

    monkeypatch.setattr(utrain.container.podman, "describe", explode)
    monkeypatch.setattr(utrain.container.podman, "list_presets", explode)

    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        # Sit on the dashboard through several refresh ticks.
        await _settle(app, pilot)
        await _settle(app, pilot)

        assert _main(app).points[f"{RUN_ID}/pretrain"]["loss"]


# -- the content panes ----------------------------------------------------


async def test_the_plot_and_log_panes_can_be_focused(app: utrain.tui.app.UtrainApp) -> None:
    """Otherwise there is no way to scroll either of them from the keyboard."""
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)

        await pilot.press("3")
        await pilot.pause()
        assert app.focused is not None and app.focused.id == "plots"

        await pilot.press("4")
        await pilot.pause()
        assert app.focused is not None and app.focused.id == "log"


async def test_tab_cycles_every_pane_on_screen(app: utrain.tui.app.UtrainApp) -> None:
    """One cycle over both columns: `tab` has to mean "the next thing".

    What it walks is the list, at whichever level it is at, and whichever
    content pane the tab is on -- nothing else is on screen, so nothing else is
    a stop.
    """
    async with app.run_test(size=SIZE) as pilot:
        await _settle(app, pilot)
        screen = _main(app)
        assert screen.pane == "runs"

        visited = []
        for _ in range(3):
            await pilot.press("tab")
            await _settle(app, pilot)
            visited.append(screen.pane)

        assert visited == ["config", "runs", "config"]


async def test_the_metric_picker_joins_the_cycle_while_it_is_open(
    app: utrain.tui.app.UtrainApp,
) -> None:
    """`tab` reaches the plots without putting the picker away.

    Checking a metric is asking to see a curve, so the picker has to still be
    in hand once the curve is drawn.
    """
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        screen = _main(app)
        await pilot.press("m")
        await _settle(app, pilot)
        assert screen.pane == "metrics"

        await pilot.press("tab")
        await _settle(app, pilot)

        assert screen.pane == "phases"
        assert screen.metrics_open()


async def test_tab_leaves_the_content_column_for_the_sidebar(
    app: utrain.tui.app.UtrainApp,
) -> None:
    """The content panes are no longer a loop of their own."""
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        screen = _main(app)
        await pilot.press("3")
        await _settle(app, pilot)
        assert screen.pane == "plots"

        visited = []
        for _ in range(2):
            await pilot.press("tab")
            await _settle(app, pilot)
            visited.append(screen.pane)

        assert visited == ["phases", "plots"]


async def test_shift_tab_cycles_the_other_way(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=SIZE) as pilot:
        await _settle(app, pilot)
        screen = _main(app)
        assert screen.pane == "runs"

        # Backwards from the first pane is the last one on screen, which with
        # the config tab up is the config.
        await pilot.press("shift+tab")
        await _settle(app, pilot)
        assert screen.pane == "config"

        await pilot.press("shift+tab")
        await _settle(app, pilot)
        assert screen.pane == "runs"


async def test_each_number_selects_its_tab_and_goes_to_it(
    app: utrain.tui.app.UtrainApp,
) -> None:
    """One key, both halves: naming a tab and staying in the sidebar is not
    something a viewer could have meant."""
    async with app.run_test(size=SIZE) as pilot:
        await _settle(app, pilot)
        screen = _main(app)

        for key, tab in (("3", "plots"), ("4", "log"), ("2", "config")):
            await pilot.press(key)
            await _settle(app, pilot)
            assert screen.tab == tab
            assert screen.pane == tab


async def test_the_sidebar_cursor_does_not_change_the_tab(
    app: utrain.tui.app.UtrainApp,
) -> None:
    """The whole point of the tabs: the content column is the viewer's choice.

    Descending into a run changes whose logs are shown, never that logs are
    what is shown. A phase that has a tab of its own is the one exception, and
    it is still the viewer's choice -- see
    `test_each_phase_comes_back_to_the_tab_it_was_left_on`.
    """
    async with app.run_test(size=SIZE) as pilot:
        await _settle(app, pilot)
        screen = _main(app)
        await pilot.press("4")
        await _settle(app, pilot)
        assert screen.tab == "log"

        await pilot.press("1")
        await pilot.press("enter")
        await _settle(app, pilot)

        assert screen.pane == "phases"
        assert screen.tab == "log"
        assert screen.query_one("#log", utrain.tui.widgets.LogTail).display


async def test_a_panes_key_is_inert_from_another_pane(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        screen = _main(app)
        metrics = screen.query_one("#metrics", utrain.tui.widgets.MetricList)
        assert screen.pane == "phases"

        await pilot.press("space")
        await pilot.press("b")
        await _settle(app, pilot)

        assert sorted(metrics.selected) == ["loss", "mfu"]
        assert all(p.charset == utrain.tui.render.CHARSET_BLOCK for p in screen.plots.values())


async def test_the_footer_shows_the_focused_panes_keys(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        assert "space" not in _main(app).active_bindings

        await pilot.press("m")
        await _settle(app, pilot)
        assert "space" in _main(app).active_bindings


async def test_the_tab_strip_names_the_three_and_marks_the_one_that_is_up(
    app: utrain.tui.app.UtrainApp,
) -> None:
    """The numbers live here now, which is what frees the pane titles to say
    whose config or whose curves they are showing."""
    async with app.run_test(size=SIZE) as pilot:
        await _settle(app, pilot)
        screen = _main(app)
        strip = screen.query_one("#tabs", textual.widgets.Static)

        assert "2 Config" in str(strip.render())
        assert "3 Plots" in str(strip.render())
        assert "4 Logs" in str(strip.render())
        # Not a number in sight: the pane title is free to say what it holds.
        assert not str(screen.query_one("#config").border_title).startswith("4 ")


async def test_the_pane_numbers_are_not_in_the_footer(app: utrain.tui.app.UtrainApp) -> None:
    """The pane titles say them; the footer has better uses for the room."""
    async with app.run_test(size=SIZE) as pilot:
        await _settle(app, pilot)
        shown = {key for key, binding in _main(app).active_bindings.items() if binding.binding.show}
        assert not shown & {"1", "2", "3", "4", "5"}


async def test_b_switches_the_plots_to_braille(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        screen = _main(app)
        assert all(p.charset == utrain.tui.render.CHARSET_BLOCK for p in screen.plots.values())

        await pilot.press("3")
        await pilot.press("b")
        await _settle(app, pilot)
        assert all(p.charset == utrain.tui.render.CHARSET_BRAILLE for p in screen.plots.values())

        await pilot.press("b")
        await _settle(app, pilot)
        assert all(p.charset == utrain.tui.render.CHARSET_BLOCK for p in screen.plots.values())


async def test_the_plots_are_drawn_without_gridlines(
    app: utrain.tui.app.UtrainApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    """uniplot's default gridline at zero lands on the frame, not in the plot.

    x is a step or an elapsed time, so it starts at zero and the line is drawn
    along the left border; a metric that bottoms out at zero puts the other one
    along the bottom. Both read as a doubled frame.
    """
    calls: list[dict[str, object]] = []
    real = uniplot.plot_gen

    def record(**kwargs: object) -> object:
        calls.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(uniplot, "plot_gen", record)

    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)

    assert calls
    assert all(c["x_gridlines"] == [] and c["y_gridlines"] == [] for c in calls)


async def test_a_plot_mounted_later_gets_the_chosen_character_set(
    app: utrain.tui.app.UtrainApp,
) -> None:
    """The key is pressed before the phase with metrics is even selected."""
    async with app.run_test(size=SIZE) as pilot:
        await _settle(app, pilot)
        # Off the runs pane first, so the content column is the plots and not
        # the config; then onto the plots pane, which is what owns `b`.
        await pilot.press("enter")
        await pilot.press("3")
        await pilot.press("b")
        await _settle(app, pilot)
        await _select_pretrain(app, pilot)

        plots = _main(app).plots
        assert plots and all(p.charset == utrain.tui.render.CHARSET_BRAILLE for p in plots.values())


# -- starting and stopping ------------------------------------------------


async def test_s_starts_the_configuring_run_under_the_cursor(
    draft_app: utrain.tui.app.UtrainApp,
) -> None:
    async with draft_app.run_test(size=SIZE) as pilot:
        await _settle(draft_app, pilot)
        assert _main(draft_app).selected_run == DRAFT_ID

        await pilot.press("s")
        await _settle(draft_app, pilot)

        # The full id, not the prefix the runs table displays.
        assert _recording(draft_app).started == [DRAFT_ID]


async def test_s_is_greyed_on_a_run_that_cannot_be_started(
    draft_app: utrain.tui.app.UtrainApp,
) -> None:
    """Greyed rather than gone: that is the answer to "why not this one"."""
    async with draft_app.run_test(size=SIZE) as pilot:
        await _settle(draft_app, pilot)
        screen = _main(draft_app)
        assert screen.active_bindings["s"].enabled

        await pilot.press("down")
        await _settle(draft_app, pilot)
        assert screen.selected_run == RUN_ID
        assert not screen.active_bindings["s"].enabled

        await pilot.press("s")
        await _settle(draft_app, pilot)
        assert _recording(draft_app).started == []


async def test_shift_s_asks_before_stopping(live_app: utrain.tui.app.UtrainApp) -> None:
    async with live_app.run_test(size=SIZE) as pilot:
        await _settle(live_app, pilot)
        assert _main(live_app).selected_run == LIVE_ID

        await pilot.press("S")
        await _settle(live_app, pilot)

        assert isinstance(live_app.screen, utrain.tui.screens.ConfirmScreen)
        assert _recording(live_app).stopped == []
        # Not just that the screen is up: that it has drawn something. A
        # container sized `auto` around children that are `1fr` resolves to
        # nothing, which is a prompt you cannot read but can still answer.
        question = live_app.screen.query_one("#confirm-question", textual.widgets.Static)
        assert "live" in str(question.render())
        assert question.size.width > 0 and question.size.height > 0
        # `No` is what a destructive prompt should open on.
        assert live_app.focused is not None
        assert live_app.focused.id == "confirm-no"

        await pilot.press("left")
        await _settle(live_app, pilot)
        assert live_app.focused is not None
        assert live_app.focused.id == "confirm-yes"

        await pilot.press("enter")
        await _settle(live_app, pilot)

        assert _recording(live_app).stopped == [LIVE_ID]


async def test_enter_on_the_focused_no_stops_nothing(live_app: utrain.tui.app.UtrainApp) -> None:
    async with live_app.run_test(size=SIZE) as pilot:
        await _settle(live_app, pilot)
        await pilot.press("S")
        await _settle(live_app, pilot)

        # `enter` without moving: the focused button is `No`.
        await pilot.press("enter")
        await _settle(live_app, pilot)

        assert isinstance(live_app.screen, utrain.tui.screens.MainScreen)
        assert _recording(live_app).stopped == []


async def test_escape_dismisses_the_prompt(live_app: utrain.tui.app.UtrainApp) -> None:
    async with live_app.run_test(size=SIZE) as pilot:
        await _settle(live_app, pilot)
        await pilot.press("S")
        await _settle(live_app, pilot)

        await pilot.press("escape")
        await _settle(live_app, pilot)

        assert isinstance(live_app.screen, utrain.tui.screens.MainScreen)
        assert _recording(live_app).stopped == []


async def test_q_at_the_prompt_does_not_quit(live_app: utrain.tui.app.UtrainApp) -> None:
    """`q` is an app binding, so it is global unless switched off."""
    async with live_app.run_test(size=SIZE) as pilot:
        await _settle(live_app, pilot)
        await pilot.press("S")
        await _settle(live_app, pilot)

        await pilot.press("q")
        await _settle(live_app, pilot)

        assert isinstance(live_app.screen, utrain.tui.screens.ConfirmScreen)


async def test_shift_s_is_greyed_on_a_run_that_is_not_running(
    live_app: utrain.tui.app.UtrainApp,
) -> None:
    async with live_app.run_test(size=SIZE) as pilot:
        await _settle(live_app, pilot)
        screen = _main(live_app)
        assert screen.active_bindings["S"].enabled

        await pilot.press("down")
        await _settle(live_app, pilot)
        assert screen.selected_run == DRAFT_ID
        assert not screen.active_bindings["S"].enabled

        await pilot.press("S")
        await _settle(live_app, pilot)
        assert isinstance(live_app.screen, utrain.tui.screens.MainScreen)
        assert _recording(live_app).stopped == []


async def test_the_lifecycle_keys_follow_the_run_not_the_focus(
    live_app: utrain.tui.app.UtrainApp,
) -> None:
    """They act on the selected run, and which run that is does not depend on
    where the focus happens to be -- so they do not come and go with it."""
    async with live_app.run_test(size=SIZE) as pilot:
        await _settle(live_app, pilot)
        await pilot.press("enter")
        await _settle(live_app, pilot)
        screen = _main(live_app)
        assert screen.level == "phases"

        assert screen.active_bindings["S"].enabled

        await pilot.press("m")
        await _settle(live_app, pilot)
        assert screen.pane == "metrics"
        assert screen.active_bindings["S"].enabled


async def test_shift_s_stops_the_run_from_the_phases_pane(
    live_app: utrain.tui.app.UtrainApp,
) -> None:
    """Standing on the phase that is running is where you decide to abandon it.

    It stops the run, not the phase: there is no stopping one phase, and the
    prompt names the run so nobody reads it as stopping less than it does.
    """
    async with live_app.run_test(size=SIZE) as pilot:
        await _select_pretrain(live_app, pilot)
        assert _main(live_app).pane == "phases"

        await pilot.press("S")
        await _settle(live_app, pilot)
        question = live_app.screen.query_one("#confirm-question", textual.widgets.Static)
        assert "live" in str(question.render())

        await pilot.press("left")
        await pilot.press("enter")
        await _settle(live_app, pilot)

        assert _recording(live_app).stopped == [LIVE_ID]


async def test_shift_s_typed_into_an_editor_is_a_value_not_a_stop(
    draft_app: utrain.tui.app.UtrainApp,
) -> None:
    async with draft_app.run_test(size=SIZE) as pilot:
        await _settle(draft_app, pilot)
        await pilot.press("2")
        await pilot.press("down")
        await pilot.press("e")
        await _settle(draft_app, pilot)
        assert _main(draft_app).editing()

        await pilot.press("S")
        await _settle(draft_app, pilot)

        assert isinstance(draft_app.screen, utrain.tui.screens.MainScreen)
        assert _recording(draft_app).stopped == []
        assert _main(draft_app).editing()


async def test_a_lifecycle_error_reaches_the_error_line(
    draft_app: utrain.tui.app.UtrainApp,
) -> None:
    """The worker runs off the message loop, so its failure has to be marshalled."""
    async with draft_app.run_test(size=SIZE) as pilot:
        await _settle(draft_app, pilot)
        _recording(draft_app).fail = "nope"

        await pilot.press("s")
        await _settle(draft_app, pilot)

        assert _main(draft_app).error == "nope"


# -- letting the cursor settle --------------------------------------------
#
# These put the real `refresh_soon` back, but only once the app has loaded:
# the first fetch selects a run, which settles like any other move, and waiting
# that out on the clock would race the suite's own parallelism.


async def _wait_out_settle(app: utrain.tui.app.UtrainApp, pilot: typing.Any) -> None:
    """Sit through the settle window, then let the fetch it starts land."""
    await pilot.pause(utrain.tui.screens._SETTLE_SECONDS * 2)
    await _settle(app, pilot)


def _settling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(utrain.tui.screens.MainScreen, "refresh_soon", _REFRESH_SOON)


async def test_a_burst_of_cursor_moves_fetches_only_where_it_lands(
    live_app: utrain.tui.app.UtrainApp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scanning a list costs one fetch, not one per row.

    A fetch reconciles every run against its directory, which is a write, so
    the runs crossed on the way must not be fetched at all.

    The moves are made back to back in one step rather than through the pilot:
    what is being asserted is that two selections inside the window collapse to
    one fetch, and pressing keys would leave that to the clock.
    """
    async with live_app.run_test(size=SIZE) as pilot:
        await _settle(live_app, pilot)
        _settling(monkeypatch)
        source = _recording(live_app)
        source.fetched.clear()

        screen = _main(live_app)
        screen.select_run(DRAFT_ID)
        screen.select_run(RUN_ID)
        await _wait_out_settle(live_app, pilot)

        assert screen.selected_run == RUN_ID
        # By run, not by count: landing on a run with phases costs a second
        # fetch for the first of them, which is the phase selection doing its
        # own settling. What matters is that the draft crossed on the way was
        # never fetched at all.
        assert set(source.fetched) == {RUN_ID}


async def test_a_single_move_still_fetches_once_the_cursor_stops(
    live_app: utrain.tui.app.UtrainApp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The delay is a settle, not a drop, and the arrow keys go through it."""
    async with live_app.run_test(size=SIZE) as pilot:
        await _settle(live_app, pilot)
        _settling(monkeypatch)
        source = _recording(live_app)
        source.fetched.clear()

        await pilot.press("down")
        await _wait_out_settle(live_app, pilot)

        assert _main(live_app).selected_run == DRAFT_ID
        # By run rather than by count, as above: the refresh tick is still
        # running, so what is asserted is that the move fetched, and fetched
        # for the row it landed on.
        assert set(source.fetched) == {DRAFT_ID}


async def test_the_phases_pane_holds_its_rows_until_the_new_ones_arrive(
    live_app: utrain.tui.app.UtrainApp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stale but stable, rather than blinking to empty on every move.

    The running run has two phases and the draft under it has none, so a pane
    that blanked while it waited would be visibly empty in between.
    """
    async with live_app.run_test(size=SIZE) as pilot:
        await _settle(live_app, pilot)
        _settling(monkeypatch)
        screen = _main(live_app)
        table = screen.table("#phases")
        assert table.row_count == 2

        # Selected, but not yet fetched for: the cursor is on the draft and the
        # pane still holds what it had.
        screen.select_run(DRAFT_ID)
        assert screen.selected_run == DRAFT_ID
        assert table.row_count == 2

        await _wait_out_settle(live_app, pilot)
        assert table.row_count == 0


async def test_a_snapshot_of_another_run_does_not_repaint_its_panes(
    app: utrain.tui.app.UtrainApp,
) -> None:
    """A thread worker cannot be interrupted, so a stale fetch still lands.

    `exclusive` stops the next fetch starting, not this one arriving, and what
    arrives is a whole run's panes -- which is what used to strobe.
    """
    async with app.run_test(size=SIZE) as pilot:
        await _settle(app, pilot)
        screen = _main(app)
        before = list(screen.phases)
        assert [p.phase for p in before] == ["tokenizer", "pretrain"]
        log_before = screen.query_one("#log", utrain.tui.widgets.LogTail).shown

        fresh = app.data.snapshot(RUN_ID, None, None)
        stale = dataclasses.replace(fresh, run_id="z" * 32, phases=[], log=["gone"])
        screen.apply(stale)
        await pilot.pause()

        assert screen.phases == before
        assert screen.query_one("#log", utrain.tui.widgets.LogTail).shown == log_before
        # The runs list is not about any one run, so it is applied regardless.
        assert screen.runs is stale.runs


# -- creating, deleting and restarting ------------------------------------


async def _dialog(app: utrain.tui.app.UtrainApp) -> utrain.tui.screens.NewRunScreen:
    screen = app.screen
    assert isinstance(screen, utrain.tui.screens.NewRunScreen)
    return screen


async def test_n_opens_the_new_run_dialog(draft_app: utrain.tui.app.UtrainApp) -> None:
    async with draft_app.run_test(size=SIZE) as pilot:
        await _settle(draft_app, pilot)

        await pilot.press("n")
        await _settle(draft_app, pilot)

        screen = await _dialog(draft_app)
        # Drawn, not merely mounted: a dialog sized from `1fr` children resolves
        # to nothing, which is a form you cannot see but can still submit.
        assert screen.query_one("#new-run").size.width > 0
        # The name is where the cursor starts -- it is the only field with
        # nothing to default to.
        assert draft_app.focused is not None
        assert draft_app.focused.id == "new-run-name"
        # Both `Select`s answer for themselves from the moment they open.
        assert screen.query_one("#new-run-image", textual.widgets.Select).value == IMAGE
        assert screen.query_one("#new-run-compute", textual.widgets.Select).value == "cpu"
        assert _recording(draft_app).created == []


async def test_escape_closes_the_dialog_and_creates_nothing(
    draft_app: utrain.tui.app.UtrainApp,
) -> None:
    async with draft_app.run_test(size=SIZE) as pilot:
        await _settle(draft_app, pilot)
        await pilot.press("n")
        await _settle(draft_app, pilot)

        await pilot.press("escape")
        await _settle(draft_app, pilot)

        assert isinstance(draft_app.screen, utrain.tui.screens.MainScreen)
        assert _recording(draft_app).created == []


async def test_filling_the_dialog_creates_the_run_and_selects_it(
    draft_app: utrain.tui.app.UtrainApp,
) -> None:
    async with draft_app.run_test(size=SIZE) as pilot:
        await _settle(draft_app, pilot)
        await pilot.press("n")
        await _settle(draft_app, pilot)

        # `enter` in the name field is what creates: the buttons are there, but
        # typing a name and pressing enter is the shortest way through.
        await pilot.press(*"fresh")
        await pilot.press("enter")
        await _settle(draft_app, pilot)

        assert isinstance(draft_app.screen, utrain.tui.screens.MainScreen)
        assert _recording(draft_app).created == [("fresh", IMAGE, "cpu")]
        # The run the viewer just made is the one under the cursor, so its
        # config -- the only thing there is to do with it next -- is on screen.
        await _settle(draft_app, pilot)
        screen = _main(draft_app)
        assert screen.selected_run == NEW_ID
        assert screen._pending_run is None


async def test_a_run_needs_a_name(draft_app: utrain.tui.app.UtrainApp) -> None:
    """Answered in the dialog rather than by `create_run` refusing it later."""
    async with draft_app.run_test(size=SIZE) as pilot:
        await _settle(draft_app, pilot)
        await pilot.press("n")
        await _settle(draft_app, pilot)

        await pilot.press("enter")
        await _settle(draft_app, pilot)

        screen = await _dialog(draft_app)
        assert "name" in str(screen.query_one("#new-run-error", textual.widgets.Static).render())
        assert _recording(draft_app).created == []


async def test_n_with_no_images_says_so(draft_app: utrain.tui.app.UtrainApp) -> None:
    async with draft_app.run_test(size=SIZE) as pilot:
        await _settle(draft_app, pilot)
        _recording(draft_app).images = []

        await pilot.press("n")
        await _settle(draft_app, pilot)

        assert isinstance(draft_app.screen, utrain.tui.screens.MainScreen)
        assert "no images" in _main(draft_app).error


async def test_n_is_live_whatever_the_run_under_the_cursor_is(
    live_app: utrain.tui.app.UtrainApp,
) -> None:
    """`n` is about the list, not about a run, so no status greys it."""
    async with live_app.run_test(size=SIZE) as pilot:
        await _settle(live_app, pilot)
        screen = _main(live_app)
        assert screen.selected_run == LIVE_ID
        assert screen.active_bindings["n"].enabled

        await pilot.press("down")
        await _settle(live_app, pilot)
        assert screen.selected_run == DRAFT_ID
        assert screen.active_bindings["n"].enabled


async def test_d_asks_before_deleting(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=SIZE) as pilot:
        await _settle(app, pilot)
        assert _main(app).selected_run == RUN_ID

        await pilot.press("d")
        await _settle(app, pilot)

        assert isinstance(app.screen, utrain.tui.screens.ConfirmScreen)
        assert _recording(app).deleted == []
        question = app.screen.query_one("#confirm-question", textual.widgets.Static)
        assert "tiny-shakespeare" in str(question.render())

        await pilot.press("left")
        await pilot.press("enter")
        await _settle(app, pilot)

        assert _recording(app).deleted == [RUN_ID]


async def test_a_declined_delete_deletes_nothing(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=SIZE) as pilot:
        await _settle(app, pilot)
        await pilot.press("d")
        await _settle(app, pilot)

        # `No` holds the focus, so `enter` without moving is the safe answer.
        await pilot.press("enter")
        await _settle(app, pilot)

        assert isinstance(app.screen, utrain.tui.screens.MainScreen)
        assert _recording(app).deleted == []


async def test_d_is_greyed_on_a_running_run(live_app: utrain.tui.app.UtrainApp) -> None:
    """Greyed, not forced: `S` is the one prompt that kills an attempt."""
    async with live_app.run_test(size=SIZE) as pilot:
        await _settle(live_app, pilot)
        screen = _main(live_app)
        assert screen.selected_run == LIVE_ID
        assert not screen.active_bindings["d"].enabled

        await pilot.press("d")
        await _settle(live_app, pilot)
        assert isinstance(live_app.screen, utrain.tui.screens.MainScreen)
        assert _recording(live_app).deleted == []

        await pilot.press("down")
        await _settle(live_app, pilot)
        assert screen.selected_run == DRAFT_ID
        assert screen.active_bindings["d"].enabled


async def test_a_delete_leaves_the_cursor_where_it_was(
    live_app: utrain.tui.app.UtrainApp,
) -> None:
    """The rows below the deleted one shift up, so the selection has to follow.

    Without this the cursor sits on one run while the panes are about another:
    a cursor that does not move raises no `RowHighlighted`, and the row under
    it changed identity rather than position.
    """
    async with live_app.run_test(size=SIZE) as pilot:
        await _settle(live_app, pilot)
        await pilot.press("down")
        await _settle(live_app, pilot)
        screen = _main(live_app)
        assert screen.selected_run == DRAFT_ID

        await pilot.press("d")
        await _settle(live_app, pilot)
        await pilot.press("left")
        await pilot.press("enter")
        await _settle(live_app, pilot)
        await _settle(live_app, pilot)

        assert _recording(live_app).deleted == [DRAFT_ID]
        assert [r.id for r in screen.runs] == [LIVE_ID, RUN_ID]
        # Still row 1, which is now the finished run.
        assert screen.selected_run == RUN_ID
        assert live_app.screen.query_one("#runs", textual.widgets.DataTable).cursor_row == 1


async def test_shift_r_restarts_from_the_top_on_the_runs_pane(
    app: utrain.tui.app.UtrainApp,
) -> None:
    async with app.run_test(size=SIZE) as pilot:
        await _settle(app, pilot)
        assert _main(app).pane == "runs"

        await pilot.press("R")
        await _settle(app, pilot)
        assert isinstance(app.screen, utrain.tui.screens.ConfirmScreen)

        await pilot.press("left")
        await pilot.press("enter")
        await _settle(app, pilot)

        # No phase: the runs pane is not pointing at one, and restarting from
        # whichever the cursor last left behind is not what `R` means there.
        assert _recording(app).restarted == [(RUN_ID, None)]


async def test_shift_r_restarts_from_the_phase_under_the_cursor(
    app: utrain.tui.app.UtrainApp,
) -> None:
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        screen = _main(app)
        assert screen.pane == "phases"
        assert screen.selected_phase == "pretrain"

        await pilot.press("R")
        await _settle(app, pilot)
        question = app.screen.query_one("#confirm-question", textual.widgets.Static)
        assert "pretrain" in str(question.render())

        await pilot.press("left")
        await pilot.press("enter")
        await _settle(app, pilot)

        assert _recording(app).restarted == [(RUN_ID, "pretrain")]


async def test_shift_r_is_greyed_on_a_run_that_never_started(
    draft_app: utrain.tui.app.UtrainApp,
) -> None:
    async with draft_app.run_test(size=SIZE) as pilot:
        await _settle(draft_app, pilot)
        screen = _main(draft_app)
        assert screen.selected_run == DRAFT_ID
        assert not screen.active_bindings["R"].enabled

        await pilot.press("R")
        await _settle(draft_app, pilot)
        assert isinstance(draft_app.screen, utrain.tui.screens.MainScreen)
        assert _recording(draft_app).restarted == []

        await pilot.press("down")
        await _settle(draft_app, pilot)
        assert screen.selected_run == RUN_ID
        assert screen.active_bindings["R"].enabled


async def test_new_is_the_one_key_the_phases_level_does_not_offer(
    app: utrain.tui.app.UtrainApp,
) -> None:
    """`n` is about the list, and only the runs level is a list of runs.

    Delete and restart act on the selected run, which is as unambiguous inside
    it as it is outside, so they stay.
    """
    async with app.run_test(size=SIZE) as pilot:
        await _settle(app, pilot)
        await pilot.press("enter")
        await _settle(app, pilot)
        screen = _main(app)
        assert screen.level == "phases"

        assert "n" not in screen.active_bindings
        assert screen.active_bindings["d"].enabled
        assert screen.active_bindings["R"].enabled

        await pilot.press("m")
        await _settle(app, pilot)
        assert screen.pane == "metrics"
        assert "n" not in screen.active_bindings
        assert screen.active_bindings["R"].enabled


async def test_the_new_keys_typed_into_an_editor_are_values(
    draft_app: utrain.tui.app.UtrainApp,
) -> None:
    """The twin of the `S` case: an `n` or a `d` at a field is a character."""
    async with draft_app.run_test(size=SIZE) as pilot:
        await _settle(draft_app, pilot)
        await pilot.press("2")
        await pilot.press("down")
        await pilot.press("e")
        await _settle(draft_app, pilot)
        assert _main(draft_app).editing()

        await pilot.press("n", "d", "R")
        await _settle(draft_app, pilot)

        assert isinstance(draft_app.screen, utrain.tui.screens.MainScreen)
        source = _recording(draft_app)
        assert source.created == [] and source.deleted == [] and source.restarted == []
        assert _main(draft_app).editing()


async def test_a_delete_error_reaches_the_error_line(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=SIZE) as pilot:
        await _settle(app, pilot)
        _recording(app).fail = "nope"

        await pilot.press("d")
        await _settle(app, pilot)
        await pilot.press("left")
        await pilot.press("enter")
        await _settle(app, pilot)

        assert _main(app).error == "nope"


# -- what the footer says about the selected run --------------------------


async def test_the_footer_follows_the_run_the_cursor_is_on(
    live_app: utrain.tui.app.UtrainApp,
) -> None:
    """The keys grey and ungrey as the cursor crosses runs in different states.

    On the widget, not on `active_bindings`: the footer is composed once and
    recomposed only when the screen publishes that its bindings changed, and
    Textual publishes that on a focus change and on nothing else. Moving the
    run cursor is not a focus change.
    """
    async with live_app.run_test(size=SIZE) as pilot:
        await _settle(live_app, pilot)
        screen = _main(live_app)

        assert screen.selected_run_row().status == "running"
        assert _footer(live_app)["S"] and not _footer(live_app)["s"]
        assert not _footer(live_app)["e"]

        await pilot.press("down")
        await _settle(live_app, pilot)
        assert screen.selected_run_row().status == "configuring"
        assert _footer(live_app)["s"] and not _footer(live_app)["S"]
        assert _footer(live_app)["e"]

        await pilot.press("down")
        await _settle(live_app, pilot)
        assert screen.selected_run_row().status == "done"
        keys = _footer(live_app)
        assert not keys["s"] and not keys["S"] and not keys["e"]


async def test_the_footer_keeps_the_lifecycle_keys_across_the_levels(
    live_app: utrain.tui.app.UtrainApp,
) -> None:
    """The footer describes the run, so it does not reshuffle on the way down."""
    async with live_app.run_test(size=SIZE) as pilot:
        await _settle(live_app, pilot)
        assert _footer(live_app)["S"]

        await pilot.press("enter")
        await _settle(live_app, pilot)
        assert _main(live_app).level == "phases"
        assert _footer(live_app)["S"]

        await pilot.press("escape")
        await _settle(live_app, pilot)
        assert _footer(live_app)["S"]


async def test_the_footer_greys_stop_when_the_run_ends_on_its_own(
    live_app: utrain.tui.app.UtrainApp,
    tmp_path: pathlib.Path,
) -> None:
    """Nobody pressed a key: the run finished under a stationary cursor."""
    async with live_app.run_test(size=SIZE) as pilot:
        await _settle(live_app, pilot)
        screen = _main(live_app)
        assert _footer(live_app)["S"]

        with utrain.db.with_db(utrain.config.Settings(data_dir=tmp_path)) as session:
            for table in (utrain.db.run_phases, utrain.db.run_attempts):
                session.execute(
                    sqlalchemy.update(table).where(table.c.run_id == LIVE_ID).values(status="done")
                )
            session.execute(
                sqlalchemy.update(utrain.db.runs)
                .where(utrain.db.runs.c.id == LIVE_ID)
                .values(status="done")
            )

        # Straight to a fetch rather than waiting out the 1s timer.
        screen.refresh_data()
        await _settle(live_app, pilot)

        assert screen.selected_run_row().status == "done"
        assert not _footer(live_app)["S"]


async def test_the_footer_offers_stop_once_the_run_is_started(
    draft_app: utrain.tui.app.UtrainApp,
    tmp_path: pathlib.Path,
) -> None:
    """`s` hands over to `S` on the run you just started."""
    async with draft_app.run_test(size=SIZE) as pilot:
        await _settle(draft_app, pilot)
        screen = _main(draft_app)
        assert _footer(draft_app)["s"] and not _footer(draft_app)["S"]

        await pilot.press("s")
        await _settle(draft_app, pilot)
        assert _recording(draft_app).started == [DRAFT_ID]

        # `Data.start_run` is recorded, not made, so stand in for what it would
        # have written and let the next fetch pick it up.
        with utrain.db.with_db(utrain.config.Settings(data_dir=tmp_path)) as session:
            session.execute(
                sqlalchemy.update(utrain.db.runs)
                .where(utrain.db.runs.c.id == DRAFT_ID)
                .values(status="running")
            )
        screen.refresh_data()
        await _settle(draft_app, pilot)

        assert _footer(draft_app)["S"] and not _footer(draft_app)["s"]
        # And the run can no longer be edited.
        assert not _footer(draft_app)["e"]


async def test_the_arrows_and_tab_both_cross_the_dialog(
    live_app: utrain.tui.app.UtrainApp,
) -> None:
    """`focus_next` is an app action, so binding the arrows to it by name would
    silently do nothing -- which is how this was wrong the first time."""
    async with live_app.run_test(size=SIZE) as pilot:
        await _settle(live_app, pilot)
        await pilot.press("S")
        await _settle(live_app, pilot)

        for key, expected in [
            ("left", "confirm-yes"),
            ("right", "confirm-no"),
            ("tab", "confirm-yes"),
        ]:
            await pilot.press(key)
            await _settle(live_app, pilot)
            assert live_app.focused is not None
            assert live_app.focused.id == expected, key

        assert _recording(live_app).stopped == []


# -- exporting a plot -----------------------------------------------------


def _export_dialog(app: utrain.tui.app.UtrainApp) -> utrain.tui.screens.ExportScreen:
    screen = app.screen
    assert isinstance(screen, utrain.tui.screens.ExportScreen)
    return screen


def _export_path(app: utrain.tui.app.UtrainApp) -> str:
    return _export_dialog(app).query_one("#export-path", textual.widgets.Input).value


async def test_shift_e_opens_the_export_dialog_named_for_the_curve(
    app: utrain.tui.app.UtrainApp,
) -> None:
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)

        await pilot.press("m")
        await pilot.press("E")
        await _settle(app, pilot)

        screen = _export_dialog(app)
        # Drawn, not merely mounted -- the point `test_n_opens_the_new_run_dialog`
        # makes about a dialog sized from `1fr` children.
        assert screen.query_one("#export").size.width > 0
        # The run, the phase and the metric under the cursor, so two runs' losses
        # do not write the same file.
        assert _export_path(app) == "tiny-shakespeare-pretrain-loss.csv"
        # The path is the field a viewer is most likely to change, and the one
        # `enter` submits from.
        assert app.focused is not None
        assert app.focused.id == "export-path"


async def test_the_export_dialog_opens_over_the_plot_pane_too(
    app: utrain.tui.app.UtrainApp,
) -> None:
    """`E` is on both panes: the metric is chosen in one and looked at in the other."""
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)

        await pilot.press("3")
        await pilot.press("E")
        await _settle(app, pilot)

        assert _export_path(app) == "tiny-shakespeare-pretrain-loss.csv"


async def test_escape_closes_the_export_dialog_and_writes_nothing(
    app: utrain.tui.app.UtrainApp, tmp_path: pathlib.Path
) -> None:
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        await pilot.press("m")
        await pilot.press("E")
        await _settle(app, pilot)

        await pilot.press("escape")
        await _settle(app, pilot)

        assert isinstance(app.screen, utrain.tui.screens.MainScreen)
        assert list(tmp_path.glob("*.csv")) == []


async def test_enter_writes_the_curve_and_says_where(
    app: utrain.tui.app.UtrainApp, tmp_path: pathlib.Path
) -> None:
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        await pilot.press("m")
        await pilot.press("E")
        await _settle(app, pilot)

        target = tmp_path / "loss.csv"
        _export_dialog(app).query_one("#export-path", textual.widgets.Input).value = str(target)
        await pilot.press("enter")
        await _settle(app, pilot)

        assert isinstance(app.screen, utrain.tui.screens.MainScreen)
        lines = target.read_text().splitlines()
        # The header plus every point the phase logged -- a csv is the data, so
        # it is not thinned the way a figure is.
        assert lines[0] == "step,loss"
        assert len(lines) == 21
        assert f"wrote {target}" in _main(app).error


async def test_the_export_is_at_full_resolution(
    app: utrain.tui.app.UtrainApp, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The widget's plot is thinned for a character grid; the file must not be.

    `build_plot` is called with `limit=0` -- `downsample`'s "leave it alone" --
    rather than reusing `MetricPlot.plot`.
    """
    limits: list[int] = []
    real = utrain.tui.render.build_plot

    def spy(
        name: str,
        points: dict[str, list[utrain.metrics.MetricPoint]],
        x_axis: str,
        limit: int = utrain.tui.render.MAX_PLOT_POINTS,
    ) -> utrain.tui.render.Plot | None:
        limits.append(limit)
        return real(name, points, x_axis, limit)

    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        monkeypatch.setattr(utrain.tui.screens.render, "build_plot", spy)
        await pilot.press("m")
        await pilot.press("E")
        await _settle(app, pilot)

        assert 0 in limits


async def test_the_exported_figure_is_the_format_that_was_picked(
    app: utrain.tui.app.UtrainApp, tmp_path: pathlib.Path
) -> None:
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        await pilot.press("m")
        await pilot.press("E")
        await _settle(app, pilot)

        screen = _export_dialog(app)
        target = tmp_path / "loss.png"
        screen.query_one("#export-path", textual.widgets.Input).value = str(target)
        picked: textual.widgets.Select[str] = screen.query_one(
            "#export-format", textual.widgets.Select
        )
        picked.value = "png"
        await pilot.press("enter")
        await _settle(app, pilot)

        assert target.read_bytes().startswith(b"\x89PNG")


async def test_picking_a_format_tracks_the_extension(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        await pilot.press("m")
        await pilot.press("E")
        await _settle(app, pilot)

        picked: textual.widgets.Select[str] = _export_dialog(app).query_one(
            "#export-format", textual.widgets.Select
        )
        picked.value = "pdf"
        await _settle(app, pilot)

        assert _export_path(app) == "tiny-shakespeare-pretrain-loss.pdf"


async def test_picking_a_format_leaves_a_typed_name_alone(
    app: utrain.tui.app.UtrainApp,
) -> None:
    """The picker rewrites the extension it last offered, not one that was typed."""
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        await pilot.press("m")
        await pilot.press("E")
        await _settle(app, pilot)

        screen = _export_dialog(app)
        screen.query_one("#export-path", textual.widgets.Input).value = "figure.dat"
        picked: textual.widgets.Select[str] = screen.query_one(
            "#export-format", textual.widgets.Select
        )
        picked.value = "svg"
        await _settle(app, pilot)

        assert _export_path(app) == "figure.dat"


async def test_an_export_with_no_path_says_so(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        await pilot.press("m")
        await pilot.press("E")
        await _settle(app, pilot)

        screen = _export_dialog(app)
        screen.query_one("#export-path", textual.widgets.Input).value = ""
        await pilot.press("enter")
        await _settle(app, pilot)

        screen = _export_dialog(app)
        error = screen.query_one("#export-error", textual.widgets.Static)
        assert "where" in str(error.render())


async def test_exporting_over_a_file_asks_first(
    app: utrain.tui.app.UtrainApp, tmp_path: pathlib.Path
) -> None:
    target = tmp_path / "loss.csv"
    target.write_text("keep me\n")
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        await pilot.press("m")
        await pilot.press("E")
        await _settle(app, pilot)

        _export_dialog(app).query_one("#export-path", textual.widgets.Input).value = str(target)
        await pilot.press("enter")
        await _settle(app, pilot)

        assert isinstance(app.screen, utrain.tui.screens.ConfirmScreen)
        assert target.read_text() == "keep me\n"

        # `No` is what a prompt that would throw work away opens on.
        await pilot.press("enter")
        await _settle(app, pilot)
        assert target.read_text() == "keep me\n"


async def test_confirming_the_overwrite_writes(
    app: utrain.tui.app.UtrainApp, tmp_path: pathlib.Path
) -> None:
    target = tmp_path / "loss.csv"
    target.write_text("keep me\n")
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        await pilot.press("m")
        await pilot.press("E")
        await _settle(app, pilot)

        _export_dialog(app).query_one("#export-path", textual.widgets.Input).value = str(target)
        await pilot.press("enter")
        await _settle(app, pilot)

        await pilot.press("left")
        await pilot.press("enter")
        await _settle(app, pilot)

        assert target.read_text().splitlines()[0] == "step,loss"


async def test_soloing_a_metric_is_what_gets_exported(app: utrain.tui.app.UtrainApp) -> None:
    """Soloing is the viewer saying which curve they are looking at."""
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        await pilot.press("m")
        await pilot.press("down")
        await pilot.press("y")
        await _settle(app, pilot)
        assert _main(app).solo == "mfu"

        await pilot.press("E")
        await _settle(app, pilot)

        assert _export_path(app) == "tiny-shakespeare-pretrain-mfu.csv"


async def test_shift_e_is_greyed_out_when_no_metric_is_drawn(
    app: utrain.tui.app.UtrainApp,
) -> None:
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        await pilot.press("m")
        # Uncheck both, so there is no curve to write.
        await pilot.press("space")
        await pilot.press("down")
        await pilot.press("space")
        await _settle(app, pilot)
        assert _main(app).plotted() == []

        assert _footer(app).get("E") is False

        await pilot.press("E")
        await _settle(app, pilot)
        assert isinstance(app.screen, utrain.tui.screens.MainScreen)


async def test_a_failed_export_stays_on_the_status_line(
    app: utrain.tui.app.UtrainApp, tmp_path: pathlib.Path
) -> None:
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        await pilot.press("m")
        await pilot.press("E")
        await _settle(app, pilot)

        target = tmp_path / "nowhere" / "loss.csv"
        _export_dialog(app).query_one("#export-path", textual.widgets.Input).value = str(target)
        await pilot.press("enter")
        await _settle(app, pilot)

        assert "no such directory" in _main(app).error


# -- chat -----------------------------------------------------------------
#
# Against the real reference container, spawned from the checkout rather than
# through podman -- the same trick `tests/test_serve_protocol.py` uses, and the
# reason these run in CI at all. Only the spawn is stood in for: the screen
# talks HTTP to a real server, over the real `utrain.chat.Client`.


class _ServingData(_RecordingData):
    """`Data` whose `start_server` runs the fake container directly."""

    def __init__(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        super().__init__(*args, **kwargs)
        self.spawn: collections.abc.Callable[[pathlib.Path], subprocess.Popen[bytes]] | None = None
        self.servers: list[utrain.serve.Server] = []
        self.served_phases: list[str | None] = []

    def start_server(self, run_id: str, phase: str | None = None) -> utrain.serve.Server:
        assert self.spawn is not None
        root = self.settings.data_dir / "serve" / run_id
        root.mkdir(parents=True, exist_ok=True)
        proc = self.spawn(root)
        # None means "the newest servable phase", which for this image is
        # `pretrain` -- the same resolution `serve.start` would do.
        self.served_phases.append(phase)
        server = utrain.serve.Server(
            proc=proc,
            log=open(root / "serve.log", "wb"),
            log_path=root / "serve.log",
            port_file=root / "serve" / "port.json",
            # Never written here: these servers are plain subprocesses, not
            # podman, so `shutdown`'s force-remove finds nothing to do.
            cid_file=root / "serve" / "container.id",
            run_name="tiny-shakespeare",
            image=IMAGE,
            phase=phase or "pretrain",
        )
        self.servers.append(server)
        return server


@pytest.fixture()
def chat_app(
    tmp_path: pathlib.Path,
    spawn_serve: collections.abc.Callable[[pathlib.Path], subprocess.Popen[bytes]],
) -> utrain.tui.app.UtrainApp:
    """The finished run and the unstarted one, with chat wired to the fake."""
    _seed(tmp_path)
    _seed_draft(tmp_path)
    source = _ServingData(
        settings=utrain.config.Settings(data_dir=tmp_path),
        describe_cache={IMAGE_ID: _describe()},
    )
    source.spawn = spawn_serve
    return utrain.tui.app.UtrainApp(source)


def _chat(app: utrain.tui.app.UtrainApp) -> utrain.tui.screens.ChatScreen:
    screen = app.screen
    assert isinstance(screen, utrain.tui.screens.ChatScreen)
    return screen


def _transcript(app: utrain.tui.app.UtrainApp) -> list[str]:
    return _chat(app).log_widget().turns


async def _open_chat(app: utrain.tui.app.UtrainApp, pilot: typing.Any) -> None:
    """`t` on the finished run, waited out until the server is listening."""
    await _settle(app, pilot)
    await pilot.press("down")  # off the draft, onto the finished run
    await _settle(app, pilot)
    await pilot.press("t")
    await _settle(app, pilot)
    # Starting the container is a worker, and so is nothing else here: waiting
    # for the workers is waiting for the port.
    for _ in range(4):
        await pilot.pause()


async def _say(app: utrain.tui.app.UtrainApp, pilot: typing.Any, text: str) -> None:
    await pilot.press(*text)
    await pilot.press("enter")
    await _settle(app, pilot)
    # The deltas are drawn on a timer rather than as they land, so the last of
    # them is one flush behind the worker that finished.
    for _ in range(6):
        await pilot.pause(0.05)


async def test_t_opens_a_chat_screen_on_the_selected_run(
    chat_app: utrain.tui.app.UtrainApp,
) -> None:
    async with chat_app.run_test(size=SIZE) as pilot:
        await _open_chat(chat_app, pilot)

        screen = _chat(chat_app)
        assert screen.run_id == RUN_ID
        # The input is live only once the container published its port.
        assert not screen.field().disabled
        assert any("serving" in line for line in _transcript(chat_app))


async def test_a_turn_lands_as_the_question_and_the_reply(
    chat_app: utrain.tui.app.UtrainApp,
) -> None:
    async with chat_app.run_test(size=SIZE) as pilot:
        await _open_chat(chat_app, pilot)
        await _say(chat_app, pilot, "hello")

        assert _transcript(chat_app)[-2:] == ["> hello", "[fake model] turn 1: 'hello'"]


async def test_the_second_turn_is_sent_with_the_first(
    chat_app: utrain.tui.app.UtrainApp,
) -> None:
    """The server keeps no history, so `turn 2` proves the client resent it."""
    async with chat_app.run_test(size=SIZE) as pilot:
        await _open_chat(chat_app, pilot)
        await _say(chat_app, pilot, "one")
        await _say(chat_app, pilot, "two")

        assert _transcript(chat_app)[-1] == "[fake model] turn 2: 'two'"


async def test_reset_forgets_the_conversation(chat_app: utrain.tui.app.UtrainApp) -> None:
    async with chat_app.run_test(size=SIZE) as pilot:
        await _open_chat(chat_app, pilot)
        await _say(chat_app, pilot, "one")
        await _say(chat_app, pilot, "/reset")

        assert _transcript(chat_app) == []
        await _say(chat_app, pilot, "two")
        # Turn 1 again: the client's history went with the transcript.
        assert _transcript(chat_app)[-1] == "[fake model] turn 1: 'two'"


async def test_q_is_a_letter_at_the_chat_prompt(chat_app: utrain.tui.app.UtrainApp) -> None:
    async with chat_app.run_test(size=SIZE) as pilot:
        await _open_chat(chat_app, pilot)
        await pilot.press("q")
        await pilot.pause()

        assert isinstance(chat_app.screen, utrain.tui.screens.ChatScreen)
        assert _chat(chat_app).field().value == "q"


async def test_leaving_the_screen_stops_the_container(
    chat_app: utrain.tui.app.UtrainApp,
) -> None:
    """The server is the screen's for exactly as long as the screen is open."""
    async with chat_app.run_test(size=SIZE) as pilot:
        await _open_chat(chat_app, pilot)
        source = chat_app.data
        assert isinstance(source, _ServingData)
        server = source.servers[0]
        assert server.proc.poll() is None

        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(chat_app.screen, utrain.tui.screens.MainScreen)

        # Shut down on a plain thread, so it is not finished the moment the
        # screen is gone.
        deadline = time.monotonic() + 15
        while server.proc.poll() is None and time.monotonic() < deadline:
            await pilot.pause(0.05)
        assert server.proc.poll() is not None


async def test_t_is_greyed_on_a_run_with_nothing_to_talk_to(
    chat_app: utrain.tui.app.UtrainApp,
) -> None:
    """The draft run has never been started, which is what `serve` refuses."""
    async with chat_app.run_test(size=SIZE) as pilot:
        await _settle(chat_app, pilot)
        assert _main(chat_app).selected_run == DRAFT_ID

        # Greyed rather than hidden: the key is the answer to "why can I not
        # talk to this one", and the reason is the run, not the pane.
        assert _footer(chat_app)["t"] is False
        assert "is still configuring" in (_main(chat_app).chattable() or "")
        await pilot.press("t")
        await pilot.pause()
        assert isinstance(chat_app.screen, utrain.tui.screens.MainScreen)


async def test_t_is_offered_from_the_phases_pane_too(
    chat_app: utrain.tui.app.UtrainApp,
) -> None:
    """Having just read a phase is when you want to hear what it trained."""
    async with chat_app.run_test(size=SIZE) as pilot:
        await _settle(chat_app, pilot)
        await pilot.press("down")  # onto the finished run
        await _settle(chat_app, pilot)
        await pilot.press("enter")
        await _settle(chat_app, pilot)
        await pilot.press("down")  # onto `pretrain`, the phase with a model
        await _settle(chat_app, pilot)

        assert _footer(chat_app)["t"] is True


async def test_t_from_the_phases_pane_talks_to_the_phase_under_the_cursor(
    chat_app: utrain.tui.app.UtrainApp,
) -> None:
    """Each phase's data dir is a snapshot, so each is its own thing to talk to.

    Standing on `pretrain` and pressing `t` asks for that phase by name rather
    than falling back to whichever one the run finished on.
    """
    async with chat_app.run_test(size=SIZE) as pilot:
        await _settle(chat_app, pilot)
        await pilot.press("down")
        await _settle(chat_app, pilot)
        await pilot.press("enter")
        await _settle(chat_app, pilot)
        await pilot.press("down")
        await _settle(chat_app, pilot)
        assert _main(chat_app).selected_phase == "pretrain"

        await pilot.press("t")
        await _settle(chat_app, pilot)
        for _ in range(4):
            await pilot.pause()

        assert _chat(chat_app).run_id == RUN_ID
        assert _recording(chat_app).served_phases == ["pretrain"]
        await _say(chat_app, pilot, "hello")
        assert _transcript(chat_app)[-1] == "[fake model] turn 1: 'hello'"


async def test_t_is_greyed_on_a_phase_that_leaves_no_model(
    chat_app: utrain.tui.app.UtrainApp,
) -> None:
    """`tokenizer` ran and succeeded, but the image does not serve it.

    The refusal names the phase: the run is fine and the key works one row
    down, so "does not support serve" would be answering a different question.
    """
    async with chat_app.run_test(size=SIZE) as pilot:
        await _settle(chat_app, pilot)
        await pilot.press("down")
        await _settle(chat_app, pilot)
        await pilot.press("enter")
        await pilot.press("up")
        await _settle(chat_app, pilot)
        assert _main(chat_app).selected_phase == "tokenizer"

        assert _footer(chat_app)["t"] is False
        assert "does not serve phase 'tokenizer'" in (_main(chat_app).chattable() or "")
        await pilot.press("t")
        await pilot.pause()
        assert isinstance(chat_app.screen, utrain.tui.screens.MainScreen)


async def test_t_is_offered_wherever_the_focus_is(
    chat_app: utrain.tui.app.UtrainApp,
) -> None:
    """What it talks to follows the level, not the focused pane, so standing in
    the metric picker is no reason for the key to disappear."""
    async with chat_app.run_test(size=SIZE) as pilot:
        await _settle(chat_app, pilot)
        await pilot.press("enter")
        await pilot.press("m")
        await _settle(chat_app, pilot)

        assert "t" in _footer(chat_app)


async def test_t_goes_live_as_soon_as_the_run_is_selected(
    app: utrain.tui.app.UtrainApp,
) -> None:
    """Without having to visit the phases pane first.

    Whether a run can be chatted with depends on its phases, which land a fetch
    after the cursor moves onto it -- so unless the footer is rebuilt for that
    too, `t` stays greyed until something else it is guarded on changes, and
    the only obvious way out is to go and look at the phases.
    """
    async with app.run_test(size=SIZE) as pilot:
        await _settle(app, pilot)
        await _settle(app, pilot)

        assert _main(app).chattable() is None
        assert _footer(app)["t"] is True


# -- adding an image ------------------------------------------------------


def _images_screen(app: utrain.tui.app.UtrainApp) -> utrain.tui.screens.ImagesScreen:
    screen = app.screen
    assert isinstance(screen, utrain.tui.screens.ImagesScreen)
    return screen


def _image_names(app: utrain.tui.app.UtrainApp) -> list[str]:
    table = _images_screen(app).table("#images")
    return [
        str(table.get_cell_at(textual.coordinate.Coordinate(row, 0)))
        for row in range(table.row_count)
    ]


async def _open_images(app: utrain.tui.app.UtrainApp, pilot: typing.Any) -> None:
    await _settle(app, pilot)
    await pilot.press("i")
    await _settle(app, pilot)


async def test_a_opens_the_add_image_dialog(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=SIZE) as pilot:
        await _open_images(app, pilot)
        assert _image_names(app) == CHOICE_IMAGES

        await pilot.press("a")
        await pilot.pause()
        assert isinstance(app.screen, utrain.tui.screens.AddImageScreen)


async def test_a_url_and_enter_adds_the_image(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=SIZE) as pilot:
        await _open_images(app, pilot)
        await pilot.press("a")
        await pilot.pause()
        await pilot.press(*"docker://example.org/shakespeare:v1")
        await pilot.press("enter")
        await _settle(app, pilot)

        source = _recording(app)
        assert source.added == ["docker://example.org/shakespeare:v1"]
        # The name is derived from the URL's last path component, as
        # `images.add_image` derives it, and the row is there without waiting
        # for the next tick.
        assert "utrain-shakespeare" in _image_names(app)
        assert _images_screen(app).error == "added utrain-shakespeare"


async def test_an_empty_url_is_refused_in_the_dialog(app: utrain.tui.app.UtrainApp) -> None:
    """Answered on the field rather than by podman failing a minute later."""
    async with app.run_test(size=SIZE) as pilot:
        await _open_images(app, pilot)
        await pilot.press("a")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, utrain.tui.screens.AddImageScreen)
        assert not _recording(app).added


async def test_escape_leaves_the_dialog_without_adding(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=SIZE) as pilot:
        await _open_images(app, pilot)
        await pilot.press("a")
        await pilot.pause()
        await pilot.press(*"docker://example.org/nope")
        await pilot.press("escape")
        await _settle(app, pilot)

        assert isinstance(app.screen, utrain.tui.screens.ImagesScreen)
        assert not _recording(app).added
        assert _image_names(app) == CHOICE_IMAGES


async def test_a_failed_pull_is_reported_and_adds_nothing(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=SIZE) as pilot:
        await _open_images(app, pilot)
        _recording(app).fail = "podman pull failed (exit 125): manifest unknown"
        await pilot.press("a")
        await pilot.pause()
        await pilot.press(*"docker://example.org/typo")
        await pilot.press("enter")
        await _settle(app, pilot)

        assert "manifest unknown" in _images_screen(app).error
        assert _image_names(app) == CHOICE_IMAGES


async def test_the_pull_message_outlives_the_refresh_that_would_clear_it(
    app: utrain.tui.app.UtrainApp,
) -> None:
    """The list refreshes once a second and `apply` clears the status line.

    A pull takes minutes, so "pulling..." cannot be a held message -- a hold
    expires. The screen stops clearing the line instead, for as long as the
    pull runs.
    """
    async with app.run_test(size=SIZE) as pilot:
        await _open_images(app, pilot)
        screen = _images_screen(app)
        screen.pull_started("docker://example.org/big")
        await pilot.pause()
        assert screen.error == "pulling docker://example.org/big..."
        # Greyed while one is running: `podman pull` is not re-entrant here.
        assert _footer(app)["a"] is False

        screen.apply([["utrain-fake", "10M", "0"]])
        assert screen.error == "pulling docker://example.org/big..."

        screen.pull_finished("utrain-big", "")
        await pilot.pause()
        assert _footer(app)["a"] is True


async def test_a_pull_outlives_the_panel_being_closed(app: utrain.tui.app.UtrainApp) -> None:
    """The app keeps the panel instance: a pull does not fit in the time the
    panel is open, so closing it cannot be what cancels it -- and the
    callbacks it reports on land on a screen whose widgets a close has taken
    away, which is what the `_closed` latch is for."""
    async with app.run_test(size=SIZE) as pilot:
        await _open_images(app, pilot)
        screen = _images_screen(app)
        screen.pull_started("docker://example.org/big")
        assert screen.pulling

        await pilot.press("escape")
        await _settle(app, pilot)
        assert isinstance(app.screen, utrain.tui.screens.MainScreen)

        # A report from the pull, landing on the closed panel: guarded, so
        # nothing is raised and nothing is drawn.
        screen.pull_finished("utrain-big", "")
        assert not screen.pulling

        await _open_images(app, pilot)
        assert app.screen is screen
        assert _image_names(app) == CHOICE_IMAGES


async def test_question_mark_opens_the_key_map(app: utrain.tui.app.UtrainApp) -> None:
    """The navigation keys are `show=False`, so this is where they are written."""
    async with app.run_test(size=SIZE) as pilot:
        await _settle(app, pilot)

        await pilot.press("question_mark")
        await _settle(app, pilot)

        assert isinstance(app.screen, utrain.tui.screens.HelpScreen)
        text = " ".join(str(s.render()) for s in app.screen.query(textual.widgets.Static))
        assert "tab / shift+tab" in text
        assert "2 / 3 / 4" in text

        await pilot.press("escape")
        await _settle(app, pilot)
        assert isinstance(app.screen, utrain.tui.screens.MainScreen)


async def test_the_key_map_is_reachable_from_every_destination(
    app: utrain.tui.app.UtrainApp,
) -> None:
    async with app.run_test(size=SIZE) as pilot:
        await _settle(app, pilot)
        await pilot.press("i")
        await _settle(app, pilot)

        await pilot.press("question_mark")
        await _settle(app, pilot)

        assert isinstance(app.screen, utrain.tui.screens.HelpScreen)


OTHER_ID = "f" * 32


def _seed_other(data_dir: pathlib.Path) -> None:
    run_dir = data_dir / "runs" / OTHER_ID
    (run_dir / "attempt" / "1" / "logs").mkdir(parents=True)
    with utrain.db.with_db(utrain.config.Settings(data_dir=data_dir)) as session:
        session.execute(
            sqlalchemy.insert(utrain.db.runs).values(
                id=OTHER_ID,
                name="other",
                image=IMAGE,
                image_id=IMAGE_ID,
                compute="cpu",
                status="done",
                config_hash="deadbeef",
                created_at=500.0,
            )
        )
        session.execute(
            sqlalchemy.insert(utrain.db.run_attempts).values(
                run_id=OTHER_ID,
                attempt=1,
                from_phase=None,
                status="done",
                pid=None,
                started_at=500.0,
                ended_at=600.0,
            )
        )
        for order, phase in enumerate(["alpha", "beta", "gamma", "delta"]):
            session.execute(
                sqlalchemy.insert(utrain.db.run_phases).values(
                    run_id=OTHER_ID,
                    attempt=1,
                    phase=phase,
                    phase_order=order,
                    status="done",
                    started_at=500.0,
                    ended_at=600.0,
                )
            )


@pytest.fixture()
def two_app(tmp_path: pathlib.Path) -> utrain.tui.app.UtrainApp:
    _seed(tmp_path)
    _seed_other(tmp_path)
    return _app(tmp_path)


async def test_the_phase_cursor_follows_the_selection_across_a_refill(
    live_app: utrain.tui.app.UtrainApp,
) -> None:
    """The rows are refilled in place, so a list that changes length would
    otherwise drift the cursor off the phase that is actually selected."""
    async with live_app.run_test(size=SIZE) as pilot:
        await _settle(live_app, pilot)
        await pilot.press("enter")
        await _settle(live_app, pilot)
        screen = _main(live_app)
        table = screen.table("#phases")
        assert screen.selected_phase == "pretrain"
        assert table.cursor_row == 1

        # Out onto the run with no phases at all, which empties the table, and
        # back: the cursor has to find its way to row 1 again.
        await pilot.press("escape")
        await _settle(live_app, pilot)
        await pilot.press("down")
        await _settle(live_app, pilot)
        assert screen.selected_run == DRAFT_ID
        await pilot.press("up")
        await _settle(live_app, pilot)

        assert screen.selected_phase == "pretrain"
        assert table.cursor_row == 1


async def test_a_run_only_passed_over_is_not_pinned_to_a_phase(
    live_app: utrain.tui.app.UtrainApp,
) -> None:
    """Standing on a run selects the phase it has got to, so that the panes
    show something -- but that is the app's choice, not a visit.

    Recording it would pin a live run to whichever phase it happened to be on
    when the cursor first crossed it, and the dashboard would stop following.
    """
    async with live_app.run_test(size=SIZE) as pilot:
        await _settle(live_app, pilot)
        screen = _main(live_app)
        assert screen.selected_run == LIVE_ID
        assert screen.selected_phase == "pretrain"

        assert screen.visited == {}


async def test_drilling_in_records_the_phase_it_opened_on(
    live_app: utrain.tui.app.UtrainApp,
) -> None:
    """A viewer who drills in and comes straight back out was on that phase."""
    async with live_app.run_test(size=SIZE) as pilot:
        await _settle(live_app, pilot)
        screen = _main(live_app)

        await pilot.press("enter")
        await _settle(live_app, pilot)

        assert screen.visited == {LIVE_ID: "pretrain"}


async def test_each_phase_comes_back_to_the_tab_it_was_left_on(
    live_app: utrain.tui.app.UtrainApp,
) -> None:
    """Phases are not alike -- a download has no curve and a train phase is
    mostly curve -- so how you were looking at one is worth keeping."""
    async with live_app.run_test(size=SIZE) as pilot:
        await _settle(live_app, pilot)
        await pilot.press("enter")
        await _settle(live_app, pilot)
        screen = _main(live_app)
        assert screen.selected_phase == "pretrain"

        # Plots on `pretrain`, logs on `tokenizer`.
        await pilot.press("3")
        await pilot.press("1")
        await _settle(live_app, pilot)
        assert screen.tab == "plots"

        await pilot.press("up")
        await _settle(live_app, pilot)
        assert screen.selected_phase == "tokenizer"
        await pilot.press("4")
        await pilot.press("1")
        await _settle(live_app, pilot)
        assert screen.tab == "log"

        await pilot.press("down")
        await _settle(live_app, pilot)
        assert screen.selected_phase == "pretrain"
        assert screen.tab == "plots"

        await pilot.press("up")
        await _settle(live_app, pilot)
        assert screen.tab == "log"


async def test_a_phase_with_no_tab_of_its_own_keeps_the_one_that_is_up(
    live_app: utrain.tui.app.UtrainApp,
) -> None:
    """Only phases you deliberately set stick; the rest do not move the pane
    under a cursor that is passing through."""
    async with live_app.run_test(size=SIZE) as pilot:
        await _settle(live_app, pilot)
        await pilot.press("enter")
        await pilot.press("4")
        await pilot.press("1")
        await _settle(live_app, pilot)
        screen = _main(live_app)
        assert screen.selected_phase == "pretrain"
        assert screen.tab == "log"

        await pilot.press("up")
        await _settle(live_app, pilot)

        assert screen.selected_phase == "tokenizer"
        assert screen.tab == "log"


async def test_the_tab_a_phase_remembers_survives_leaving_the_run(
    live_app: utrain.tui.app.UtrainApp,
) -> None:
    async with live_app.run_test(size=SIZE) as pilot:
        await _settle(live_app, pilot)
        await pilot.press("enter")
        await pilot.press("3")
        await pilot.press("1")
        await _settle(live_app, pilot)
        screen = _main(live_app)
        assert screen.tab == "plots"

        # Out to the runs, onto another run, and back in.
        await pilot.press("escape")
        await _settle(live_app, pilot)
        await pilot.press("down")
        await _settle(live_app, pilot)
        await pilot.press("2")
        await pilot.press("1")
        await _settle(live_app, pilot)
        assert screen.tab == "config"

        await pilot.press("up")
        await _settle(live_app, pilot)

        assert screen.selected_run == LIVE_ID
        assert screen.tab == "plots"


async def test_the_charset_setting_reaches_the_plots(tmp_path: pathlib.Path) -> None:
    """`tui_charset` is the pin for a viewer whose font braille is wrong for.

    Whether the font has the glyphs is the one thing a terminal program cannot
    ask -- see `render.default_charset` -- so the answer has to be settable.
    """
    _seed(tmp_path)
    app = _app(tmp_path, charset="braille")
    async with app.run_test(size=SIZE) as pilot:
        await _select_pretrain(app, pilot)
        await pilot.press("3")
        await _settle(app, pilot)
        screen = _main(app)

        assert screen.charset == utrain.tui.render.CHARSET_BRAILLE
        assert screen.plots
        assert all(p.charset == utrain.tui.render.CHARSET_BRAILLE for p in screen.plots.values())
