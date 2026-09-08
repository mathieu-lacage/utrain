"""The TUI's screens: one main screen, plus images and compute.

The layout is lazygit's. A narrow left column stacks the three lists that say
what you are looking at -- runs, then that run's phases, then that phase's
metrics -- and a wide right column shows the content: the run's config while the
runs pane has focus and the run can still be configured, and otherwise the
metric plots above the phase's log.

Navigation is selection, not a screen stack. Everything is on screen at once, so
one fetch fills all of it: `data.Data.snapshot` is a single worker producing a
single `Snapshot`, applied in one pass. Five panes refreshing on five schedules
would show five slightly different moments.

Every fetch runs in a thread worker. `podman describe` starts a container and
`compute.collect_compute` shells out to nvidia-smi; either on the message loop
would freeze the app for seconds.

Screens reach the app through :class:`Host` rather than through `self.app`.
Textual types `self.app` as `App[Unknown]`, so every use of it costs a cast; a
small protocol is both better typed and a clearer statement of what a screen is
allowed to ask of its app.
"""

import collections.abc
import dataclasses
import pathlib
import threading
import time
import typing

import textual.app
import textual.binding
import textual.containers
import textual.coordinate
import textual.screen
import textual.timer
import textual.widget
import textual.widgets

from .. import chat as chatmod
from .. import exceptions, metrics, reconcile, serve, types
from . import data, export, render, widgets

# The screen's actions that a single key reaches, and which are therefore
# switched off while an editor has the focus (`q` is the app's, and
# `UtrainApp.check_action` switches that one off). `back` is not here: it is
# how you get out again.
_EDIT_MODE_OFF = frozenset(
    {
        "focus_pane",
        "next_pane",
        "drill_in",
        "images",
        "compute",
        "force_refresh",
        # The row cursor belongs to the list, not to an open editor: `down`
        # while typing would leave the editor behind and move on.
        "next_row",
        "prev_row",
        # An `S` typed at a bool field is a value, not "stop the run"; the
        # same goes for the `n`, `d` and `R` of the rest of the lifecycle.
        "start_run",
        "stop_run",
        "restart_run",
        "delete_run",
        "new_run",
        # A `t` typed at a config field is a letter, not "chat with this run".
        "chat_run",
    }
)

# What `MainScreen.lifecycle` was asked to do. A literal rather than an enum:
# there are four of them, they are never stored, and the call sites read as
# prose.
_Op = typing.Literal["start", "stop", "restart", "delete"]

# The keys that change a run's existence or its state, as opposed to what is
# being looked at. They share a gating rule in `MainScreen.check_action`.
_LIFECYCLE = frozenset({"start_run", "stop_run", "restart_run", "delete_run", "new_run"})

# What a chat turn is sent with. Fixed rather than offered as fields: the two
# numbers are the CLI's defaults, and a screen that asked about them before
# every conversation would be asking a question most viewers do not have an
# answer to.
_CHAT_MAX_TOKENS = 200
_CHAT_TEMPERATURE = 0.8

# How often a streaming reply is handed to the widget. One `call_from_thread`
# per token would put a message on the loop for every few characters; at this
# interval a reply still arrives visibly word by word, for a fraction of the
# traffic.
_CHAT_FLUSH_SECONDS = 0.05

# Refresh cadence. Fast enough that a training curve visibly grows, slow enough
# that the query layer's reconcile-and-read costs nothing noticeable.
_REFRESH_SECONDS = 1.0

# How long the cursor must sit still before the panes are fetched for it. Short
# enough not to feel like lag on a deliberate move, long enough that scanning a
# list costs one fetch rather than one per row -- and a fetch reconciles every
# run against its directory, which is a write.
_SETTLE_SECONDS = 0.15

# How long a message that answers a keypress stays up. Every refresh clears the
# error line, which is what makes a fetch error heal itself when the next fetch
# works -- but it would also blink "config saved" out of existence in under a
# second.
_MESSAGE_SECONDS = 3.0

# What the status line wears when the message on it is not an error. Textual
# reserves a leading dash for classes a widget sets on itself, which is exactly
# what this is; `UtrainApp.CSS` gives it a colour.
_OK_CLASS = "-ok"

# How many phases' metrics are kept open at once. Moving the cursor down the
# phase list opens a `metrics.Tail` per phase, and each holds a file and the
# row schemas it has accumulated; a browse should not leave all of them behind.
_MAX_TAILS = 8


# The answer a modal comes back with. Spelled as a TypeVar rather than with
# PEP 695's `def ask[T]`, which needs 3.12; this package supports 3.11.
Answer = typing.TypeVar("Answer")


class Host(typing.Protocol):
    """What a screen needs from the app hosting it."""

    def open(self, screen: textual.screen.Screen[None]) -> None:
        """Push a screen onto the stack."""

    def ask(
        self,
        screen: textual.screen.ModalScreen[Answer],
        answer: collections.abc.Callable[[Answer | None], None],
    ) -> None:
        """Push a modal question and hand its answer back.

        Separate from `open` because that one is typed for a screen with no
        result, and a question has one. `None` is a screen dismissed without an
        answer, which callers read as "no" -- and which `NewRunScreen`, whose
        own answer type already includes `None`, dismisses with directly.

        Generic in the answer: `ConfirmScreen` answers with a bool and
        `NewRunScreen` with a `NewRun`, and a `bool`-only signature would make
        the second a cast at every call site.
        """

    def from_thread(
        self,
        callback: collections.abc.Callable[..., None],
        *args: object,
    ) -> None:
        """Run ``callback`` on the message loop, from a worker thread."""


class _Screen(textual.screen.Screen[None]):
    """Shared chrome: a header, a footer, an error line and a refresh timer."""

    BINDINGS = [
        textual.binding.Binding("r", "force_refresh", "refresh"),
        textual.binding.Binding("escape", "app.pop_screen", "back", show=False),
    ]

    def __init__(self, host: Host, source: data.Data) -> None:
        super().__init__()
        self.host = host
        self.data = source
        # Kept as well as displayed: a `Static`'s content is not readable back
        # off the widget, and both the tests and `showing_config` want to know
        # what the viewer was last told.
        self.error = ""
        self._held_until = 0.0

    def compose(self) -> textual.app.ComposeResult:
        yield textual.widgets.Header()
        yield from self.compose_body()
        yield textual.widgets.Static("", id="error")
        yield textual.widgets.Footer()

    def compose_body(self) -> textual.app.ComposeResult:
        return iter(())

    def on_mount(self) -> None:
        # The timer is in place before the first fetch, because `refresh_data`
        # restarts it.
        self.timer = self.set_interval(_REFRESH_SECONDS, self.refresh_data)
        self.refresh_data()

    def action_force_refresh(self) -> None:
        """Drop the podman caches too, so a rebuilt image is picked up."""
        self.data.refresh()
        self.refresh_data()

    def refresh_data(self) -> None:
        raise NotImplementedError

    def table(self, identifier: str) -> textual.widgets.DataTable[render.Cell]:
        """A DataTable by id.

        The type parameter goes on the annotation rather than on the argument:
        `query_one` isinstance-checks whatever class it is handed, and a
        subscripted generic cannot be used in an isinstance check.
        """
        found: textual.widgets.DataTable[render.Cell] = self.query_one(
            identifier, textual.widgets.DataTable
        )
        return found

    def show_error(self, message: str, *, hold: bool = False) -> None:
        """Put a message on the error line.

        `hold` is for a message that answers a keypress rather than a fetch:
        it is held for `_MESSAGE_SECONDS` against the refresh that would
        otherwise clear it before it had been read.
        """
        self._status(message, hold=hold, ok=False)

    def show_message(self, message: str, *, hold: bool = False) -> None:
        """The same line, for something that went right.

        The line itself is shared -- there is one row of chrome under the panes
        and two would be a row of blank most of the time -- but "wrote
        loss.png" is not an error, so it is not drawn in the colour errors are.
        """
        self._status(message, hold=hold, ok=True)

    def _status(self, message: str, *, hold: bool, ok: bool) -> None:
        self.error = message
        self._held_until = time.monotonic() + _MESSAGE_SECONDS if hold else 0.0
        line = self.query_one("#error", textual.widgets.Static)
        line.set_class(ok, _OK_CLASS)
        line.update(message)

    def held_error(self, message: str) -> None:
        """`show_error(..., hold=True)`, reachable from a worker.

        `Host.from_thread` passes arguments positionally and `hold` is
        keyword-only, which is right at the call sites that read as prose.
        """
        self.show_error(message, hold=True)

    def held_message(self, message: str) -> None:
        """`show_message(..., hold=True)`, reachable from a worker."""
        self.show_message(message, hold=True)

    def holding(self) -> bool:
        """A held message is still on screen."""
        return time.monotonic() < self._held_until

    def clear_error(self) -> None:
        if self.holding():
            return
        self.error = ""
        self._held_until = 0.0
        line = self.query_one("#error", textual.widgets.Static)
        line.remove_class(_OK_CLASS)
        line.update("")


def _table(
    identifier: str,
    title: str,
    columns: tuple[str, ...],
) -> textual.widgets.DataTable[render.Cell]:
    """One of the sidebar's lists, framed and titled like a lazygit pane."""
    table: textual.widgets.DataTable[render.Cell] = widgets.PaneTable(id=identifier)
    table.cursor_type = "row"
    table.border_title = title
    table.add_columns(*columns)
    return table


def _fill(table: textual.widgets.DataTable[render.Cell], rows: list[list[render.Cell]]) -> None:
    """Bring a table's rows up to date in place.

    Deliberately not clear-and-refill. The cursor *is* the selection here, and
    `clear` puts it back on row 0, which the screen would read as the viewer
    having selected the first phase -- once a second, and again a moment later
    when the cursor was restored. Updating the cells that changed leaves the
    cursor untouched, so no selection event is invented.
    """
    for index, cells in enumerate(rows):
        if index >= table.row_count:
            table.add_row(*cells)
            continue
        for column, cell in enumerate(cells):
            coordinate = textual.coordinate.Coordinate(index, column)
            if table.get_cell_at(coordinate) != cell:
                table.update_cell_at(coordinate, cell, update_width=True)
    while table.row_count > len(rows):
        table.remove_row(table.ordered_rows[-1].key)


def min_prefix_len(run_ids: list[str]) -> int:
    """Shortest prefix that still tells these run ids apart."""
    if not run_ids:
        return 1
    for prefix_len in range(1, 33):
        if len({rid[:prefix_len] for rid in run_ids}) == len(run_ids):
            return prefix_len
    return 32


@dataclasses.dataclass(frozen=True)
class NewRun:
    """What the new-run dialog came back with: `runs.create_run`'s three arguments."""

    name: str
    image: str
    compute: str


@dataclasses.dataclass(frozen=True)
class Export:
    """What the export dialog came back with: where to write, and in what."""

    path: pathlib.Path
    fmt: str


class MainScreen(_Screen):
    """Runs, phases and metrics on the left; plots, logs or config on the right.

    The metrics reader for each phase is held open: following a live phase means
    resuming from where the last read stopped, and that needs the reader itself,
    not just an offset (see `utrain.metrics`). Because the selected phase now
    changes with a cursor move rather than with a screen push, those readers are
    kept in a small cache and closed as they fall out of it.
    """

    TITLE = "utrain"

    BINDINGS = [
        # The pane keys are `show=False`: each pane's number is in its title,
        # and a footer that repeated all five would leave no room for the keys
        # that are actually about the focused pane.
        textual.binding.Binding("tab", "next_pane", "pane", show=False),
        textual.binding.Binding("1", "focus_pane('runs')", "runs", show=False),
        textual.binding.Binding("2", "focus_pane('phases')", "phases", show=False),
        textual.binding.Binding("3", "focus_pane('metrics')", "metrics", show=False),
        textual.binding.Binding("4", "focus_pane('top')", "content", show=False),
        textual.binding.Binding("5", "focus_pane('log')", "logs", show=False),
        # On the screen rather than on the form: you pick the run in the runs
        # pane, so that is where you want to be able to say "edit this one".
        # `check_action` greys it out unless the selected run can be edited.
        textual.binding.Binding("e", "edit_config", "edit"),
        # Lifecycle, next to `e` and for the same reason: you pick the run in
        # the runs pane, so that is where you say "start this one".
        # `check_action` hides them outside that pane and greys them out on a
        # run whose status does not allow them.
        textual.binding.Binding("s", "start_run", "start"),
        textual.binding.Binding("S", "stop_run", "stop"),
        textual.binding.Binding("n", "new_run", "new"),
        textual.binding.Binding("d", "delete_run", "delete"),
        # `r` is the base screen's refresh, so restart takes the shifted key --
        # the same convention `S` follows.
        textual.binding.Binding("R", "restart_run", "restart"),
        # Overrides `_Screen`'s escape, which pops the screen. There is nothing
        # under this one but the app's blank default screen, so popping it is
        # never what was meant; backing out of a pane is.
        textual.binding.Binding("escape", "back", "back", show=False),
        textual.binding.Binding("i", "images", "images"),
        textual.binding.Binding("c", "compute", "compute"),
        # `t` for talk: `c` is compute, and a `C` next to it would read as a
        # variant of it. Offered from the runs and phases panes both, the way
        # `S` and `R` are; `check_action` greys it out on a run there is
        # nothing to talk to yet.
        textual.binding.Binding("t", "chat_run", "chat"),
    ]

    # The panes, by column. `tab` cycles within one of these; the number keys
    # are how you cross from one to the other. The right column is one or the
    # other list, never both: the config and the plots are the same slot.
    _LEFT = ("runs", "phases", "metrics")
    _RIGHT_CONFIG = ("config",)
    _RIGHT_DASHBOARD = ("plots", "log")
    # Focus in either of these means the content column is showing config: the
    # runs pane is what it follows, and the config is where you land from it.
    _CONFIG_PANES = ("runs", "config")

    def __init__(self, host: Host, source: data.Data) -> None:
        super().__init__(host, source)
        self.runs: list[types.RunRow] = []
        self.phases: list[types.PhaseListEntry] = []
        # Which sidebar pane the content column is following. Not simply
        # whatever has focus: clicking into the log or into the config form
        # must not change what the content column is about.
        self.pane = "runs"
        self.selected_run: str | None = None
        self.selected_phase: str | None = None
        # A run created this session, waiting for the fetch that will list it.
        # `apply_runs` puts the cursor on it and clears this.
        self._pending_run: str | None = None
        self.x_axis = render.X_STEP
        self.solo: str | None = None
        # The plots the image asks for on the selected phase, and whether the
        # viewer has taken the dashboard back off it. Both per-phase, and both
        # reset by `reset_phase_view`.
        self.pinned: list[render.PlotKey] = []
        self.overridden = False
        self.log_y = False
        self.charset = render.CHARSET_BLOCK
        # Both keyed by phase address, so returning to a phase shows the series
        # already read rather than starting over from an exhausted Tail.
        self.points: dict[str, dict[str, list[metrics.MetricPoint]]] = {}
        self.columns: dict[str, list[str]] = {}
        self._tails: dict[str, metrics.Tail] = {}
        self.plots: dict[render.PlotKey, widgets.MetricPlot] = {}
        # What the footer was last built for; see `sync_bindings`.
        self._binding_state: tuple[str, str | None, bool, bool, str | None] | None = None
        # The pending fetch for a cursor that is still moving; see `refresh_soon`.
        self._settle: textual.timer.Timer | None = None
        # Whether a fetch is running, and whether another was asked for while
        # it was; see `refresh_data`.
        self._fetching = False
        self._again = False

    # -- layout -----------------------------------------------------------

    def compose_body(self) -> textual.app.ComposeResult:
        with textual.containers.Horizontal(id="main"):
            with textual.containers.Vertical(id="sidebar"):
                yield _table("runs", "1 runs", render.RUN_COLUMNS)
                yield _table("phases", "2 phases", render.PHASE_COLUMNS)
                yield widgets.MetricList(id="metrics")
            with textual.containers.Vertical(id="content"):
                yield widgets.ConfigPane(id="config")
                with widgets.PlotPane(id="plots"):
                    yield textual.widgets.Static("no metrics yet", id="empty")
                yield widgets.LogTail(id="log")

    def on_mount(self) -> None:
        super().on_mount()
        self.table("#runs").focus()

    def on_unmount(self) -> None:
        if self._settle is not None:
            self._settle.stop()
        for tail in self._tails.values():
            tail.close()

    # -- fetching ---------------------------------------------------------

    def selected_entry(self) -> types.PhaseListEntry | None:
        """The listed phase the cursor is on."""
        for entry in self.phases:
            if entry.phase == self.selected_phase:
                return entry
        return None

    def attempt(self) -> int | None:
        """The attempt the selected phase's output is under, if not the latest.

        A `--from-phase` restart does not re-run the phases before it, so their
        log and their metrics are in the attempt that did run them.
        """
        entry = self.selected_entry()
        return None if entry is None else entry.inherited_from

    def address(self) -> str | None:
        """The selected phase's address, as the query layer names it.

        Three-part for an inherited phase, since the two-part form means the
        latest attempt.
        """
        if self.selected_run is None or self.selected_phase is None:
            return None
        attempt = self.attempt()
        if attempt is None:
            return f"{self.selected_run}/{self.selected_phase}"
        return f"{self.selected_run}/{attempt}/{self.selected_phase}"

    def refresh_data(self) -> None:
        if self._fetching:
            # One fetch at a time: a Tail is a single open reader with a cursor
            # in it, and two threads reading it interleave. `fetch_done` runs
            # this refresh as soon as the reader is free, and any number asked
            # for in the meantime coalesce into that one.
            self._again = True
            return
        # A fetch started by a cursor move restarts the clock, so a selection
        # is never followed a few milliseconds later by a tick that cancels the
        # fetch it just started.
        self.timer.reset()
        addr = self.address()
        # The Tail is read on the loop and handed to the worker rather than
        # looked up there: a thread worker cannot be interrupted, so `exclusive`
        # only stops a new one starting, and two overlapping fetches must not
        # both decide to open a reader for the same phase.
        self._fetching = True
        self.fetch(
            self.selected_run, self.selected_phase, self.attempt(), self._tails.get(addr or "")
        )

    def refresh_soon(self) -> None:
        """Fetch once the cursor has stopped moving.

        Not `refresh_data`, which starts the thread straight away: a thread
        worker cannot be interrupted, so `exclusive` stops the next fetch
        starting but does not take back the work of the one already running.
        Scanning a list would run a full fetch per row and throw all but the
        last away.
        """
        if self._settle is not None:
            self._settle.stop()
        # The tick goes out with it, so a scan is not interrupted halfway by a
        # refresh for whichever row the cursor happened to be passing.
        self.timer.reset()
        self._settle = self.set_timer(_SETTLE_SECONDS, self.refresh_data)

    @textual.work(thread=True, exclusive=True, group="snapshot")
    def fetch(
        self,
        run_id: str | None,
        phase: str | None,
        attempt: int | None,
        tail: metrics.Tail | None,
    ) -> None:
        # `_fetching` is cleared in a `finally`, so that an error the query
        # layer did not wrap costs one tick rather than every tick after it.
        try:
            try:
                snapshot = self.data.snapshot(run_id, phase, attempt, tail)
            except exceptions.UI as e:
                self.host.from_thread(self.show_error, str(e))
                return
            self.host.from_thread(self.apply, snapshot)
        finally:
            self.host.from_thread(self.fetch_done)

    def fetch_done(self) -> None:
        self._fetching = False
        if self._again:
            self._again = False
            self.refresh_data()

    def apply(self, snapshot: data.Snapshot) -> None:
        self.clear_error()
        self.apply_runs(snapshot)
        # Everything below is about one run, and the cursor can have moved off
        # it since the fetch started: a thread worker cannot be interrupted, so
        # `exclusive` stops the next fetch starting, not this one landing.
        # Scanning the runs list would otherwise repaint these panes once per
        # run crossed. Dropping the snapshot leaves the previous run's rows up
        # until the cursor settles, which is one change rather than a strobe.
        #
        # `apply_runs` is outside it because the runs list is not about any one
        # run, and `apply_focus`/`sync_bindings` because they read `self.pane`
        # and `self.runs`, which it has just brought up to date.
        if snapshot.run_id == self.selected_run:
            self.apply_phases(snapshot)
            self.apply_metrics(snapshot)
            # The log is per phase, so it takes the finer of the two checks --
            # the one `apply_metrics` makes for itself.
            if snapshot.address == self.address():
                self.query_one("#log", widgets.LogTail).show(snapshot.log)
            self.apply_config(snapshot)
        else:
            self.discard_tail(snapshot.tail)
        self.apply_focus()
        self.sync_bindings()

    def apply_runs(self, snapshot: data.Snapshot) -> None:
        self.runs = snapshot.runs
        # The id column is truncated to the shortest prefix that separates the
        # runs on screen, as `cli/render` does. Addresses always use the full
        # id: the short form depends on what happens to be displayed.
        width = min_prefix_len([r.id for r in snapshot.runs])
        table = self.table("#runs")
        _fill(table, [render.run_cells(r, width) for r in snapshot.runs])
        # A run just created is selected as soon as it is listed, so its config
        # -- which is the only thing there is to do with it next -- comes up
        # without the viewer having to find it. Moving the cursor is what
        # selects: `RowHighlighted` routes back through `select_run`.
        if self._pending_run is not None:
            for index, run in enumerate(snapshot.runs):
                if run.id == self._pending_run:
                    self._pending_run = None
                    table.move_cursor(row=index)
                    # And say so, rather than leaving it to `RowHighlighted`: a
                    # new run sorts to the top, so the cursor is already on
                    # row 0 and moving it there is a no-op that raises no event
                    # -- the row under it changed identity, not position.
                    # `select_run` returns early if the event does arrive too.
                    self.select_run(run.id)
                    return
        if self.selected_run is None and snapshot.runs:
            # Whatever the cursor is on, not row 0: this branch is both the
            # first fetch, where the cursor is at 0 anyway, and the fetch after
            # a delete, where the rows below the deleted one have shifted up
            # under a stationary cursor. Clamped, because the row it was on may
            # have been the last one.
            row = min(max(table.cursor_row, 0), len(snapshot.runs) - 1)
            table.move_cursor(row=row)
            self.select_run(snapshot.runs[row].id)

    def apply_phases(self, snapshot: data.Snapshot) -> None:
        self.phases = snapshot.phases
        _fill(
            self.table("#phases"),
            [render.phase_cells(p, snapshot.now) for p in snapshot.phases],
        )
        if self.selected_phase is None and snapshot.phases:
            self.select_phase(snapshot.phases[0].phase)

    def apply_metrics(self, snapshot: data.Snapshot) -> None:
        addr = self.address()
        metric_list = self.query_one("#metrics", widgets.MetricList)
        if addr is None:
            metric_list.show([])
            self.discard_tail(snapshot.tail)
            return
        if snapshot.address != addr:
            # The cursor moved while the fetch was in flight; these points
            # belong to the phase that was selected then, not to this one.
            self.discard_tail(snapshot.tail)
            return

        self.adopt_tail(addr, snapshot.tail)
        series = self.points.setdefault(addr, {})
        for name, points in snapshot.update.points.items():
            series.setdefault(name, []).extend(points)
        seen = self.columns.setdefault(addr, [])
        seen.extend(c for c in snapshot.update.columns if c not in seen)

        pinned = [] if self.overridden else [(spec.y, spec.x) for spec in snapshot.phase_plots]
        changed = pinned != self.pinned
        self.pinned = pinned
        if metric_list.sync(seen, self.default_checked()) or snapshot.update.points or changed:
            self.update_plots()
        metric_list.show(
            render.metric_rows(
                metric_list.columns,
                {name: points[-1].value for name, points in series.items() if points},
                self.plotted(),
                self.solo,
            )
        )

    def apply_config(self, snapshot: data.Snapshot) -> None:
        form = self.query_one("#config", widgets.ConfigPane)
        if snapshot.run is None or snapshot.config_schema is None:
            return
        form.show(
            snapshot.run.run.id,
            render.run_summary(snapshot.run),
            render.config_rows(snapshot.config_schema, snapshot.config, snapshot.phase_order),
            editable=snapshot.run.run.status == "configuring",
        )
        # Each pane names what it is showing, since which of the two is up
        # depends on where the focus is. Both are numbered 4: they are the same
        # slot in the content column, and the number is only in the title now
        # that the footer no longer carries it.
        form.border_title = f"4 {render.run_title(snapshot.run)}"
        self.query_one(
            "#plots", widgets.PlotPane
        ).border_title = f"4 {render.phase_title(snapshot.phase_label, snapshot.run)}"

    # -- what the content column shows ------------------------------------

    def selected_run_row(self) -> types.RunRow | None:
        for run in self.runs:
            if run.id == self.selected_run:
                return run
        return None

    def run_is(self, status: str) -> bool:
        """The selected run is in `status`. What every lifecycle key turns on."""
        run = self.selected_run_row()
        return run is not None and run.status == status

    def showing_config(self) -> bool:
        """The content column shows config while the runs pane is being followed
        -- or while the viewer is standing in the config itself.

        Every run, not only one that can still be edited: what a finished run
        was configured with is the first thing you want when comparing it to
        another. A run past `configuring` gets the same fields read-only.
        """
        return self.pane in self._CONFIG_PANES and self.selected_run_row() is not None

    def right_column(self) -> tuple[str, ...]:
        """The content column's panes: the config, or the plots and the log."""
        return self._RIGHT_CONFIG if self.showing_config() else self._RIGHT_DASHBOARD

    def column(self, pane: str) -> tuple[str, ...]:
        """The panes `tab` can reach from `pane`, in order."""
        return self._LEFT if pane in self._LEFT else self.right_column()

    def apply_focus(self) -> None:
        config = self.showing_config()
        self.query_one("#config", widgets.ConfigPane).display = config
        self.query_one("#plots", widgets.PlotPane).display = not config
        self.query_one("#log", widgets.LogTail).display = not config

    def _pane_of(self, node: textual.widget.Widget | None) -> str | None:
        """The pane `node` sits in, or None for anything else."""
        panes = self._LEFT + self._RIGHT_CONFIG + self._RIGHT_DASHBOARD
        for parent in widgets.ancestors(node):
            if parent.id in panes:
                return parent.id
        return None

    def editing(self) -> bool:
        """An editor is open on a config row, so the keys belong to it."""
        return widgets.in_config_field(self.focused)

    def sync_bindings(self) -> None:
        """Rebuild the footer when what the keys can do has changed.

        `check_action` is re-read whenever anyone asks, but the footer is a
        widget: it is composed once and recomposed only when the screen
        publishes that its bindings changed. Textual publishes that on a focus
        change and on nothing else -- so a run finishing under a stationary
        cursor, or a cursor moving from a configuring run to a finished one,
        would leave the footer still offering `e`, `s` or `S`.

        Guarded on the four things those answers depend on, because the
        alternative is recomposing the footer once a second for the life of the
        app. Editing is not among them: opening and closing an editor is a
        focus change, which Textual already notices.

        Whether there is a curve to export is the third, and unlike the others
        it changes on a keypress rather than on a fetch -- unchecking the last
        metric must grey `E` there and then -- which is why `redraw_metrics`
        calls this as well.

        Whether there is a model to talk to is the fourth, and it is the one
        that is not answered by the run row: `chattable` reads the run's phases,
        and those land a fetch *after* the cursor reaches the run, one apply
        later than the row itself. Without it in the guard, `t` is greyed on the
        frame the run is selected and nothing rebuilds the footer afterwards --
        so it comes back only when some unrelated part of the state moves, which
        from the viewer's side looks like having to go and visit the phases pane
        before the key will work.
        """
        run = self.selected_run_row()
        state = (
            self.pane,
            None if run is None else run.status,
            self.export_metric() is not None,
            self.chattable() is None,
            # `t` greys per phase now, so moving down the phases pane has to
            # rebuild the footer even though the pane and run have not changed.
            self.chat_phase(),
        )
        if state != self._binding_state:
            self._binding_state = state
            self.refresh_bindings()

    # -- selection --------------------------------------------------------

    def select_run(self, run_id: str) -> None:
        if run_id == self.selected_run:
            return
        self.selected_run = run_id
        self.selected_phase = None
        self.phases = []
        self.reset_phase_view()
        # Before the fetch, not after it: `self.runs` already holds the row, so
        # the footer can answer for the new run this frame rather than a
        # refresh later.
        self.sync_bindings()
        self.refresh_soon()

    def select_phase(self, phase: str) -> None:
        if phase == self.selected_phase:
            return
        self.selected_phase = phase
        self.reset_phase_view()
        # As soon as the cursor stops rather than on the next tick: a move that
        # took a second to reach the plots would feel like the app had missed
        # it, and one that fetched per row on the way would strobe.
        self.refresh_soon()

    def reset_phase_view(self) -> None:
        """Start the metrics panes over for a different phase.

        The x axis goes back to step because a metric axis chosen for one phase
        need not exist in the next, and the solo goes with it. So does the
        override: which plots a phase names is the phase's answer, and leaving
        one phase's plots says nothing about the next phase's.
        """
        self.x_axis = render.X_STEP
        self.solo = None
        self.overridden = False
        # Read off the describe cache rather than waited for from the next
        # snapshot: the metrics already read are re-adopted below, and a phase
        # whose plots arrived a tick later would check every one of them first
        # and still have them checked once the viewer left its plots.
        run = self.selected_run_row()
        self.pinned = [
            (spec.y, spec.x)
            for spec in data.phase_plots(
                None if run is None else self.data.described(run.image), self.selected_phase
            )
        ]
        self.query_one("#metrics", widgets.MetricList).reset()
        for plot in self.plots.values():
            plot.remove()
        self.plots = {}
        addr = self.address()
        if addr is not None:
            # Re-adopt what an earlier visit read: the Tail has already handed
            # those column names over and will not repeat them.
            metric_list = self.query_one("#metrics", widgets.MetricList)
            if metric_list.sync(self.columns.get(addr, []), self.default_checked()):
                self.update_plots()

    # Two handlers with selectors rather than one that asks the event which
    # table it came from: moving the cursor means something different in each
    # list, and `@textual.on` says so without an `if`.
    @textual.on(textual.widgets.DataTable.RowHighlighted, "#runs")
    def _run_highlighted(self, event: textual.widgets.DataTable.RowHighlighted) -> None:
        # Moving the cursor is the selection: there is nothing to drill into any
        # more, so waiting for enter would only make the panes lag the cursor.
        if 0 <= event.cursor_row < len(self.runs):
            self.select_run(self.runs[event.cursor_row].id)

    @textual.on(textual.widgets.DataTable.RowHighlighted, "#phases")
    def _phase_highlighted(self, event: textual.widgets.DataTable.RowHighlighted) -> None:
        if 0 <= event.cursor_row < len(self.phases):
            self.select_phase(self.phases[event.cursor_row].phase)

    def on_descendant_focus(self) -> None:
        pane = self._pane_of(self.focused)
        if pane is not None:
            self.pane = pane
        self.apply_focus()
        # Textual refreshes bindings on a focus change already, but it does so
        # before this runs -- and `self.pane`, which decides whether the
        # lifecycle keys are offered at all, is only settled here.
        self.sync_bindings()

    # -- metrics ----------------------------------------------------------

    def discard_tail(self, tail: metrics.Tail | None) -> None:
        """Close a reader the fetch opened that no pane is going to adopt.

        A snapshot for a phase the cursor has left is thrown away, and the Tail
        the fetch opened for it goes with it -- otherwise an open file for every
        row a scan passed over. Never one the screen already holds: that one is
        closed when it is evicted or when the screen goes.
        """
        if tail is None or any(held is tail for held in self._tails.values()):
            return
        tail.close()

    def adopt_tail(self, addr: str, tail: metrics.Tail | None) -> None:
        """Keep the reader the fetch opened, evicting the oldest if need be."""
        if tail is None or self._tails.get(addr) is tail:
            return
        existing = self._tails.get(addr)
        if existing is not None:
            existing.close()
        self._tails[addr] = tail
        while len(self._tails) > _MAX_TAILS:
            oldest = next(iter(self._tails))
            if oldest == addr:
                break
            self._tails.pop(oldest).close()
            self.points.pop(oldest, None)
            self.columns.pop(oldest, None)

    def mount_plots(self, keys: list[render.PlotKey]) -> None:
        """Give each plot not on screen yet a widget of its own.

        One plot per metric rather than one shared pair of axes: `loss` sits
        around 2, `mfu` around 0.3 and `gpu_power_w` around 250, so overlaying
        them would flatten all but the largest into the baseline. Soloing one
        metric is what `y` is for.

        The widgets are held by reference rather than looked up by id: a plot is
        named by the metrics it draws and `train/loss` is not a Textual id.
        """
        container = self.query_one("#plots", widgets.PlotPane)
        for key in keys:
            if key in self.plots:
                continue
            plot = widgets.MetricPlot()
            plot.log_y = self.log_y
            plot.charset = self.charset
            self.plots[key] = plot
            container.mount(plot)

    def default_checked(self) -> set[str] | None:
        """Which metrics a newly seen name may check itself, or None for any.

        A phase that names its own plots checks only those, so that a viewer who
        steps off them with `space` or `x` lands on the metrics the phase cared
        about rather than on everything it logs.
        """
        if not self.pinned:
            return None
        return {y for y, _ in self.pinned}

    def plotted(self) -> list[render.PlotKey]:
        """The plots drawn right now.

        The soloed metric when there is one, then the phase's own plots, and
        otherwise every checked metric against the screen's x axis. Soloing
        deliberately does not count as taking the dashboard back: `y` twice
        returns to the phase's plots.
        """
        if self.solo is not None:
            return [(self.solo, self.x_axis)]
        if self.pinned:
            return list(self.pinned)
        metric_list = self.query_one("#metrics", widgets.MetricList)
        return [(name, self.x_axis) for name in metric_list.selected]

    def take_over_plots(self) -> None:
        """The viewer has asked for something the phase's plots cannot show."""
        self.overridden = True
        self.pinned = []

    def apply_selection(self) -> None:
        drawn = set(self.plotted())
        for key, plot in self.plots.items():
            plot.display = key in drawn
        empty = self.query_one("#empty", textual.widgets.Static)
        empty.display = not drawn
        if not drawn and self.plots:
            empty.update("no metrics selected")

    def update_plots(self) -> None:
        addr = self.address()
        series = self.points.get(addr or "", {})
        drawn = self.plotted()
        # Mounted here rather than up front: which plots exist follows what is
        # drawn now that a plot is a pair of metrics, and the pairs a viewer
        # never asks for are ones nothing would ever show.
        self.mount_plots(drawn)
        self.apply_selection()
        for name, x_axis in drawn:
            plot = self.plots.get((name, x_axis))
            if plot is not None:
                plot.plot = render.build_plot(name, series, x_axis)

    def redraw_metrics(self) -> None:
        self.update_plots()
        # Checking, unchecking and soloing all change whether `E` has anything
        # to write, and the footer is the only place that key is written down.
        self.sync_bindings()
        metric_list = self.query_one("#metrics", widgets.MetricList)
        series = self.points.get(self.address() or "", {})
        metric_list.show(
            render.metric_rows(
                metric_list.columns,
                {name: points[-1].value for name, points in series.items() if points},
                self.plotted(),
                self.solo,
            )
        )

    # -- actions ----------------------------------------------------------

    def action_focus_pane(self, slot: str) -> None:
        """Focus the pane a number key names, if it is on screen.

        `slot` is a position rather than a widget id: `top` is whichever of the
        config and the plots the content column is currently showing, since the
        two share the number.

        A number key moves the focus; it never changes what the content column
        shows. So `5` does nothing while the config is up -- there is no log
        pane to focus -- and the way to the log is through a pane that puts the
        dashboard back.
        """
        pane = self.right_column()[0] if slot == "top" else slot
        widget = self.query_one(f"#{pane}")
        if widget.display:
            widget.focus()

    def action_drill_in(self) -> None:
        """`enter` on a sidebar list: show me the one under the cursor.

        Which content pane that is has already been worked out --
        `right_column` reads it off the sidebar pane being followed -- so this
        is the same as `4` and needs no state of its own.
        """
        self.action_focus_pane("top")

    def action_back(self) -> None:
        """`escape`: out of the editor, then out of the pane, then nothing.

        Closing an editor keeps the file's value: a committed field has already
        been written, so there is never anything here to lose.
        """
        if self.query_one("#config", widgets.ConfigPane).close_editor():
            return
        if self.pane in self.right_column():
            self.action_focus_pane("runs" if self.showing_config() else "phases")

    def action_next_row(self) -> None:
        self.query_one("#config", widgets.ConfigPane).focus_row(1)

    def action_prev_row(self) -> None:
        self.query_one("#config", widgets.ConfigPane).focus_row(-1)

    def action_next_pane(self) -> None:
        """The next pane in the focused pane's own column.

        `tab` stays in its column -- crossing between the two is what the
        numbers are for -- so it cycles the three lists on the left, or the plots
        and the log on the right, and stays put on a config that is a column of
        one.
        """
        column = self.column(self.pane)
        index = column.index(self.pane) if self.pane in column else -1
        self.action_focus_pane(column[(index + 1) % len(column)])

    def action_toggle_metric(self) -> None:
        """`space`: draw or stop drawing the metric, or the extended range."""
        if not self.query_one("#metrics", widgets.MetricList).toggle_range():
            return
        # Checking a metric is a statement about the stacked view, so it also
        # says the viewer is done looking at one metric on its own -- and that
        # they want the dashboard rather than the plots the phase named.
        self.solo = None
        self.take_over_plots()
        self.redraw_metrics()

    def on_option_list_option_selected(self) -> None:
        # Enter on the metric list toggles too, so the list behaves the way the
        # tables do rather than needing a key of its own.
        self.action_toggle_metric()

    def action_solo_metric(self) -> None:
        name = self.query_one("#metrics", widgets.MetricList).highlighted_name
        if name is None:
            return
        self.solo = None if self.solo == name else name
        self.redraw_metrics()

    def action_cycle_x(self) -> None:
        choices = render.x_axis_choices(self.query_one("#metrics", widgets.MetricList).columns)
        index = choices.index(self.x_axis) if self.x_axis in choices else 0
        self.x_axis = choices[(index + 1) % len(choices)]
        # A phase's own plots carry their own x axes, so one screen-wide x is
        # not something they can be shown under: cycling it asks for the
        # dashboard.
        self.take_over_plots()
        self.redraw_metrics()

    def action_log_y(self) -> None:
        self.log_y = not self.log_y
        for plot in self.plots.values():
            plot.log_y = self.log_y

    def action_charset(self) -> None:
        """Draw the curves with braille dots instead of half-blocks, or back.

        Braille is finer -- 2x4 points per cell against 2x2 -- but needs a font
        that has the glyphs, which is why it is a key rather than the default.
        """
        self.charset = (
            render.CHARSET_BRAILLE if self.charset == render.CHARSET_BLOCK else render.CHARSET_BLOCK
        )
        for plot in self.plots.values():
            plot.charset = self.charset

    def export_metric(self) -> render.PlotKey | None:
        """The plot an export would write.

        The soloed one when there is one -- soloing is the viewer saying "this
        is the curve I am looking at" -- and otherwise the one the metrics pane
        has the cursor on, provided it is actually drawn. Falling back to the
        first drawn plot matters for a viewer who never left the plot pane: the
        metrics cursor sits on row zero, which may be a metric they unchecked.

        A plot rather than a metric name, so that `E` on one of a phase's own
        plots writes it against the x axis it is drawn against.
        """
        drawn = self.plotted()
        if not drawn:
            return None
        if self.solo is not None:
            return (self.solo, self.x_axis)
        highlighted = self.query_one("#metrics", widgets.MetricList).highlighted_name
        for key in drawn:
            if key[0] == highlighted:
                return key
        return drawn[0]

    def action_export_plot(self) -> None:
        """`E`: write the current curve out as a csv, png, svg or pdf.

        The plot is rebuilt here at full resolution rather than taken from the
        widget: `MetricPlot` holds the 1000-point thinning `render.build_plot`
        does by default, which is right for a character grid and wrong for a
        file someone will zoom into or load into a notebook. `limit=0` is
        `downsample`'s "leave it alone".
        """
        key = self.export_metric()
        addr = self.address()
        if key is None or addr is None:
            self.show_error("nothing to export; check a metric first", hold=True)
            return
        name, x_axis = key
        plot = render.build_plot(name, self.points.get(addr, {}), x_axis, limit=0)
        if plot is None:
            self.show_error(f"no points to export for {name}", hold=True)
            return
        run = self.selected_run_row()
        run_name = run.name if run is not None else "run"
        phase = self.selected_phase or "phase"
        caption = f"{run_name} -- {phase}"
        default = export.default_path(run_name, phase, name, export.FORMATS[0])

        def answered(spec: Export | None) -> None:
            if spec is not None:
                self.confirm_export(plot, spec, caption)

        self.host.ask(ExportScreen(default, export.FORMATS[0]), answered)

    def confirm_export(self, plot: render.Plot, spec: Export, caption: str) -> None:
        """Ask before writing over a file, the way `d` asks before deleting one."""
        if not spec.path.exists():
            self.write_export(plot, spec, caption)
            return

        def answered(answer: bool | None) -> None:
            if answer:
                self.write_export(plot, spec, caption)

        self.host.ask(ConfirmScreen(f"overwrite '{spec.path}'?"), answered)

    @textual.work(thread=True, group="export")
    def write_export(self, plot: render.Plot, spec: Export, caption: str) -> None:
        """Off the message loop: matplotlib takes a moment to import and to draw."""
        try:
            export.write(plot, spec.path, spec.fmt, log_y=self.log_y, caption=caption)
        except exceptions.UI as e:
            self.host.from_thread(self.held_error, str(e))
            return
        self.host.from_thread(self.held_message, f"wrote {spec.path}")

    def action_edit_config(self) -> None:
        """Edit the field the cursor is on, in place.

        `check_action` has already decided whether the key does anything, so
        the first guard is a fallback rather than the path a viewer takes; the
        second is not, since an image is free to declare no config at all.
        """
        form = self.query_one("#config", widgets.ConfigPane)
        if not self.showing_config() or not form.editable:
            self.show_error("only a configuring run can be edited", hold=True)
            return
        if not form.edit_here():
            self.show_error("this image declares no config", hold=True)

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Which of this screen's keys are live right now.

        Textual reads the three answers as enabled, `False` for hidden and
        `None` for greyed-but-shown, and both of the last two are wanted here:
        a key that editing has switched off is gone from the footer, because
        that footer is the only place the keys are written down and one that
        listed the five it will not do is worse than one that lists two; but
        `e` on a run that cannot be edited stays, greyed, because that is the
        answer to "why can I not edit this one".

        Editing switches the single-key shortcuts off. An `Input` swallows them
        already, but a `Switch` and a `Select` do not, so without this a `q`
        typed at a bool field would quit the app and take the edit with it.
        """
        if action == "export_plot":
            # Greyed rather than hidden: `E` is written in the footer of the two
            # panes that own it, and a viewer who has unchecked every metric is
            # owed the answer "there is no curve", not a key that vanished.
            return True if self.export_metric() is not None else None
        if action == "edit_config":
            if self.editing():
                return False
            # The run's status rather than `ConfigPane.editable`, which is the
            # same fact one tick late: the form is only told what it is showing
            # when the next snapshot lands, so asking it would leave `e` greyed
            # for the run the cursor was on a moment ago.
            return None if not (self.showing_config() and self.run_is("configuring")) else True
        if action == "chat_run" and not self.editing():
            # From the runs pane, where you pick the run, and from the phases
            # pane, which is where you end up: having just read a loss curve is
            # exactly when you want to hear what the thing it trained sounds
            # like, and going back up a pane to ask is a step for nothing.
            #
            # What it talks to follows the cursor. On the phases pane that is
            # the phase under it, whose data dir is a frozen snapshot of the run
            # as it stood when that phase ended; on the runs pane, where no
            # phase is in view, the run's newest servable one.
            if self.pane not in ("runs", "phases"):
                return False
            return True if self.chattable() is None else None
        if action in _LIFECYCLE and not self.editing():
            # Only from the pane the key is about: hidden elsewhere rather than
            # greyed, because on the metrics pane they are not keys that happen
            # not to apply -- they are not offered at all.
            #
            # Stop and restart reach the phases pane as a pair: standing on the
            # phase that is running is exactly where you decide to abandon it,
            # and offering the restart there without the stop would make the
            # cheaper of the two answers the one you had to leave the pane for.
            # Both still act on the whole run -- there is no stopping one phase.
            panes = ("runs", "phases") if action in ("restart_run", "stop_run") else ("runs",)
            if self.pane not in panes:
                return False
            # `n` is about the list, not about a run, so it is always live.
            if action == "new_run":
                return True
            run = self.selected_run_row()
            if run is None:
                return None
            if action == "start_run":
                return True if run.status == "configuring" else None
            if action == "stop_run":
                return True if run.status == "running" else None
            if action == "restart_run":
                # Every status but `configuring`, which is the only one
                # `runs.restart_run` refuses.
                return None if run.status == "configuring" else True
            # delete_run: a running run is stopped with `S` first, so that the
            # one prompt that kills an attempt is the one that says so.
            return None if run.status == "running" else True
        if self.editing() and action in _EDIT_MODE_OFF:
            return False
        return True

    def on_input_submitted(self) -> None:
        """`enter` in an editor: what is in it is what the field should be."""
        self.query_one("#config", widgets.ConfigPane).commit()

    def on_select_changed(self) -> None:
        """Picking an option is the whole edit; there is no enter to wait for."""
        self.query_one("#config", widgets.ConfigPane).commit(only_if_changed=True)

    def on_config_pane_edited(self, event: widgets.ConfigPane.Edited) -> None:
        """Write one field, and say so only when it could not be written.

        The rest of the config goes with it because `runs.write_config` takes
        the whole file: it validates every field against the schema, so it has
        to be given every field.
        """
        form = self.query_one("#config", widgets.ConfigPane)
        if self.selected_run is None:
            return
        # The values are read here, on the loop, and the write goes to a
        # thread: `write_run_config` validates against the image's schema, and
        # describing an image that is not in the cache is a `podman run`.
        self.write_config(
            self.selected_run,
            form.values(replacing=(event.path, event.value)),
            event.path,
            event.value,
        )

    @textual.work(thread=True, group="config")
    def write_config(
        self, run_id: str, values: dict[str, object], path: tuple[str, ...], value: object
    ) -> None:
        try:
            self.data.write_run_config(run_id, values)
        except exceptions.UI as e:
            self.host.from_thread(self.config_rejected, str(e))
            return
        self.host.from_thread(self.config_written, path, value)

    def config_rejected(self, message: str) -> None:
        """The editor stays open on a bad value, because the thing to do about
        it is to fix it."""
        self.show_error(message, hold=True)

    def config_written(self, path: tuple[str, ...], value: object) -> None:
        form = self.query_one("#config", widgets.ConfigPane)
        form.applied(path, value)
        form.close_editor()

    # -- lifecycle --------------------------------------------------------

    def action_start_run(self) -> None:
        """`s`: start the selected run.

        No prompt: starting a run that is only configured costs nothing to undo
        -- `S` is right there -- whereas stopping one throws work away.
        """
        run = self.selected_run_row()
        if run is None or run.status != "configuring":
            self.show_error("only a configuring run can be started", hold=True)
            return
        self.lifecycle(run.id, "start", None)

    def action_stop_run(self) -> None:
        """`S`: stop the selected run, once."""
        run = self.selected_run_row()
        if run is None or run.status != "running":
            self.show_error("only a running run can be stopped", hold=True)
            return
        run_id = run.id

        def answered(answer: bool | None) -> None:
            # A named function rather than a lambda: `lifecycle` is a worker
            # decorator, so it returns a `Worker` that a lambda would hand back
            # as the callback's result.
            if answer:
                self.lifecycle(run_id, "stop", None)

        self.host.ask(ConfirmScreen(f"stop run '{run.name}'?"), answered)

    def action_restart_run(self) -> None:
        """`R`: run it again, from the top or from the phase under the cursor.

        Asks, because a restart stops whatever attempt is running and starts a
        new one over the top of it -- the same work `S` throws away.
        """
        run = self.selected_run_row()
        if run is None or run.status == "configuring":
            self.show_error("only a started run can be restarted", hold=True)
            return
        # Only from the phases pane: on the runs pane there is no phase being
        # pointed at, and restarting from whichever one the cursor happened to
        # leave behind is not what pressing `R` over a run means.
        from_phase = self.selected_phase if self.pane == "phases" else None
        run_id = run.id
        question = (
            f"restart run '{run.name}'?"
            if from_phase is None
            else f"restart run '{run.name}' from phase '{from_phase}'?"
        )

        def answered(answer: bool | None) -> None:
            if answer:
                self.lifecycle(run_id, "restart", from_phase)

        self.host.ask(ConfirmScreen(question), answered)

    def action_delete_run(self) -> None:
        """`d`: delete the selected run, once.

        `runs.delete_run` removes the run directory, so the question says so.
        It is never forced: `check_action` greys `d` out on a running run, and
        stopping one is `S`'s prompt to ask.
        """
        run = self.selected_run_row()
        if run is None or run.status == "running":
            self.show_error("stop the run before deleting it", hold=True)
            return
        run_id = run.id

        def answered(answer: bool | None) -> None:
            if answer:
                self.lifecycle(run_id, "delete", None)

        self.host.ask(
            ConfirmScreen(f"delete run '{run.name}'? its logs and metrics go with it"), answered
        )

    def action_new_run(self) -> None:
        """`n`: create a run.

        Two hops, because both ends shell out: a worker reads the image and
        compute lists, the dialog opens on the loop with them in hand, and the
        answer goes back to a worker to be created.
        """
        self.offer_new_run()

    @textual.work(thread=True, group="lifecycle")
    def offer_new_run(self) -> None:
        try:
            choices = self.data.new_run_choices()
        except exceptions.UI as e:
            self.host.from_thread(self.held_error, str(e))
            return
        if not choices.images:
            self.host.from_thread(self.held_error, "no images; add one with 'utrain image add'")
            return
        self.host.from_thread(self.open_new_run, choices)

    def open_new_run(self, choices: data.NewRunChoices) -> None:
        def answered(answer: NewRun | None) -> None:
            if answer is not None:
                self.create(answer)

        self.host.ask(NewRunScreen(choices), answered)

    @textual.work(thread=True, group="lifecycle")
    def create(self, spec: NewRun) -> None:
        """`runs.create_run` describes the image, which starts a container."""
        try:
            run_id = self.data.create_run(spec.name, spec.image, spec.compute)
        except exceptions.UI as e:
            self.host.from_thread(self.held_error, str(e))
            return
        self.host.from_thread(self.select_when_listed, run_id)

    def select_when_listed(self, run_id: str) -> None:
        """Put the cursor on a run the next fetch will bring in.

        Not `select_run`: the row is not in the table yet, and the cursor is
        what the selection follows. `apply_runs` moves it once the row lands.
        """
        self._pending_run = run_id
        self.refresh_data()

    def forget_selection(self) -> None:
        """Let `apply_runs` pick a run again, after the selected one was deleted."""
        self.selected_run = None
        self.selected_phase = None
        self.phases = []
        self.reset_phase_view()
        self.refresh_data()

    @textual.work(thread=True, group="lifecycle")
    def lifecycle(self, run_id: str, op: _Op, from_phase: str | None) -> None:
        """Start, stop, restart or delete, off the message loop.

        A thread because all four shell out: `runs.start_run` and
        `runs.restart_run` call `container.podman.list_presets` and
        `container.podman.describe`, and the latter starts a container. On the
        loop that would freeze the app for seconds. They go through the query
        layer's own uncached describe rather than this app's cache -- what
        `start_run` needs is the query layer's business, and reaching in to hand
        it a cached answer would be a way for the two to disagree.

        Its own group, so the 1s refresh -- which is `exclusive` within
        `snapshot` -- can neither cancel this nor be cancelled by it.
        """
        try:
            if op == "start":
                self.data.start_run(run_id)
            elif op == "stop":
                self.data.stop_run(run_id)
            elif op == "restart":
                self.data.restart_run(run_id, from_phase)
            else:
                self.data.delete_run(run_id)
        except exceptions.UI as e:
            self.host.from_thread(self.held_error, str(e))
            return
        # Straight away rather than on the next tick: the run should turn green
        # and the config pane go read-only as the key is pressed. There is no
        # message on success -- the screen itself is the answer.
        if op == "delete":
            self.host.from_thread(self.forget_selection)
        else:
            self.host.from_thread(self.refresh_data)

    def chat_phase(self) -> str | None:
        """The phase `t` would talk to, or None to mean the newest servable one.

        From the phases pane it is the phase under the cursor; from the runs
        pane nothing is in view to name one, so the default stands.
        """
        return self.selected_phase if self.pane == "phases" else None

    def chattable(self) -> str | None:
        """Why `t` cannot talk to the selection, or None if it can.

        Everything `serve.start` refuses, asked before the key rather than
        reported after it. The image is looked up in the describe cache only --
        `described`, not `describe` -- because this runs on the message loop,
        both to grey the key and to answer it.
        """
        run = self.selected_run_row()
        if run is None:
            return "no run selected"
        described = self.data.described(run.image)
        if described is None:
            return f"still reading image '{run.image}'"
        # The image's answer first, and in the same order `serve.start` asks:
        # what it can do is true whatever state the run is in.
        servable = serve.servable_phases(described)
        if not servable:
            return f"image '{run.image}' does not support serve"
        phase = self.chat_phase()
        if phase is not None and phase not in servable:
            return f"image '{run.image}' does not serve phase '{phase}'"
        if run.status not in reconcile.TERMINAL:
            return f"run '{run.name}' is still {run.status}; chat needs a finished run"
        if phase is None:
            if not any(e.status == "done" and e.phase in servable for e in self.phases):
                return f"run '{run.name}' has no completed phase, so there is no model to serve"
            return None
        if not any(e.phase == phase and e.status == "done" for e in self.phases):
            return f"phase '{phase}' has not completed"
        return None

    def action_chat_run(self) -> None:
        """`t`: talk to what the selected run produced.

        The server is the chat screen's to start and to stop, so all that
        happens here is the push -- bringing a container up takes tens of
        seconds, and that wait belongs on the screen that is waiting.
        """
        run = self.selected_run_row()
        refusal = self.chattable()
        if run is None or refusal is not None:
            self.show_error(refusal or "no run selected", hold=True)
            return
        self.host.open(ChatScreen(self.host, self.data, run.id, run.name, self.chat_phase()))

    def action_images(self) -> None:
        self.host.open(ImagesScreen(self.host, self.data))

    def action_compute(self) -> None:
        self.host.open(ComputeScreen(self.host, self.data))


class ConfirmScreen(textual.screen.ModalScreen[bool]):
    """A yes/no question with two buttons.

    Not a `_Screen`: it has no refresh timer, no error line and no data of its
    own.

    The arrows move between the buttons, `enter` presses the focused one and
    `escape` cancels -- which is what a dialog is expected to do, and means the
    question can be answered without knowing a mnemonic. `No` holds the focus
    when it opens: the button that does nothing is the safe thing to land on
    when the other one kills a training run.
    """

    BINDINGS = [
        textual.binding.Binding("escape", "cancel", "cancel"),
        # `tab` reaches the buttons on its own; these make the arrows do it too,
        # since the two buttons read as a row rather than as a focus chain.
        # Their own actions rather than `focus_next`: that one lives on the app,
        # not the screen, so naming it here would bind to nothing at all.
        textual.binding.Binding("left", "prev_button", "", show=False),
        textual.binding.Binding("right", "next_button", "", show=False),
    ]

    def __init__(self, question: str) -> None:
        super().__init__()
        self.question = question

    def compose(self) -> textual.app.ComposeResult:
        with textual.containers.Vertical(id="confirm"):
            yield textual.widgets.Static(self.question, id="confirm-question")
            with textual.containers.Horizontal(id="confirm-buttons"):
                yield textual.widgets.Button("Yes", variant="error", id="confirm-yes", compact=True)
                yield textual.widgets.Button("No", variant="primary", id="confirm-no", compact=True)

    def on_mount(self) -> None:
        self.query_one("#confirm-no", textual.widgets.Button).focus()

    @textual.on(textual.widgets.Button.Pressed, "#confirm-yes")
    def _yes(self) -> None:
        self.dismiss(True)

    @textual.on(textual.widgets.Button.Pressed, "#confirm-no")
    def _no(self) -> None:
        self.dismiss(False)

    # Restricted to the buttons, so the arrows cycle the two answers rather
    # than walking whatever else the dialog might grow.
    def action_next_button(self) -> None:
        self.focus_next(textual.widgets.Button)

    def action_prev_button(self) -> None:
        self.focus_previous(textual.widgets.Button)

    def action_cancel(self) -> None:
        self.dismiss(False)


class NewRunScreen(textual.screen.ModalScreen[NewRun | None]):
    """Name, image and compute for a run that does not exist yet.

    The three fields `runs.create_run` takes, and no more: everything else about
    a run is config, and config is edited in the main screen's form once the run
    is there. Dismisses `None` on escape or cancel.

    The choices are handed in already fetched. Building them here would mean
    `podman images` and `nvidia-smi` running while the dialog composes, on the
    message loop.
    """

    BINDINGS = [
        textual.binding.Binding("escape", "cancel", "cancel"),
    ]

    def __init__(self, choices: data.NewRunChoices) -> None:
        super().__init__()
        self.choices = choices

    def compose(self) -> textual.app.ComposeResult:
        box = textual.containers.Vertical(id="new-run")
        # The dialog is titled on its border, the way the panes are, rather
        # than by a line inside it: it is one row shorter and it looks like the
        # rest of the app.
        box.border_title = "new run"
        with box:
            with textual.containers.Horizontal(classes="new-run-row"):
                yield textual.widgets.Label("name")
                yield textual.widgets.Input(id="new-run-name", compact=True)
            with textual.containers.Horizontal(classes="new-run-row"):
                yield textual.widgets.Label("image")
                # `allow_blank=False` so there is always an answer to read back:
                # the caller has already checked that there is at least one
                # image, and `compute_options` always offers the cpu.
                yield textual.widgets.Select[str](
                    [(name, name) for name in self.choices.images],
                    allow_blank=False,
                    compact=True,
                    id="new-run-image",
                )
            with textual.containers.Horizontal(classes="new-run-row"):
                yield textual.widgets.Label("compute")
                yield textual.widgets.Select[str](
                    render.compute_options(self.choices.compute),
                    allow_blank=False,
                    compact=True,
                    id="new-run-compute",
                )
            yield textual.widgets.Static("", id="new-run-error")
            with textual.containers.Horizontal(id="new-run-buttons"):
                yield textual.widgets.Button(
                    "Create", variant="primary", id="new-run-create", compact=True
                )
                yield textual.widgets.Button(
                    "Cancel", variant="default", id="new-run-cancel", compact=True
                )

    def on_mount(self) -> None:
        # The name is the one field with nothing to default to, so it is where
        # the cursor starts; `tab` reaches the rest.
        self.query_one("#new-run-name", textual.widgets.Input).focus()

    def run_name(self) -> str:
        """Not `name`: `DOMNode` already has one, and it means something else."""
        return self.query_one("#new-run-name", textual.widgets.Input).value.strip()

    def _picked(self, identifier: str) -> str:
        """One `Select`'s value.

        The type parameter goes on the annotation rather than on the argument,
        for the reason `_Screen.table` gives: `query_one` isinstance-checks
        whatever class it is handed, and a subscripted generic cannot be.
        """
        picked: textual.widgets.Select[str] = self.query_one(identifier, textual.widgets.Select)
        return str(picked.value)

    @textual.on(textual.widgets.Input.Submitted, "#new-run-name")
    def _submitted(self) -> None:
        """`enter` in the name field creates, rather than doing nothing."""
        self._create()

    @textual.on(textual.widgets.Button.Pressed, "#new-run-create")
    def _create(self) -> None:
        name = self.run_name()
        if not name:
            # Answered here rather than by `create_run` refusing it later: the
            # field to fix in is on screen.
            self.query_one("#new-run-error", textual.widgets.Static).update("a run needs a name")
            return
        self.dismiss(
            NewRun(
                name=name,
                image=self._picked("#new-run-image"),
                compute=self._picked("#new-run-compute"),
            )
        )

    @textual.on(textual.widgets.Button.Pressed, "#new-run-cancel")
    def _cancel(self) -> None:
        self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class ExportScreen(textual.screen.ModalScreen[Export | None]):
    """Where to write one plot, and in what format.

    Two fields, because there are only two decisions: the metric is already
    settled by the cursor and the axes by the dashboard's own keys, so an
    export dialog that asked about them again would be asking the viewer to say
    twice what the screen behind it already shows.

    The path is prefilled and relative, so `enter` straight away is a complete
    answer and the file lands where `utrain tui` was started. Dismisses `None`
    on escape or cancel.
    """

    BINDINGS = [
        textual.binding.Binding("escape", "cancel", "cancel"),
    ]

    def __init__(self, path: pathlib.Path, fmt: str) -> None:
        super().__init__()
        self._path = path
        self._fmt = fmt

    def compose(self) -> textual.app.ComposeResult:
        box = textual.containers.Vertical(id="export")
        box.border_title = "export plot"
        with box:
            with textual.containers.Horizontal(classes="new-run-row"):
                yield textual.widgets.Label("format")
                yield textual.widgets.Select[str](
                    [(name, name) for name in export.FORMATS],
                    value=self._fmt,
                    allow_blank=False,
                    compact=True,
                    id="export-format",
                )
            with textual.containers.Horizontal(classes="new-run-row"):
                yield textual.widgets.Label("file")
                yield textual.widgets.Input(value=str(self._path), id="export-path", compact=True)
            yield textual.widgets.Static("", id="export-error")
            with textual.containers.Horizontal(id="export-buttons"):
                yield textual.widgets.Button(
                    "Export", variant="primary", id="export-write", compact=True
                )
                yield textual.widgets.Button(
                    "Cancel", variant="default", id="export-cancel", compact=True
                )

    def on_mount(self) -> None:
        # The path is the field a viewer is most likely to want to change, and
        # the one `enter` submits from; `tab` reaches the format.
        self.query_one("#export-path", textual.widgets.Input).focus()

    def _field(self) -> textual.widgets.Input:
        return self.query_one("#export-path", textual.widgets.Input)

    def _format(self) -> str:
        picked: textual.widgets.Select[str] = self.query_one(
            "#export-format", textual.widgets.Select
        )
        return str(picked.value)

    @textual.on(textual.widgets.Select.Changed, "#export-format")
    def _reformat(self) -> None:
        """Track the format in the filename -- unless the name was typed over.

        The extension is swapped only while the field still ends in the one
        this dialog last offered. A viewer who typed `~/paper/fig1.png` and then
        picked `pdf` gets `~/paper/fig1.pdf`; one who typed a name ending in
        something else meant it, and having the picker rewrite it under them
        would be the dialog arguing.
        """
        chosen = self._format()
        field = self._field()
        current = pathlib.Path(field.value.strip())
        if current.suffix == f".{self._fmt}":
            field.value = str(export.replace_suffix(current, chosen))
        self._fmt = chosen

    @textual.on(textual.widgets.Input.Submitted, "#export-path")
    def _submitted(self) -> None:
        """`enter` in the path field exports, rather than doing nothing."""
        self._write()

    @textual.on(textual.widgets.Button.Pressed, "#export-write")
    def _write(self) -> None:
        raw = self._field().value.strip()
        if not raw:
            # Answered here rather than by the write failing later: the field to
            # fix is on screen.
            self.query_one("#export-error", textual.widgets.Static).update("where should it go?")
            return
        self.dismiss(Export(path=pathlib.Path(raw).expanduser(), fmt=self._format()))

    @textual.on(textual.widgets.Button.Pressed, "#export-cancel")
    def _cancel(self) -> None:
        self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class AddImageScreen(textual.screen.ModalScreen[str | None]):
    """The one argument `utrain image add` takes: where to pull from.

    A dialog rather than a prompt on the images screen itself, for the reason
    the new-run one is: a field that is only sometimes there moves the list
    under the viewer every time it opens. Dismisses `None` on escape or cancel.
    """

    BINDINGS = [
        textual.binding.Binding("escape", "cancel", "cancel"),
    ]

    def compose(self) -> textual.app.ComposeResult:
        box = textual.containers.Vertical(id="add-image")
        box.border_title = "add image"
        with box:
            with textual.containers.Horizontal(classes="new-run-row"):
                yield textual.widgets.Label("url")
                yield textual.widgets.Input(
                    # The three shapes `images.add_image` accepts, which is more
                    # than the word "url" says on its own.
                    placeholder="docker://host/path:tag, a registry ref, or podman://local",
                    id="add-image-url",
                    compact=True,
                )
            yield textual.widgets.Static("", id="add-image-error")
            with textual.containers.Horizontal(id="add-image-buttons"):
                yield textual.widgets.Button(
                    "Add", variant="primary", id="add-image-add", compact=True
                )
                yield textual.widgets.Button(
                    "Cancel", variant="default", id="add-image-cancel", compact=True
                )

    def on_mount(self) -> None:
        self.query_one("#add-image-url", textual.widgets.Input).focus()

    def url(self) -> str:
        return self.query_one("#add-image-url", textual.widgets.Input).value.strip()

    @textual.on(textual.widgets.Input.Submitted, "#add-image-url")
    def _submitted(self) -> None:
        """`enter` in the field adds, rather than doing nothing."""
        self._add()

    @textual.on(textual.widgets.Button.Pressed, "#add-image-add")
    def _add(self) -> None:
        url = self.url()
        if not url:
            # Answered here rather than by podman refusing it a minute later:
            # the field to fix is on screen.
            self.query_one("#add-image-error", textual.widgets.Static).update(
                "where should it come from?"
            )
            return
        self.dismiss(url)

    @textual.on(textual.widgets.Button.Pressed, "#add-image-cancel")
    def _cancel(self) -> None:
        self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


def _shutdown_server(server: serve.Server) -> None:
    """Stop a serve container, on a plain thread.

    Not a worker: `shutdown` waits up to ten seconds on a container that
    ignores SIGTERM, which on the message loop is ten seconds of frozen app --
    and the screen's own workers are being cancelled as it leaves, so it cannot
    be one of those either. It touches no widgets, so it needs nothing from the
    loop.
    """
    threading.Thread(target=server.shutdown, daemon=True).start()


class ChatScreen(_Screen):
    """A conversation with what one run produced.

    The container is this screen's for exactly as long as it is open: `on_mount`
    spawns it, `on_unmount` stops it. Keeping one alive in the background would
    make re-entry instant, at the price of the app owning a container that
    outlives the thing that shows it -- and a serve container holds a GPU.

    Nothing here polls, so `_Screen`'s refresh timer is not started: a
    transcript changes when the model says something or when the viewer does.

    Every blocking call is a thread. Starting the container covers importing
    torch and loading a checkpoint -- tens of seconds is normal -- and a turn
    lasts as long as the model takes to generate it.
    """

    BINDINGS = [
        textual.binding.Binding("escape", "app.pop_screen", "back"),
    ]

    def __init__(
        self,
        host: Host,
        source: data.Data,
        run_id: str,
        run_name: str,
        phase: str | None = None,
    ) -> None:
        super().__init__(host, source)
        self.run_id = run_id
        self.run_name = run_name
        self.phase = phase
        self.server: serve.Server | None = None
        self.client: chatmod.Client | None = None
        # Set when the screen is left, under `_lock`; see `adopt_server`.
        self._closed = False
        # Deltas that have arrived but not yet been drawn; see `_flush`.
        self._pending: list[str] = []
        self._lock = threading.Lock()

    def compose_body(self) -> textual.app.ComposeResult:
        yield widgets.ChatLog(id="chat-log")
        # Disabled until the container is listening: a turn typed at a server
        # that does not exist yet would be sent to nothing.
        yield textual.widgets.Input(placeholder="starting...", id="chat-input", compact=True)

    def on_mount(self) -> None:
        """`_Screen`'s, without the refresh timer -- there is nothing to poll."""
        self.log_widget().border_title = f"chat: {self.run_name}"
        self.field().disabled = True
        self.log_widget().note(f"starting a server for '{self.run_name}'...")
        self.connect()

    def refresh_data(self) -> None:
        """`r` is the base screen's key; here there is nothing behind it."""

    def log_widget(self) -> widgets.ChatLog:
        return self.query_one("#chat-log", widgets.ChatLog)

    def field(self) -> textual.widgets.Input:
        return self.query_one("#chat-input", textual.widgets.Input)

    # -- the server -------------------------------------------------------

    @textual.work(thread=True, group="chat")
    def connect(self) -> None:
        """Spawn the container and wait for it to publish its port.

        Both halves are slow enough for the viewer to press escape in the
        middle of them -- `podman run` takes seconds and `wait_for_port` waits
        minutes -- so both check afterwards whether the screen is still there,
        and a container that outlived it is stopped here.
        """
        try:
            server = self.data.start_server(self.run_id, self.phase)
        except exceptions.UI as e:
            self.report(self.failed, str(e), None)
            return
        if not self.adopt_server(server):
            _shutdown_server(server)
            return
        try:
            port = server.wait_for_port()
        except exceptions.UI as e:
            self.report(self.failed, str(e), server.log_tail(20))
            return
        self.report(self.ready, port)

    def adopt_server(self, server: serve.Server) -> bool:
        """Hand the started container to the screen. False if it has gone."""
        with self._lock:
            if self._closed:
                return False
            self.server = server
            return True

    def report(self, callback: collections.abc.Callable[..., None], *args: object) -> None:
        """Call back onto the message loop, unless the screen has gone.

        Every one of these follows a wait long enough for the viewer to leave,
        and every one of them touches a widget: `query_one` on an unmounted
        screen raises `NoMatches` where nothing can catch it.
        """
        if self._closed:
            return
        self.host.from_thread(callback, *args)

    def ready(self, port: int) -> None:
        """The container is listening: let the viewer type."""
        self.client = chatmod.Client(port, _CHAT_MAX_TOKENS, _CHAT_TEMPERATURE)
        server = self.server
        phase = f", phase '{server.phase}'" if server is not None else ""
        image = server.image if server is not None else ""
        self.log_widget().note(f"serving {image}{phase} on http://127.0.0.1:{port}/v1")
        field = self.field()
        field.disabled = False
        field.placeholder = "say something ('/reset' forgets the conversation)"
        field.focus()

    def failed(self, message: str, tail: list[str] | None) -> None:
        """Report a server that never came up, or one that died mid-session.

        The log tail goes into the transcript rather than only onto the status
        line: it is several lines, it is the only place a serve container's
        output is ever seen, and the line under the screen holds one.
        """
        self.show_error(message, hold=True)
        self.log_widget().note(message)
        if tail:
            log = self.server.log_path if self.server is not None else "the container log"
            self.log_widget().note(f"{log}:\n" + "\n".join(f"  {line}" for line in tail))
        field = self.field()
        field.disabled = True
        field.placeholder = "no server"

    def on_unmount(self) -> None:
        """Stop the container on the way out, and say that the screen has gone.

        Under the lock, because `connect` is racing this: whichever of the two
        goes second is the one that stops the container.
        """
        with self._lock:
            self._closed = True
            server = self.server
        if server is not None:
            _shutdown_server(server)

    # -- turns ------------------------------------------------------------

    @textual.on(textual.widgets.Input.Submitted, "#chat-input")
    def _submitted(self) -> None:
        field = self.field()
        text = field.value.strip()
        if not text:
            return
        field.value = ""
        if text == "/reset":
            # The same word the CLI's REPL uses. There is no `/quit` -- escape
            # is how a screen is left here -- and no `/help`, because a TUI
            # writes its keys in the footer.
            if self.client is not None:
                self.client.reset()
            self.log_widget().clear_all()
            return
        transcript = self.log_widget()
        transcript.start_turn("user", text)
        transcript.end_turn()
        transcript.start_turn("assistant")
        # Locked out for the turn: the API is stateless and the history is
        # resent whole, so a second turn sent while the first is streaming
        # would be sent without the reply it is answering.
        field.disabled = True
        self.send(text)

    @textual.work(thread=True, group="chat-turn")
    def send(self, text: str) -> None:
        """One turn, streamed back onto the message loop as it arrives."""
        client = self.client
        if client is None:
            return
        try:
            for delta in client.stream(text):
                self.buffer(delta)
        except exceptions.UI as e:
            self.report(self.show_error, str(e))
        except OSError:
            # The connection broke, which for a local server means the
            # container is gone. Its log is the only account of why.
            server = self.server
            tail = server.log_tail(20) if server is not None else None
            self.report(self.failed, "the server stopped answering", tail)
            return
        self.report(self.finish)

    def buffer(self, delta: str) -> None:
        """Hold a delta for the next flush, from the worker thread.

        Every few characters would otherwise be a message on the loop. The
        flush is scheduled only when the buffer was empty, so a fast reply
        costs one scheduled call per interval rather than one per token.
        """
        with self._lock:
            first = not self._pending
            self._pending.append(delta)
        if first:
            self.report(self.schedule_flush)

    def schedule_flush(self) -> None:
        self.set_timer(_CHAT_FLUSH_SECONDS, self.flush)

    def flush(self) -> None:
        with self._lock:
            text = "".join(self._pending)
            self._pending.clear()
        if text:
            self.log_widget().append(text)

    def finish(self) -> None:
        """The reply is complete: draw what is left of it and take the lock off."""
        self.flush()
        self.log_widget().end_turn()
        field = self.field()
        field.disabled = False
        field.focus()


class ImagesScreen(_Screen):
    """Images in the local store, as `utrain image list` shows them.

    And, on `a`, `utrain image add`: this is the list the added image appears
    in, so it is where adding one belongs.
    """

    TITLE = "utrain"
    SUB_TITLE = "images"

    BINDINGS = [
        textual.binding.Binding("a", "add_image", "add"),
    ]

    def __init__(self, host: Host, source: data.Data) -> None:
        super().__init__(host, source)
        # A pull is in flight. The list keeps refreshing under it -- so that the
        # image appears the moment it lands -- and `apply` clears the status
        # line, which would take "pulling..." off the screen a second after it
        # went up.
        self.pulling = False

    def compose_body(self) -> textual.app.ComposeResult:
        yield _table("images", "images", render.IMAGE_COLUMNS)

    def refresh_data(self) -> None:
        self.fetch()

    @textual.work(thread=True, exclusive=True, group="images")
    def fetch(self) -> None:
        try:
            rows = [[i.name, i.size_str, str(i.run_count)] for i in self.data.list_images()]
        except exceptions.UI as e:
            self.host.from_thread(self.show_error, str(e))
            return
        self.host.from_thread(self.apply, rows)

    def apply(self, rows: list[list[render.Cell]]) -> None:
        if not self.pulling:
            self.clear_error()
        _fill(self.table("#images"), rows)

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """One pull at a time: `a` is greyed while one is running.

        Greyed rather than hidden, because the reason it cannot be pressed is
        written on the line right under it.
        """
        if action == "add_image":
            return None if self.pulling else True
        return True

    def action_add_image(self) -> None:
        """`a`: pull an image and tag it as a preset."""
        if self.pulling:
            return

        def answered(answer: str | None) -> None:
            if answer is not None:
                self.add(answer)

        self.host.ask(AddImageScreen(), answered)

    @textual.work(thread=True, group="image-add")
    def add(self, url: str) -> None:
        """`podman pull`, which can take minutes on a container of any size.

        Its own group, so the 1s refresh -- which is `exclusive` within
        `fetch` -- can neither cancel this nor be cancelled by it.
        """
        self.host.from_thread(self.pull_started, url)
        try:
            name = self.data.add_image(url)
        except exceptions.UI as e:
            self.host.from_thread(self.pull_finished, "", str(e))
            return
        self.host.from_thread(self.pull_finished, name, "")

    def pull_started(self, url: str) -> None:
        self.pulling = True
        self.refresh_bindings()
        # Not held: a hold expires, and this has to stay up for as long as the
        # pull does. `apply` leaves it alone while `pulling` is set.
        self.show_message(f"pulling {url}...")

    def pull_finished(self, name: str, error: str) -> None:
        self.pulling = False
        self.refresh_bindings()
        if error:
            self.held_error(error)
        else:
            self.held_message(f"added {name}")
        # Straight away rather than on the next tick: the row is what the key
        # was pressed for.
        self.refresh_data()


class ComputeScreen(_Screen):
    """Host CPU and GPUs, as `utrain compute list` shows them."""

    TITLE = "utrain"
    SUB_TITLE = "compute"

    def compose_body(self) -> textual.app.ComposeResult:
        yield _table("compute", "compute", render.COMPUTE_COLUMNS)

    def refresh_data(self) -> None:
        self.fetch()

    @textual.work(thread=True, exclusive=True, group="compute")
    def fetch(self) -> None:
        try:
            rows = render.compute_rows(self.data.compute())
        except exceptions.UI as e:
            # An unreadable `/proc/meminfo` or a malformed fixture is a line
            # under the table, as it is on the other two screens.
            self.host.from_thread(self.show_error, str(e))
            return
        self.host.from_thread(self.apply, rows)

    def apply(self, rows: list[list[render.Cell]]) -> None:
        self.clear_error()
        _fill(self.table("#compute"), rows)
