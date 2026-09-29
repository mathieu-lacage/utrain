"""The TUI's screens: one main screen, plus two panels over it and a chat.

The layout is lazygit's. A narrow left column stacks the three lists that say
what you are looking at -- runs, then that run's phases, then that phase's
metrics -- and a wide right column shows the content: the run's config while the
runs pane has focus and the run can still be configured, and otherwise the
metric plots above the phase's log.

Navigation is selection, not a screen stack. Everything is on screen at once, so
one fetch fills all of it: `data.Data.snapshot` is a single worker producing a
single `Snapshot`, applied in one pass. Five panes refreshing on five schedules
would show five slightly different moments.

Images and compute are the exception, and they are panels rather than screens
of their own: the two resources a run is made of, glanced at from wherever the
viewer was and closed with `escape`. They are pushed over the screen they were
asked from, so nothing of what was being watched moves -- see `_Popover`.

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

import rich.text
import textual.app
import textual.binding
import textual.containers
import textual.coordinate
import textual.screen
import textual.suggester
import textual.timer
import textual.widget
import textual.widgets
import textual.widgets.option_list

from .. import chat as chatmod
from .. import exceptions, metrics, reconcile, serve, tuistate, types
from .. import sweeps as sweepsmod
from . import commands, data, export, menus, render, widgets
from . import compare as comparemod

# The screen's actions that a single key reaches, and which are therefore
# switched off while an editor has the focus (`q` is the app's, and
# `UtrainApp.check_action` switches that one off). `back` is not here: it is
# how you get out again.
_EDIT_MODE_OFF = frozenset(
    {
        "focus_pane",
        "focus_list",
        "focus_content",
        "next_pane",
        "prev_pane",
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
        # And a `?` is a character, not "show me the keys".
        "help",
        # And an `m` is a letter, not "open the metric picker".
        "toggle_metrics",
        # Nor is a `z` a zoom, or a space a mark.
        "zoom",
        "toggle_mark",
        "expand",
        "collapse",
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

    def charset(self) -> str:
        """Which character set to draw curves with, before the viewer says.

        The app's to answer because the console is the app's: see
        `render.default_charset` for what the guess is made of.
        """
        ...

    def close(self) -> None:
        """Pop the screen on top of the stack.

        The counterpart to `open`: the app owns the stack, so a screen that has
        finished asks rather than reaches for `self.app`.
        """

    def go(self, which: str) -> None:
        """Show one of the resource panels -- `images` or `compute`.

        Over this screen, not instead of it: a panel is a modal, and `escape`
        is how it is closed. The app owns the panel instances, so that one
        stays alive while it is closed -- a pull does not fit in the time a
        panel is open (see `ImagesScreen.add`).
        """

    def from_thread(
        self,
        callback: collections.abc.Callable[..., None],
        *args: object,
    ) -> None:
        """Run ``callback`` on the message loop, from a worker thread."""

    def new_sweep(self, base: str | None) -> None:
        """Ask for a new sweep and create it: from a run's config, or an image's.

        The app's, because the dialogs outlive the screen they were asked
        from -- the sweep is shown in Sweeps once it exists, wherever that was.
        """

    def show_run(self, run_id: str) -> None:
        """Go to Runs, with the cursor on this run."""

    def show_sweep(self, sweep_id: str) -> None:
        """Go to Sweeps, with the cursor on this sweep."""

    def compare_sweep(self, sweep_id: str) -> None:
        """Go to Compare, comparing this sweep's runs."""


class _Screen(textual.screen.Screen[None]):
    """Shared chrome: a header, a footer, an error line and a refresh timer."""

    # Every workspace's bindings go through `commands.footer`: the footer lists
    # the keys that move around the workspace, and the menu the commands.
    BINDINGS = commands.footer(
        [
            # Textual's own, restated to be listed: moving between panes is what
            # the footer is for.
            textual.binding.Binding("tab", "app.focus_next", "pane"),
            textual.binding.Binding("shift+tab", "app.focus_previous", "pane", show=False),
            textual.binding.Binding("r", "force_refresh", "refresh"),
            textual.binding.Binding("escape", "back", "back", show=False),
            # `?` is the app's, bound to this screen's `help`: an app binding
            # is listed after the screen's, so the keys that go anywhere --
            # the menu, goto, help -- come last in every footer.
        ]
    )

    def __init__(self, host: Host, source: data.Data) -> None:
        super().__init__()
        self.host = host
        self.data = source
        # Kept as well as displayed: a `Static`'s content is not readable back
        # off the widget, and the tests want to know what the viewer was last
        # told.
        self.error = ""
        self._held_until = 0.0
        # None until `on_mount` starts it, and left as None by a screen that
        # has nothing to poll -- `ChatScreen`, whose transcript changes when
        # somebody says something and not on a clock.
        self.timer: textual.timer.Timer | None = None

    # How often the screen fetches. A class attribute so that a screen whose
    # fetch is heavier -- Compare reads every run in its set -- can ask less.
    REFRESH_SECONDS = _REFRESH_SECONDS

    # The workspace this screen belongs to, which is the title the top row
    # picks out. A screen pushed within a workspace -- a chat -- keeps its
    # workspace's.
    WORKSPACE = "runs"

    def compose(self) -> textual.app.ComposeResult:
        bar = menus.TopBar(self.WORKSPACE, menus.available(self.host))
        yield bar
        yield from self.compose_body()
        yield textual.widgets.Static("", id="error")
        yield textual.widgets.Footer()

    def show_top_status(self) -> None:
        """Put the app's latest status on this screen's top row.

        The app refreshes it on a timer of its own and draws it on the screen
        in front; a screen coming back into view has missed those draws.
        """
        line = str(getattr(self.host, "status_line", ""))
        for bar in self.query(menus.TopBar):
            bar.show_status(line)

    def compose_body(self) -> textual.app.ComposeResult:
        return iter(())

    def on_mount(self) -> None:
        # The timer is in place before the first fetch, because `refresh_data`
        # restarts it.
        self.timer = self.set_interval(self.REFRESH_SECONDS, self.refresh_data)
        self.refresh_data()

    # The pane `escape` comes back to on a workspace's own screen -- which is
    # the bottom of its stack, with nothing under it to go back to. None for
    # a screen pushed over one (a chat, a panel), which `escape` closes.
    HOME: str | None = None

    def action_back(self) -> None:
        """`escape`: leave this screen, or on a workspace, go back to its list.

        MainScreen overrides this with its ladder, since it is where the ladder
        bottoms out: out of the editor or the picker, out of the pane, up one
        list level, and there it stops -- `q` is how the app is left. The other
        workspaces stop at their `HOME` pane the same way. A panel (see
        `_Popover`) or a chat is closed, which is the point of it being pushed
        rather than switched to.
        """
        if self.HOME is not None:
            self.query_one(self.HOME).focus()
            return
        self.host.close()

    def action_help(self) -> None:
        """`?`: the keys that are not in the footer.

        The footer carries the keys of the pane in focus, which is what makes
        it useful and what stops it carrying the navigation: the numbers and
        `tab` and `enter` are `show=False` precisely so there is room for the
        rest. This is where they are written down.
        """
        self.host.ask(HelpScreen(), lambda _answer: None)

    def on_screen_suspend(self) -> None:
        """Stop polling a screen nobody is looking at.

        Each destination's screen stays alive while another is showing, and its
        refresh is a `podman` call: three of them ticking at once would spend
        most of the app's time answering questions nobody asked. A modal
        suspends the screen under it too, which is the same argument.
        """
        if self.timer is not None:
            self.timer.pause()

    def on_screen_resume(self) -> None:
        self.show_top_status()
        if self.timer is None:
            return
        self.timer.resume()
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
        # A screen can be gone by the time this runs: a fetch outlives the
        # panel that started it, and a pop takes a screen's widgets away while
        # the screen itself is kept. `is_mounted` still answers yes then -- it
        # means "has been mounted", and never goes back -- so the line is
        # looked up softly rather than asked for: no line, nothing to write on.
        self.error = message
        self._held_until = time.monotonic() + _MESSAGE_SECONDS if hold else 0.0
        lines = self.query("#error")
        if not lines:
            return
        line = typing.cast(textual.widgets.Static, lines.first())
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
        lines = self.query("#error")
        if not lines:
            return
        line = typing.cast(textual.widgets.Static, lines.first())
        line.remove_class(_OK_CLASS)
        line.update("")


class _Popover(_Screen, textual.screen.ModalScreen[None]):
    """A panel over the screen beneath: the images, the compute devices.

    Not destinations: they are the two resources a run is made of, glanced at
    from wherever the viewer was, so `i` pushes one and `escape` closes it --
    and closing it suspends the screen underneath rather than throwing it away,
    which is what keeps the run, the phase and the tab the viewer had selected.

    The app holds one instance of each panel rather than building one per
    visit: `ImagesScreen.add` shells out to `podman pull`, which outlives the
    panel being closed, and its callbacks are written on the instance. A pop
    takes the panel's widgets away while the instance is kept, so callbacks
    that arrive after the close check `_closed` before they touch any.
    """

    BINDINGS = [
        # Shown, unlike on the screens it covers: the panel's footer is the
        # only key list the panel has, and `escape` is the whole of how it is
        # left. The `i`/`c` of the screen underneath stay there -- a modal
        # keeps them from firing, so there is no sideways step to offer.
        textual.binding.Binding("escape", "back", "close"),
        # The app's `?` does not reach a modal, so the panel binds its own.
        textual.binding.Binding("question_mark", "help", "help", key_display="?"),
    ]

    def __init__(self, host: Host, source: data.Data) -> None:
        super().__init__(host, source)
        # Set when the panel is closed, cleared when it is opened again.
        # `is_mounted` is no use here: in this Textual it means "has been
        # mounted", and stays yes after a pop has taken the widgets away --
        # the same latch `ChatScreen` sets, for the same reason.
        self._closed = False

    def compose(self) -> textual.app.ComposeResult:
        # A surface under the panes rather than a box around them: the list is
        # framed and titled like every other pane in the app, and a second
        # border around it would be chrome saying what the first already says.
        with textual.containers.Vertical(id="panel"):
            yield from self.compose_body()
            yield textual.widgets.Static("", id="error")
            yield textual.widgets.Footer()

    def on_mount(self) -> None:
        # After `super()`, which starts the first fetch: the list is where the
        # keys go from the moment the panel is up, and there is nothing else
        # here to focus.
        self._closed = False
        super().on_mount()
        self.query(textual.widgets.DataTable).first().focus()

    def on_unmount(self) -> None:
        # Fired by the pop that takes the panel off the stack, after its
        # widgets have gone -- which is what makes the latch answer the only
        # question the callbacks ask: is there anything left to draw on?
        self._closed = True


def _table(
    identifier: str,
    title: str,
    columns: tuple[str, ...],
    drill: bool = True,
    pane: bool = False,
) -> textual.widgets.DataTable[render.Cell]:
    """One of the screens' lists, framed and titled like a lazygit pane.

    `drill=False` for a list with nothing under it -- the images and the
    compute devices are the whole of what there is to see about them -- so that
    its footer does not offer an `enter` that would do nothing. `pane=True`
    for one of a workspace's panes, which the app's `.pane` rule frames and
    lights up on focus.
    """
    table: textual.widgets.DataTable[render.Cell] = (
        widgets.PaneTable(id=identifier)
        if drill
        else textual.widgets.DataTable[render.Cell](id=identifier)
    )
    table.cursor_type = "row"
    table.border_title = title
    table.add_columns(*columns)
    if pane:
        table.add_class("pane")
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


def _stale(
    event: textual.widgets.DataTable.RowHighlighted,
    table: textual.widgets.DataTable[render.Cell],
) -> bool:
    """Whether the cursor has already left the row this event is about.

    The cursor *is* the selection, so a highlight message naming any other row
    is out of date by the time it is read. Two arrive together whenever a table
    is filled for the first time: Textual highlights row 0 as the cursor
    becomes valid, and `apply_phases` moves the cursor to the phase the run has
    got to in the same pass. A snapshot landing between the two -- it arrives
    from a thread, not through the message queue -- would read the first as the
    viewer having selected row 0 and move the cursor down onto it to match.

    It also collapses a burst of cursor moves: only the row the viewer stopped
    on is selected, rather than every row crossed on the way.

    The table is passed rather than read off the event, whose `data_table` is
    typed as an unparameterised `DataTable`.
    """
    return event.cursor_row != table.cursor_row


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
    """The Runs workspace: a tree of sweeps, runs and phases, and what one is.

    One list on the left, and fixed panes on the right for whatever the cursor
    is on, the shape every workspace has. The list is a tree -- a sweep, the
    runs it made, a run's phases -- opened and closed in place, so the runs are
    always in view around the one being looked at. The right side does not
    have views to pick between:

    - a run that has started, or one of its phases: the plots above the log,
      for that phase or for the phase the run is on now;
    - a run that has not started (configuring, or a sweep run still queued):
      its config, which is all there is of it yet, edited in place;
    - a sweep: its runs as a status grid.

    `z` gives the focused pane the whole right side, and `e` on a run that has
    started shows its config in a read-only popup, since there is nothing left
    to change in it.

    The metrics reader for each phase is held open: following a live phase means
    resuming from where the last read stopped, and that needs the reader itself,
    not just an offset (see `utrain.metrics`). Because the selected phase
    changes with a cursor move rather than with a screen push, those readers are
    kept in a small cache and closed as they fall out of it.
    """

    TITLE = "utrain"

    BINDINGS = commands.footer(
        [
            # `tab` is listed; the numbers are not, each pane carrying its own in
            # its title.
            textual.binding.Binding("tab", "next_pane", "pane"),
            textual.binding.Binding("shift+tab", "prev_pane", "pane", show=False),
            textual.binding.Binding("1", "focus_list", "list", show=False),
            textual.binding.Binding("2", "focus_content", "plots", show=False),
            textual.binding.Binding("3", "focus_pane('log')", "log", show=False),
            # The panels: the two resources a run is made of, opened over this
            # screen and closed with escape.
            textual.binding.Binding("i", "images", "images", show=False),
            textual.binding.Binding("c", "compute", "compute", show=False),
            # Everything below but `z` acts on the selected run, and is in the Runs
            # menu with its key, not in the footer.
            textual.binding.Binding("e", "edit_config", "config"),
            textual.binding.Binding("s", "start_run", "start", show=False),
            textual.binding.Binding("S", "stop_run", "stop", show=False),
            textual.binding.Binding("n", "new_run", "new", show=False),
            textual.binding.Binding("N", "new_sweep_from_run", "new sweep", show=False),
            textual.binding.Binding("d", "delete_run", "delete", show=False),
            # `r` is the base screen's refresh, so restart takes the shifted key --
            # the same convention `S` follows.
            textual.binding.Binding("R", "restart_run", "restart", show=False),
            # `t` for talk: `c` is compute.
            textual.binding.Binding("t", "chat_run", "chat", show=False),
            textual.binding.Binding("m", "toggle_metrics", "metrics", show=False),
            textual.binding.Binding("z", "zoom", "zoom"),
        ]
    )

    # Every pane in reading order -- the tree, then the content column. This
    # is what `tab` walks, skipping whatever is not on screen.
    _PANES = ("runs", "config", "sweep", "plots", "metrics", "log")

    # The right side's panes, of which `apply_focus` shows the ones that
    # describe what is selected.
    _CONTENT = ("config", "sweep", "plots", "log")

    def __init__(self, host: Host, source: data.Data) -> None:
        super().__init__(host, source)
        self.runs: list[types.RunRow] = []
        self.sweeps: list[types.SweepRow] = []
        self.phases: list[types.PhaseListEntry] = []
        # The tree as last drawn, and which of its rows are open, by key. The
        # cursor is kept as a key too, so that rows appearing above it -- an
        # expand, a new run -- do not move the selection with them.
        self.nodes: list[render.TreeNode] = []
        self.expanded: set[str] = set()
        self.cursor_key: str | None = None
        self.marked: set[str] = set()
        # The phase lists of the open runs, and when they were read, from the
        # last fetch; what `draw_tree` lays out and times phases against.
        self.open_phases: dict[str, list[types.PhaseListEntry]] = {}
        self.now = 0.0
        # The pane that has the focus. Kept rather than read off
        # `self.focused` because a keypress asks about the pane, not about the
        # widget inside it that happens to hold the cursor.
        self.pane = "runs"
        # The content pane given the whole right side by `z`, or None.
        self.zoomed: str | None = None
        self.selected_run: str | None = None
        self.selected_phase: str | None = None
        # A phase picked in the tree, rather than the one the run is on now:
        # the right side stays on it while the run moves on.
        self.pinned_phase = False
        self.selected_sweep: str | None = None
        # A run created this session, waiting for the fetch that will list it.
        # `apply_runs` puts the cursor on it and clears this.
        self._pending_run: str | None = None
        # A phase's row to put the cursor on once it is drawn, by key: the
        # goto line opens the run, and its phases come with the next fetch.
        self._pending_phase: str | None = None
        # The last snapshot, for the read-only config popup.
        self._config: data.Snapshot | None = None
        self.x_axis = render.X_STEP
        self.solo: str | None = None
        # The plots the image asks for on the selected phase, and whether the
        # viewer has taken the dashboard back off it. Both per-phase, and both
        # reset by `reset_phase_view`.
        self.pinned: list[render.PlotKey] = []
        self.overridden = False
        self.log_y = False
        # Settled in `on_mount`, where the console that answers half of it
        # exists. Blocks until then, because they are the safe half.
        self.charset = render.CHARSET_BLOCK
        # Both keyed by phase address, so returning to a phase shows the series
        # already read rather than starting over from an exhausted Tail.
        self.points: dict[str, dict[str, list[metrics.MetricPoint]]] = {}
        self.columns: dict[str, list[str]] = {}
        self._tails: dict[str, metrics.Tail] = {}
        # The metrics file each address's series were read from, so that a
        # restart -- which moves the phase's output to a new attempt directory
        # without moving the address -- is seen and the series started over.
        self._metric_paths: dict[str, pathlib.Path | None] = {}
        self.plots: dict[render.PlotKey, widgets.MetricPlot] = {}
        # What the footer was last built for; see `sync_bindings`.
        self._binding_state: tuple[object, ...] | None = None
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
                tree = widgets.RunsTree(id="runs")
                tree.cursor_type = "row"
                tree.border_title = "1 runs"
                tree.add_columns(*render.TREE_COLUMNS)
                yield tree
            with textual.containers.Vertical(id="content"):
                yield widgets.ConfigPane(id="config")
                sweep = textual.containers.VerticalScroll(id="sweep")
                sweep.border_title = "2 sweep"
                with sweep:
                    yield textual.widgets.Static("", id="sweep-grid")
                with widgets.PlotPane(id="plots"):
                    yield textual.widgets.Static("no metrics yet", id="empty")
                yield widgets.LogTail(id="log")
        # Outside `#main`, because it docks against the screen and takes its
        # width from the content column. Mounted once and hidden rather than
        # built on each open: the list owns the check set and the range anchor,
        # and both have to survive being put away.
        yield widgets.MetricList(id="metrics")

    def on_mount(self) -> None:
        # Before `super()`, which starts the first fetch: a plot mounted by the
        # snapshot that lands from it is given this.
        self.charset = self.host.charset()
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
        if self.timer is not None:
            self.timer.reset()
        addr = self.address()
        # The Tail is read on the loop and handed to the worker rather than
        # looked up there: a thread worker cannot be interrupted, so `exclusive`
        # only stops a new one starting, and two overlapping fetches must not
        # both decide to open a reader for the same phase.
        self._fetching = True
        self.fetch(
            self.selected_run,
            self.selected_phase,
            self.attempt(),
            self._tails.get(addr or ""),
            self.expanded_runs(),
        )

    def expanded_runs(self) -> list[str]:
        """The runs the tree has open, whose phases the fetch must list."""
        prefix = render.run_key("")
        return [key.removeprefix(prefix) for key in self.expanded if key.startswith(prefix)]

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
        if self.timer is not None:
            self.timer.reset()
        self._settle = self.set_timer(_SETTLE_SECONDS, self.refresh_data)

    @textual.work(thread=True, exclusive=True, group="snapshot")
    def fetch(
        self,
        run_id: str | None,
        phase: str | None,
        attempt: int | None,
        tail: metrics.Tail | None,
        expanded: list[str],
    ) -> None:
        # `_fetching` is cleared in a `finally`, so that an error the query
        # layer did not wrap costs one tick rather than every tick after it.
        try:
            try:
                snapshot = self.data.snapshot(run_id, phase, attempt, tail, expanded=expanded)
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
        if self.selected_sweep is not None:
            self.apply_sweep()
        self.apply_focus()
        self.sync_bindings()

    def apply_runs(self, snapshot: data.Snapshot) -> None:
        self.runs = snapshot.runs
        self.sweeps = snapshot.sweeps
        self.marked = set(snapshot.marked)
        self.open_phases = dict(snapshot.expanded)
        self.now = snapshot.now
        # A run just created is selected as soon as it is listed, so its config
        # -- which is the only thing there is to do with it next -- comes up
        # without the viewer having to find it.
        pending = next((r for r in self.runs if r.id == self._pending_run), None)
        if pending is not None:
            if pending.sweep_id is not None:
                # A sweep's run is only in the tree while its sweep is open.
                self.expanded.add(render.sweep_key(pending.sweep_id))
            self.cursor_key = render.run_key(pending.id)
            self._pending_run = None
        self.draw_tree()

    def draw_tree(self) -> None:
        """Lay the tree out again from what the last fetch brought, and keep
        the cursor on the node it was on.

        Called by every fetch, and straight away by an expand or a collapse so
        the list answers the key rather than the next tick. The selected run's
        own phases stand in for an expanded one's that a fetch has not listed
        yet, which is what lets opening the run under the cursor show its
        phases at once.
        """
        phases = dict(self.open_phases)
        if self.selected_run is not None and self.selected_run not in phases:
            phases[self.selected_run] = self.phases
        self.nodes = render.build_tree(self.runs, self.sweeps, self.expanded, phases)
        table = self.table("#runs")
        _fill(table, [render.tree_cells(n, self.marked, self.now) for n in self.nodes])
        if not self.nodes:
            self.cursor_key = None
            return
        if self._pending_phase is not None:
            pending_run = self._pending_phase.removeprefix("phase:").split("/", 1)[0]
            if any(n.key == self._pending_phase for n in self.nodes):
                self.cursor_key = self._pending_phase
                self._pending_phase = None
            elif phases.get(pending_run):
                # Its phases are listed, and it is not one of them.
                name = self._pending_phase.rsplit("/", 1)[-1]
                self.show_error(f"no phase '{name}' in that run", hold=True)
                self._pending_phase = None
        index = self.index_of(self.cursor_key)
        if index is None:
            # The node has gone -- deleted, or folded away under a collapse --
            # so the cursor stays on the row it was on, clamped: the rows below
            # a deleted one have shifted up under it.
            index = min(max(table.cursor_row, 0), len(self.nodes) - 1)
        if table.cursor_row != index:
            table.move_cursor(row=index)
        self.select_node(self.nodes[index])

    def index_of(self, key: str | None) -> int | None:
        """Where a node is in the tree, following a phase up to its run."""
        if key is None:
            return None
        for index, node in enumerate(self.nodes):
            if node.key == key:
                return index
        if key.startswith("phase:"):
            run_id = key.removeprefix("phase:").split("/", 1)[0]
            return self.index_of(render.run_key(run_id))
        return None

    def cursor_node(self) -> render.TreeNode | None:
        for node in self.nodes:
            if node.key == self.cursor_key:
                return node
        return None

    def select_node(self, node: render.TreeNode) -> None:
        """Make the right side about `node`: a sweep, a run, or one phase."""
        self.cursor_key = node.key
        if node.kind == "sweep":
            assert node.sweep is not None
            self.selected_sweep = node.sweep.id
            self.pinned_phase = False
            self.select_run(None)
            self.apply_sweep()
            self.apply_focus()
            return
        self.selected_sweep = None
        run_id = node.run_id
        assert run_id is not None
        if node.kind == "phase":
            assert node.phase is not None
            self.select_run(run_id)
            self.pinned_phase = True
            self.select_phase(node.phase.phase)
        else:
            was_pinned = self.pinned_phase
            self.pinned_phase = False
            self.select_run(run_id)
            if was_pinned and self.phases:
                # Up from one of its phases onto the run itself: back to where
                # the run has got to, now rather than on the next tick.
                current = data.current_phase(self.phases)
                if current is not None:
                    self.select_phase(current)
        self.apply_focus()

    def apply_phases(self, snapshot: data.Snapshot) -> None:
        self.phases = snapshot.phases
        if not snapshot.phases:
            return
        listed = [entry.phase for entry in snapshot.phases]
        if self.pinned_phase and self.selected_phase in listed:
            return
        # Nothing picked in the tree: where the run has got to, so that a run
        # under the cursor shows its live curve and its live log.
        phase = data.current_phase(snapshot.phases)
        if phase is not None:
            self.select_phase(phase)

    def apply_sweep(self) -> None:
        sweep = next((s for s in self.sweeps if s.id == self.selected_sweep), None)
        pane = self.query_one("#sweep", textual.containers.VerticalScroll)
        grid = self.query_one("#sweep-grid", textual.widgets.Static)
        if sweep is None:
            grid.update("")
            return
        members = sorted(
            (r for r in self.runs if r.sweep_id == sweep.id), key=lambda r: r.created_at
        )
        pane.border_title = f"2 {sweep.name} · {sweep.status} · {render.sweep_progress(sweep)}"
        grid.update(render.sweep_grid(sweep, members))

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
        # The metrics file the series were read from can move underneath a held
        # Tail: a restart starts a new attempt directory, and the two-part
        # address this state is keyed by keeps meaning "the latest attempt".
        # Points read from the old file are a previous run of the phase, not a
        # prefix of this one, so the whole series -- columns and picker with
        # them -- starts over. `path_changed` also forces the redraw below,
        # because an attempt that has logged nothing yet arrives as an empty
        # update, which would otherwise leave the previous run's curves up.
        path_changed = snapshot.metrics_path != self._metric_paths.get(addr)
        if path_changed:
            self._metric_paths[addr] = snapshot.metrics_path
            self.points.pop(addr, None)
            self.columns.pop(addr, None)
            metric_list.reset()
        series = self.points.setdefault(addr, {})
        for name, points in snapshot.update.points.items():
            series.setdefault(name, []).extend(points)
        seen = self.columns.setdefault(addr, [])
        seen.extend(c for c in snapshot.update.columns if c not in seen)

        pinned = [] if self.overridden else [(spec.y, spec.x) for spec in snapshot.phase_plots]
        changed = pinned != self.pinned
        self.pinned = pinned
        rechecked = metric_list.sync(seen, self.default_checked())
        if path_changed or rechecked or snapshot.update.points or changed:
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
        # Each content pane names what it is showing, after the number that
        # reaches it: whose config, whose curves, whose log.
        form.border_title = f"2 {render.run_title(snapshot.run)}"
        self.query_one("#plots", widgets.PlotPane).border_title = "2 " + render.phase_title(
            snapshot.phase_label, snapshot.run
        )
        self._config = snapshot

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

    def shown_content(self) -> set[str]:
        """The right-hand panes that describe what the cursor is on."""
        run = self.selected_run_row()
        if self.selected_sweep is not None:
            shown = {"sweep"}
        elif run is None:
            shown = {"plots", "log"}
        elif run.status in ("configuring", "queued"):
            # Not started: there is nothing to plot or log, and the config is
            # what there is to see -- and, while configuring, to change.
            shown = {"config"}
        else:
            shown = {"plots", "log"}
        if self.zoomed in shown:
            shown = {self.zoomed}
        return shown

    def apply_focus(self) -> None:
        """Show the right-hand panes for what is selected, zoomed if asked."""
        shown = self.shown_content()
        for pane in self._CONTENT:
            self.query_one(f"#{pane}").display = pane in shown
        log = self.query_one("#log", widgets.LogTail)
        log.border_title = f"3 log · {self.selected_phase}" if self.selected_phase else "3 log"
        if "plots" not in shown:
            # The picker decides which curves are drawn, so it goes with the
            # plots it is beside.
            self.show_metrics(False)
        if self.pane in self._CONTENT and self.pane not in shown:
            # The pane the focus was in has just been hidden; take the focus to
            # the one that replaced it rather than letting Textual pick.
            self.action_focus_content()

    def _pane_of(self, node: textual.widget.Widget | None) -> str | None:
        """The pane `node` sits in, or None for anything else."""
        for parent in widgets.ancestors(node):
            if parent.id in self._PANES:
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
            self.cursor_key,
            self.zoomed,
            None if run is None else run.status,
            self.export_metric() is not None,
            self.chattable() is None,
            # `t` greys per phase, so moving between a run's phases has to
            # rebuild the footer even though the pane and run have not changed.
            self.chat_phase(),
        )
        if state != self._binding_state:
            self._binding_state = state
            self.refresh_bindings()

    # -- selection --------------------------------------------------------

    def select_run(self, run_id: str | None) -> None:
        """Make `run_id` the run the right side is about; None for no run."""
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
                None if run is None else self.data.described(run.image_id),
                self.selected_phase,
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

    @textual.on(textual.widgets.DataTable.RowHighlighted, "#runs")
    def _row_highlighted(self, event: textual.widgets.DataTable.RowHighlighted) -> None:
        # Moving the cursor is the selection: waiting for enter would only make
        # the panes lag the cursor.
        if _stale(event, self.table("#runs")):
            return
        if 0 <= event.cursor_row < len(self.nodes):
            self.select_node(self.nodes[event.cursor_row])
            self.sync_bindings()

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
            self._metric_paths.pop(oldest, None)

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

        A pane that is not on screen is not focused: the metric picker while
        it is put away, or the two content panes the tab is not on.
        """
        widget = self.query_one(f"#{slot}")
        if widget.display:
            widget.focus()

    def action_focus_content(self) -> None:
        """`2`: the first pane on the right, whichever that is for the selection."""
        shown = self.shown_content()
        for pane in self._CONTENT:
            if pane in shown:
                self.action_focus_pane(pane)
                return

    def metrics_open(self) -> bool:
        return self.query_one("#metrics", widgets.MetricList).display

    def action_toggle_metrics(self) -> None:
        """`m`: put the metric picker over the plots, or take it away.

        Only where there are plots: a run that has not started has none, and
        a sweep row shows its grid.

        It keeps the focus while it is up, because everything it is opened for
        -- checking a curve, soloing one, changing the x axis -- is a keypress
        aimed at it. `tab` still reaches the plots without closing it, so the
        curve a check draws can be looked at with the picker still in hand.
        """
        if self.metrics_open():
            self.show_metrics(False)
            return
        if "plots" not in self.shown_content():
            return
        self.show_metrics(True)

    def show_metrics(self, open_: bool) -> None:
        metric_list = self.query_one("#metrics", widgets.MetricList)
        if metric_list.display == open_:
            return
        metric_list.display = open_
        if open_:
            metric_list.focus()
        elif self.pane == "metrics":
            # Back to the plots the picker was beside, rather than to whatever
            # Textual would pick once the focused widget vanishes.
            self.action_focus_pane("plots")

    def action_focus_list(self) -> None:
        """`1`: the tree."""
        self.action_focus_pane("runs")

    def action_drill_in(self) -> None:
        """`enter` on the tree: open or close the row, or go into a phase.

        A sweep or a run opens in place, showing what is under it; a phase has
        nothing under it, so `enter` there hands over to the plots, which the
        cursor has been feeding all along -- "let me at it".
        """
        node = self.cursor_node()
        if node is None:
            return
        if node.expanded is None:
            self.action_focus_content()
            return
        self.set_expanded(node, not node.expanded)

    def action_expand(self) -> None:
        """`right`: open the row under the cursor."""
        node = self.cursor_node()
        if node is not None and node.expanded is False:
            self.set_expanded(node, True)

    def action_collapse(self) -> None:
        """`left`: close the row, or from inside one, go up to the row it is in."""
        node = self.cursor_node()
        if node is None:
            return
        if node.expanded:
            self.set_expanded(node, False)
            return
        parent = self.parent_of(node)
        if parent is not None:
            self.move_to(parent.key)

    def parent_of(self, node: render.TreeNode) -> render.TreeNode | None:
        index = self.nodes.index(node)
        for candidate in reversed(self.nodes[:index]):
            if candidate.depth < node.depth:
                return candidate
        return None

    def move_to(self, key: str) -> None:
        index = self.index_of(key)
        if index is not None:
            self.table("#runs").move_cursor(row=index)
            self.select_node(self.nodes[index])

    def set_expanded(self, node: render.TreeNode, open_: bool) -> None:
        if open_:
            self.expanded.add(node.key)
        else:
            self.expanded.discard(node.key)
        # Drawn now, from what is known, so the list answers the key; the
        # fetch brings an opened run's phases if they were not already here.
        self.draw_tree()
        self.refresh_data()

    def action_toggle_mark(self) -> None:
        """`space`: mark the run under the cursor for comparison, or unmark it."""
        node = self.cursor_node()
        if node is None or node.run_id is None:
            return
        self.mark(node.run_id)

    @textual.work(thread=True, group="mark")
    def mark(self, run_id: str) -> None:
        try:
            self.data.toggle_mark(run_id)
        except exceptions.UI as e:
            self.host.from_thread(self.held_error, str(e))
            return
        self.host.from_thread(self.marked_changed)

    def marked_changed(self) -> None:
        self.refresh_data()
        refresh = getattr(self.host, "refresh_status", None)
        if callable(refresh):
            refresh()

    def action_zoom(self) -> None:
        """`z`: give the focused pane the whole right side, or give it back."""
        if self.zoomed is not None:
            self.zoomed = None
        elif self.pane in self._CONTENT:
            self.zoomed = self.pane
        elif self.pane == "metrics":
            self.zoomed = "plots"
        else:
            # From the tree: the first pane on the right.
            shown = self.shown_content()
            self.zoomed = next((p for p in self._CONTENT if p in shown), None)
        self.apply_focus()
        self.sync_bindings()

    def action_back(self) -> None:
        """`escape`: out of the editor or the picker, then out of the pane.

        Closing an editor keeps the file's value: a committed field has already
        been written, so there is never anything here to lose.
        """
        if self.query_one("#config", widgets.ConfigPane).close_editor():
            return
        if self.metrics_open():
            self.show_metrics(False)
            return
        if self.zoomed is not None:
            self.zoomed = None
            self.apply_focus()
            self.sync_bindings()
            return
        if self.pane != "runs":
            # Out of the content column and back to the tree.
            self.action_focus_pane("runs")
        # From the tree, nothing: `q` is how the app is left, and an escape
        # that quit would be a surprise.

    def action_images(self) -> None:
        self.host.go("images")

    def action_compute(self) -> None:
        self.host.go("compute")

    def action_next_row(self) -> None:
        self.query_one("#config", widgets.ConfigPane).focus_row(1)

    def action_prev_row(self) -> None:
        self.query_one("#config", widgets.ConfigPane).focus_row(-1)

    def visible_panes(self) -> list[str]:
        """The panes on screen, in reading order."""
        return [pane for pane in self._PANES if self.query_one(f"#{pane}").display]

    def action_next_pane(self) -> None:
        self.step_pane(1)

    def action_prev_pane(self) -> None:
        self.step_pane(-1)

    def step_pane(self, delta: int) -> None:
        """`tab`: the next pane on screen, wrapping; `shift+tab`: the previous.

        One cycle over both columns rather than one per column. `tab` is the
        one key every viewer tries first, and it has to mean "the next thing":
        a cycle that stopped at the column boundary would leave the content
        panes reachable only by their numbers, and the numbers are deliberately
        not in the footer.
        """
        panes = self.visible_panes()
        if not panes:
            return
        # A focus that is not in a pane at all -- nothing focused yet -- starts
        # the cycle rather than stepping from an index it does not have.
        index = (panes.index(self.pane) + delta) % len(panes) if self.pane in panes else 0
        self.action_focus_pane(panes[index])

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
        """`e`: the selected run's config -- edited in place, or looked at.

        A configuring run's config is the pane on the right, and `e` edits the
        field its cursor is on there. Any other run's config can no longer
        change (`runs.write_config` refuses it), so `e` shows it in a
        read-only popup over the screen: a glance, not a place to work, and the
        plots and log keep going behind it.
        """
        run = self.selected_run_row()
        if run is None:
            return
        if run.status != "configuring":
            self.view_config(run)
            return
        form = self.query_one("#config", widgets.ConfigPane)
        if not form.editable:
            # The form is one tick behind a run that has just been selected.
            self.show_error("the config is still loading; try again", hold=True)
            return
        if not form.edit_here():
            self.show_error("this image declares no config", hold=True)

    def view_config(self, run: types.RunRow) -> None:
        snapshot = self._config
        if (
            snapshot is None
            or snapshot.run is None
            or snapshot.run.run.id != run.id
            or snapshot.config_schema is None
        ):
            self.show_error("the config is still loading; try again", hold=True)
            return
        rows = render.config_rows(snapshot.config_schema, snapshot.config, snapshot.phase_order)

        def answered(answer: str | None) -> None:
            if answer == "new_sweep":
                self.host.new_sweep(run.id)

        self.host.ask(ConfigViewScreen(run, render.run_summary(snapshot.run), rows), answered)

    def action_new_sweep_from_run(self) -> None:
        """`N`: a sweep around the selected run -- its image, its config."""
        run = self.selected_run_row()
        if run is not None:
            self.host.new_sweep(run.id)

    def sweep_here(self) -> str | None:
        """The sweep the cursor is on, or the one the selected run belongs to."""
        if self.selected_sweep is not None:
            return self.selected_sweep
        run = self.selected_run_row()
        return run.sweep_id if run is not None else None

    def action_show_in_sweeps(self) -> None:
        sweep_id = self.sweep_here()
        if sweep_id is not None:
            self.host.show_sweep(sweep_id)

    def restore(self, cursor_key: str | None, expanded: list[str]) -> None:
        """Back to where the viewer was last session: the rows open, the cursor."""
        self.expanded |= set(expanded)
        if cursor_key is not None:
            self.cursor_key = cursor_key
        # Not mounted yet, the first fetch lays the tree out from these.
        if self.nodes:
            self.draw_tree()
            self.refresh_data()

    def reveal_run(self, run_id: str, phase: str | None = None) -> None:
        """Put the cursor on a run, or on one of its phases, opening what it is in."""
        if phase is not None:
            self.expanded.add(render.run_key(run_id))
            self._pending_phase = render.phase_key(run_id, phase)
        self.select_when_listed(run_id)

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
            # Edited in place while configuring, shown read-only otherwise:
            # live whenever there is a run.
            return True if self.selected_run_row() is not None else None
        if action == "chat_run" and not self.editing():
            # Offered wherever the focus is, because the run it would talk to
            # is the selected one wherever the focus is. What it talks to
            # follows the level: at the phases level, the phase the cursor is
            # on, whose data dir is a frozen snapshot of the run as it stood
            # when that phase ended; at the runs level, the newest servable one.
            return True if self.chattable() is None else None
        if action in _LIFECYCLE and not self.editing():
            # No pane gate. These act on the selected run, and which run that
            # is does not depend on where the focus happens to be -- so a key
            # that vanished on the way to the plots and came back on the way
            # out would be describing the focus, not the run.
            if action == "new_run":
                # The exception: `n` is about the list, not the selected run.
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
        if action == "toggle_mark":
            node = self.cursor_node()
            return True if node is not None and node.run_id is not None else None
        if action == "new_sweep_from_run":
            return True if self.selected_run_row() is not None else None
        if action == "show_in_sweeps":
            return True if self.sweep_here() is not None else None
        if action == "zoom":
            return True if self.zoomed is not None or self.shown_content() else None
        return super().check_action(action, parameters)

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
        # Only from a phase row: standing on the run, there is no phase being
        # pointed at, and restarting from whichever one it happens to be on is
        # not what `R` over a run means. The prompt names the phase either way.
        from_phase = self.selected_phase if self.pinned_phase else None
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
        `runs.restart_run` resolve the run's frozen image id and call
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

        On a phase row it is that phase; on the run itself no phase is being
        pointed at, so the default stands.
        """
        return self.selected_phase if self.pinned_phase else None

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
        described = self.data.described(run.image_id)
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


def help_sections(
    workspaces: collections.abc.Sequence[commands.Workspace],
) -> list[tuple[str, list[tuple[str, str]]]]:
    """What `?` writes down, built from the same registry the menus are.

    The keys that move between places first, since no menu has them; then
    each workspace's menu, submenus flattened into `Run > Start`. An item with
    no key of its own is listed as reachable from the menu, which it is.
    """
    sections: list[tuple[str, list[tuple[str, str]]]] = [
        ("getting around", list(commands.NAVIGATION)),
        ("panes", list(commands.PANE_KEYS)),
    ]

    def flatten(
        entries: collections.abc.Sequence[commands.Entry], prefix: str
    ) -> list[tuple[str, str]]:
        lines: list[tuple[str, str]] = []
        for entry in entries:
            if isinstance(entry, commands.Submenu):
                lines += flatten(entry.items, f"{prefix}{entry.label} > ")
            elif isinstance(entry, commands.Item) and not entry.action.startswith("app.workspace"):
                lines.append((entry.key or "menu", f"{prefix}{entry.label.rstrip('.')}"))
        return lines

    for ws in workspaces:
        sections.append((f"{ws.title} ({ws.key_display})", flatten(commands.MENUS[ws.name], "")))
    return sections


class HelpScreen(textual.screen.ModalScreen[bool]):
    """The key map, because five navigation keys are not in the footer."""

    BINDINGS = [
        textual.binding.Binding("escape", "cancel", "close"),
        textual.binding.Binding("question_mark", "cancel", "close", show=False),
    ]

    def compose(self) -> textual.app.ComposeResult:
        with textual.containers.VerticalScroll(id="help"):
            for heading, entries in help_sections(menus.available(menus.app_of(self))):
                yield textual.widgets.Static(render.help_heading(heading))
                for keys, what in entries:
                    yield textual.widgets.Static(render.help_line(keys, what))

    def on_mount(self) -> None:
        box = self.query_one("#help", textual.containers.VerticalScroll)
        box.border_title = "keys"
        box.focus()

    def action_cancel(self) -> None:
        self.dismiss(False)


class ConfigViewScreen(textual.screen.ModalScreen[str]):
    """A started run's config, read-only, over the screen it was asked from.

    A glance rather than a place: nothing in it can change any more, and the
    plots and log keep updating behind it until `escape` puts it away. The
    fields a sweep varied for this run are marked, since those are what tell
    it from its siblings.
    """

    BINDINGS = [
        textual.binding.Binding("escape", "close", "close"),
        textual.binding.Binding("e", "close", "close", show=False),
        textual.binding.Binding("N", "new_sweep", "new sweep from this run"),
    ]

    DEFAULT_CSS = """
    ConfigViewScreen {
        align: center middle;
    }
    ConfigViewScreen > ConfigPane {
        width: 70%;
        max-width: 90;
        min-width: 50;
        height: 80%;
        background: $surface;
        border: round $accent;
    }
    """

    def __init__(
        self,
        run: types.RunRow,
        summary: list[tuple[str, str]],
        rows: list[render.ConfigRow],
    ) -> None:
        super().__init__()
        self.run = run
        self.summary = summary
        self.rows = [self._marked(row) for row in rows]

    def _marked(self, row: render.ConfigRow) -> render.ConfigRow:
        if ".".join(row.path) not in self.run.point:
            return row
        field = row.field.model_copy(
            update={"label": f"{row.field.label or row.field.key} ◂ swept"}
        )
        return dataclasses.replace(row, field=field)

    def compose(self) -> textual.app.ComposeResult:
        yield widgets.ConfigPane(id="config-view")

    def on_mount(self) -> None:
        pane = self.query_one("#config-view", widgets.ConfigPane)
        pane.border_title = f"config · {self.run.name} · read-only"
        pane.show(self.run.id, self.summary, self.rows, editable=False)
        pane.focus()

    def action_close(self) -> None:
        self.dismiss(None)

    def action_new_sweep(self) -> None:
        """`N`: the usual next step from a run worth a second look."""
        self.dismiss("new_sweep")


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
                    # Left alone, the `Select` defaults to its first option --
                    # the cpu -- and a run meant for the gpu would train on the
                    # cpu unless the viewer noticed (#44). No extra probing:
                    # the choices are in hand.
                    value=render.default_compute(self.choices.compute),
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
        textual.binding.Binding("escape", "back", "back"),
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


class ImagesScreen(_Popover):
    """Images in the local store, as `utrain image list` shows them.

    And, on `a`, `utrain image add`: this is the list the added image appears
    in, so it is where adding one belongs. A panel over the screen the viewer
    came from rather than a place of its own: the pull it starts is theirs to
    wait for or to walk away from, and the list is here again -- still pulling
    -- whenever they look back.
    """

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
        yield _table("images", "images", render.IMAGE_COLUMNS, drill=False)

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
        # A fetch can land after the panel was closed: the instance is kept
        # (see `UtrainApp.go`), but the pop took its widgets away, and
        # `query_one` on them raises where nothing can catch it. The next
        # `on_mount` fetches again, so the rows are not lost, only deferred.
        if self._closed:
            return
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
        return super().check_action(action, parameters)

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
        if self._closed:
            # The panel was closed with the pull still running: the flag is
            # kept for the next visit, where `on_mount`'s fetch will show the
            # image whenever it lands.
            return
        self.refresh_bindings()
        # Not held: a hold expires, and this has to stay up for as long as the
        # pull does. `apply` leaves it alone while `pulling` is set.
        self.show_message(f"pulling {url}...")

    def pull_finished(self, name: str, error: str) -> None:
        self.pulling = False
        if self._closed:
            return
        self.refresh_bindings()
        if error:
            self.held_error(error)
        else:
            self.held_message(f"added {name}")
        # Straight away rather than on the next tick: the row is what the key
        # was pressed for.
        self.refresh_data()


class ComputeScreen(_Popover):
    """Host CPU and GPUs, as `utrain compute list` shows them."""

    def compose_body(self) -> textual.app.ComposeResult:
        yield _table("compute", "compute", render.COMPUTE_COLUMNS, drill=False)

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
        # See `ImagesScreen.apply`: a fetch can land on a closed panel.
        if self._closed:
            return
        self.clear_error()
        _fill(self.table("#compute"), rows)


class SystemScreen(_Screen):
    """The machine: its compute devices, its images, and the data store.

    What `i` and `c` show as panels, as a workspace of its own, with the
    commands that manage them: adding and deleting images, checking and
    cleaning the store. The panels stay, for a quick look from wherever the
    viewer is; this is where the housekeeping is done.
    """

    WORKSPACE = "system"
    HOME = "#images"

    BINDINGS = commands.footer(
        [
            textual.binding.Binding("1", "focus_pane('compute')", "compute", show=False),
            textual.binding.Binding("2", "focus_pane('images')", "images", show=False),
            textual.binding.Binding("a", "add_image", "add image"),
            textual.binding.Binding("d", "delete_image", "delete image"),
            textual.binding.Binding("G", "gc_store", "clean store"),
        ]
    )

    DEFAULT_CSS = """
    SystemScreen #compute {
        height: auto;
        max-height: 12;
    }
    SystemScreen #images {
        height: 1fr;
    }
    SystemScreen #store {
        height: 3;
        padding: 0 1;
    }
    """

    def __init__(self, host: Host, source: data.Data) -> None:
        super().__init__(host, source)
        self.image_names: list[str] = []
        self.pulling = False

    def compose_body(self) -> textual.app.ComposeResult:
        yield _table("compute", "1 compute", render.SYSTEM_COMPUTE_COLUMNS, drill=False, pane=True)
        yield _table("images", "2 images", render.IMAGE_COLUMNS, drill=False, pane=True)
        store = textual.widgets.Static("", id="store", classes="pane")
        store.border_title = "data store"
        yield store

    def on_mount(self) -> None:
        super().on_mount()
        self.table("#images").focus()

    def refresh_data(self) -> None:
        self.fetch()

    @textual.work(thread=True, exclusive=True, group="system")
    def fetch(self) -> None:
        try:
            snapshot = self.data.system()
        except exceptions.UI as e:
            self.host.from_thread(self.show_error, str(e))
            return
        self.host.from_thread(self.apply, snapshot)

    def apply(self, snapshot: data.SystemSnapshot) -> None:
        if not self.pulling:
            self.clear_error()
            if snapshot.problem:
                self.show_error(snapshot.problem)
        _fill(self.table("#compute"), render.system_compute_rows(snapshot.compute, snapshot.runs))
        self.image_names = [i.name for i in snapshot.images]
        _fill(
            self.table("#images"),
            [[i.name, i.size_str, str(i.run_count)] for i in snapshot.images],
        )
        self.query_one("#store", textual.widgets.Static).update(render.store_line(snapshot.store))

    def action_focus_pane(self, pane: str) -> None:
        self.query_one(f"#{pane}").focus()

    def selected_image(self) -> str | None:
        table = self.table("#images")
        if 0 <= table.cursor_row < len(self.image_names):
            return self.image_names[table.cursor_row]
        return None

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action == "add_image":
            return None if self.pulling else True
        if action == "delete_image":
            return True if self.selected_image() is not None else None
        return super().check_action(action, parameters)

    # -- images -----------------------------------------------------------

    def action_add_image(self) -> None:
        if self.pulling:
            return

        def answered(answer: str | None) -> None:
            if answer is not None:
                self.add(answer)

        self.host.ask(AddImageScreen(), answered)

    @textual.work(thread=True, group="image-add")
    def add(self, url: str) -> None:
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
        self.show_message(f"pulling {url}...")

    def pull_finished(self, name: str, error: str) -> None:
        self.pulling = False
        self.refresh_bindings()
        if error:
            self.held_error(error)
        else:
            self.held_message(f"added {name}")
        self.refresh_data()

    def action_delete_image(self) -> None:
        """`d`: remove the image under the cursor, once. Refused while a run uses it."""
        name = self.selected_image()
        if name is None:
            return

        def answered(answer: bool | None) -> None:
            if answer:
                self.remove_image(name)

        self.host.ask(ConfirmScreen(f"delete image '{name}'?"), answered)

    @textual.work(thread=True, group="image-remove")
    def remove_image(self, name: str) -> None:
        try:
            self.data.remove_image(name)
        except exceptions.UI as e:
            self.host.from_thread(self.held_error, str(e))
            return
        self.host.from_thread(self.held_message, f"deleted {name}")
        self.host.from_thread(self.refresh_data)

    # -- the store --------------------------------------------------------

    def action_check_store(self) -> None:
        self.check_store()

    @textual.work(thread=True, group="store")
    def check_store(self) -> None:
        """Hashes every store file, which takes a while on a big store."""
        self.host.from_thread(self.show_message, "checking the data store...")
        try:
            result = self.data.store_check()
        except exceptions.UI as e:
            self.host.from_thread(self.held_error, str(e))
            return
        verdict = f"{len(result.problems)} problem(s)" if result.problems else "ok"
        message = (
            f"checked {result.data_files} data file(s) against "
            f"{result.store_files} store file(s): {verdict}"
        )
        if result.problems:
            self.host.from_thread(self.held_error, message)
        else:
            self.host.from_thread(self.held_message, message)

    def action_gc_store(self) -> None:
        """`G`: reclaim what no run links to any more, once asked."""

        def answered(answer: bool | None) -> None:
            if answer:
                self.gc()

        self.host.ask(ConfirmScreen("delete the store files no run uses any more?"), answered)

    @textual.work(thread=True, group="store")
    def gc(self) -> None:
        try:
            result = self.data.store_gc()
        except exceptions.UI as e:
            self.host.from_thread(self.held_error, str(e))
            return
        self.host.from_thread(
            self.held_message,
            f"removed {result.removed} file(s), reclaimed "
            f"{render.human_size(result.reclaimed_bytes)}",
        )
        self.host.from_thread(self.refresh_data)


# -- sweeps -----------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class NewSweep:
    """What the new-sweep dialog came back with: everything but the axes."""

    name: str
    image: str
    # The run whose config the points start from, or None for the image's
    # defaults, and its name, for the form's title.
    base: str | None
    base_name: str
    compute: list[str]


class SweepsScreen(_Screen):
    """The Sweeps workspace: the sweeps, and for one, its grid, queue and spec.

    The grid is a table with a cell cursor: a cell is one point of the sweep's
    first two axes, `enter` opens its run in Runs and `space` marks it. The
    queue and the spec are read-only and never scroll -- the queue is one line
    per compute and one for what comes next -- so only the list and the grid
    take the focus.
    """

    WORKSPACE = "sweeps"
    HOME = "#sweeps"

    BINDINGS = commands.footer(
        [
            textual.binding.Binding("1", "focus_pane('sweeps')", "sweeps", show=False),
            textual.binding.Binding("2", "focus_pane('matrix')", "grid", show=False),
            textual.binding.Binding("space", "toggle_mark", "mark"),
            textual.binding.Binding("N", "new_sweep", "new"),
            textual.binding.Binding("s", "start_sweep", "start"),
            textual.binding.Binding("p", "pause_sweep", "pause"),
            textual.binding.Binding("plus", "extend_sweep", "extend", key_display="+", show=False),
            textual.binding.Binding("R", "retry_sweep", "retry"),
            textual.binding.Binding("S", "cancel_sweep", "cancel", show=False),
            textual.binding.Binding("d", "delete_sweep", "delete", show=False),
            textual.binding.Binding("C", "compare_sweep", "compare"),
        ]
    )

    DEFAULT_CSS = """
    SweepsScreen #sweeps {
        width: 42;
        height: 1fr;
    }
    SweepsScreen #sweep-right {
        width: 1fr;
    }
    SweepsScreen #matrix {
        height: 1fr;
        border-subtitle-align: left;
    }
    SweepsScreen #queue {
        height: auto;
        padding: 0 1;
    }
    SweepsScreen #spec {
        height: auto;
        padding: 0 1;
    }
    """

    def __init__(self, host: Host, source: data.Data) -> None:
        super().__init__(host, source)
        self.sweeps: list[types.SweepRow] = []
        self.runs: list[types.RunRow] = []
        self.marked: set[str] = set()
        self.selected_sweep: str | None = None
        # A sweep asked for before it was listed -- one just created, or one
        # another workspace sent the viewer to. `apply` moves the cursor to it.
        self.wanted: str | None = None
        self.matrix: render.SweepMatrix | None = None
        self.values: dict[str, str] = {}
        self.value_label = ""
        # The grid's columns as last drawn: a different sweep has a different
        # shape, and a table's columns are rebuilt rather than updated.
        self._matrix_columns: list[str] = []
        self._matrix_sweep: str | None = None

    def compose_body(self) -> textual.app.ComposeResult:
        with textual.containers.Horizontal(id="main"):
            yield _table("sweeps", "1 sweeps", render.SWEEP_COLUMNS, pane=True)
            with textual.containers.Vertical(id="sweep-right"):
                matrix = widgets.PaneTable(id="matrix", classes="pane")
                matrix.cursor_type = "cell"
                matrix.border_title = "2 grid"
                yield matrix
                queue = textual.widgets.Static("", id="queue", classes="pane")
                queue.border_title = "queue"
                yield queue
                spec = textual.widgets.Static("", id="spec", classes="pane")
                spec.border_title = "spec"
                yield spec

    def on_mount(self) -> None:
        super().on_mount()
        self.table("#sweeps").focus()

    def refresh_data(self) -> None:
        self.fetch()

    @textual.work(thread=True, exclusive=True, group="sweeps")
    def fetch(self) -> None:
        try:
            snapshot = self.data.sweeps_snapshot(self.wanted or self.selected_sweep)
        except exceptions.UI as e:
            self.host.from_thread(self.show_error, str(e))
            return
        self.host.from_thread(self.apply, snapshot)

    def apply(self, snapshot: data.SweepsSnapshot) -> None:
        self.clear_error()
        self.sweeps = snapshot.sweeps
        self.runs = snapshot.runs
        self.marked = set(snapshot.marked)
        # Scored for the sweep that was selected when the fetch started.
        self.values = snapshot.values
        self.value_label = snapshot.value_label
        table = self.table("#sweeps")
        _fill(table, [render.sweep_cells(s) for s in self.sweeps])
        ids = [s.id for s in self.sweeps]
        if self.wanted in ids:
            self.selected_sweep = self.wanted
            self.wanted = None
        if self.selected_sweep not in ids:
            # Gone -- deleted -- or never chosen: whatever row the cursor is on.
            row = min(max(table.cursor_row, 0), len(ids) - 1) if ids else -1
            self.selected_sweep = ids[row] if row >= 0 else None
        if self.selected_sweep is not None:
            index = ids.index(self.selected_sweep)
            if table.cursor_row != index:
                table.move_cursor(row=index)
        if snapshot.scored != self.selected_sweep:
            # Scored for another sweep, or for none: not this one's scores.
            self.values = {}
            self.value_label = ""
            self.refresh_data()
        self.apply_selected()
        self.refresh_bindings()

    def selected(self) -> types.SweepRow | None:
        return next((s for s in self.sweeps if s.id == self.selected_sweep), None)

    def members(self, sweep: types.SweepRow) -> list[types.RunRow]:
        return sorted((r for r in self.runs if r.sweep_id == sweep.id), key=lambda r: r.created_at)

    def apply_selected(self) -> None:
        sweep = self.selected()
        grid = self.table("#matrix")
        queue = self.query_one("#queue", textual.widgets.Static)
        spec = self.query_one("#spec", textual.widgets.Static)
        if sweep is None:
            self.matrix = None
            grid.clear(columns=True)
            self._matrix_columns = []
            grid.border_title = "2 grid"
            grid.border_subtitle = ""
            queue.update(rich.text.Text("no sweeps yet: N makes one", style="dim"))
            spec.update("")
            return
        members = self.members(sweep)
        self.matrix = render.sweep_matrix(sweep, members)
        columns, rows = (
            render.matrix_table(self.matrix, self.values, self.marked)
            if self.matrix is not None
            else ([], [])
        )
        if columns != self._matrix_columns or sweep.id != self._matrix_sweep:
            grid.clear(columns=True)
            grid.add_columns(*columns)
            self._matrix_columns = columns
            self._matrix_sweep = sweep.id
            if len(columns) > 1:
                # The first column is the row labels; the cursor starts on a run.
                grid.cursor_coordinate = textual.coordinate.Coordinate(0, 1)
        _fill(grid, rows)
        title = f"2 {sweep.name} · {sweep.status} · {render.sweep_progress(sweep)}"
        if self.value_label:
            title += f" · cell: {self.value_label}"
        grid.border_title = title
        grid.border_subtitle = render.sweep_legend()
        queue.update(render.sweep_queue(sweep, members, self.runs))
        base = next((r.name for r in self.runs if r.id == sweep.base), None)
        text = rich.text.Text()
        for index, (label, value) in enumerate(render.sweep_spec(sweep, base)):
            if index:
                text.append("\n")
            text.append(f"{label:<10}", style="bold")
            text.append(value)
        spec.update(text)

    @textual.on(textual.widgets.DataTable.RowHighlighted, "#sweeps")
    def _sweep_highlighted(self, event: textual.widgets.DataTable.RowHighlighted) -> None:
        table = self.table("#sweeps")
        if _stale(event, table) or not 0 <= table.cursor_row < len(self.sweeps):
            return
        sweep_id = self.sweeps[table.cursor_row].id
        if sweep_id != self.selected_sweep:
            self.selected_sweep = sweep_id
            # The scores are the previous sweep's until the fetch brings these.
            self.values = {}
            self.value_label = ""
            self.apply_selected()
            self.refresh_bindings()
            self.refresh_data()

    def action_drill_in(self) -> None:
        """`enter`: on a sweep, into its grid; on a cell, its run in Runs."""
        if self.focused is self.table("#matrix"):
            self.action_open_run()
        else:
            self.action_focus_pane("matrix")

    @textual.on(textual.widgets.DataTable.CellHighlighted, "#matrix")
    def _cell_highlighted(self) -> None:
        self.refresh_bindings()

    def select_sweep(self, sweep_id: str) -> None:
        """Put the cursor on a sweep, now if it is listed and when it is if not."""
        self.wanted = sweep_id
        if any(s.id == sweep_id for s in self.sweeps):
            self.selected_sweep = sweep_id
            self.wanted = None
            self.table("#sweeps").move_cursor(row=[s.id for s in self.sweeps].index(sweep_id))
            self.apply_selected()
        self.refresh_data()

    def action_focus_pane(self, pane: str) -> None:
        self.query_one(f"#{pane}").focus()

    def cell_runs(self) -> list[types.RunRow]:
        """The runs in the grid cell under the cursor, when the grid has the focus.

        From the list, the grid's cursor is not what the viewer is pointing at,
        so there are none.
        """
        if self.matrix is None or self.focused is not self.table("#matrix"):
            return []
        coordinate = self.table("#matrix").cursor_coordinate
        row, column = coordinate.row, coordinate.column - 1
        if not 0 <= row < len(self.matrix.members) or column < 0:
            return []
        cells = self.matrix.members[row]
        return cells[column] if column < len(cells) else []

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        sweep = self.selected()
        if action == "new_sweep":
            return True
        if action == "compare_sweep":
            return True if sweep is not None else None
        if action in ("open_run", "toggle_mark"):
            return True if self.cell_runs() else None
        if action == "start_sweep":
            return True if sweep is not None and sweep.status in ("draft", "paused") else None
        if action == "pause_sweep":
            return True if sweep is not None and sweep.status == "running" else None
        if action == "extend_sweep":
            return True if sweep is not None and sweep.status != "cancelled" else None
        if action == "retry_sweep":
            retryable = sweep is not None and sweep.counts.failed + sweep.counts.stopped > 0
            return True if retryable else None
        if action == "cancel_sweep":
            live = sweep is not None and sweep.status in ("draft", "running", "paused")
            return True if live else None
        if action == "delete_sweep":
            # A sweep with runs going is cancelled first, as a running run is
            # stopped first: the one prompt that kills something says so.
            return True if sweep is not None and sweep.counts.running == 0 else None
        return super().check_action(action, parameters)

    # -- what a cell leads to -------------------------------------------

    def action_open_run(self) -> None:
        """`enter` on a cell: that run, in Runs."""
        members = self.cell_runs()
        if members:
            self.host.show_run(members[0].id)

    def action_toggle_mark(self) -> None:
        """`space` on a cell: mark its runs for comparison, or unmark them."""
        members = [r.id for r in self.cell_runs()]
        if members:
            self.mark(members)

    @textual.work(thread=True, group="mark")
    def mark(self, run_ids: list[str]) -> None:
        try:
            tray = self.data.tray()
            if all(r in tray for r in run_ids):
                tray = [r for r in tray if r not in run_ids]
            else:
                tray = tray + [r for r in run_ids if r not in tray]
            self.data.set_tray(tray)
        except exceptions.UI as e:
            self.host.from_thread(self.held_error, str(e))
            return
        self.host.from_thread(self.marked_changed)

    def marked_changed(self) -> None:
        self.refresh_data()
        refresh = getattr(self.host, "refresh_status", None)
        if callable(refresh):
            refresh()

    # -- the sweep's lifecycle ------------------------------------------

    def action_new_sweep(self) -> None:
        self.host.new_sweep(None)

    def action_compare_sweep(self) -> None:
        """`C`: this sweep's runs, in Compare."""
        sweep = self.selected()
        if sweep is not None:
            self.host.compare_sweep(sweep.id)

    def action_start_sweep(self) -> None:
        sweep = self.selected()
        if sweep is not None:
            self.lifecycle(sweep.id, "start", f"started {sweep.name}")

    def action_pause_sweep(self) -> None:
        sweep = self.selected()
        if sweep is not None:
            self.lifecycle(sweep.id, "pause", f"paused {sweep.name}; its running runs carry on")

    def action_retry_sweep(self) -> None:
        sweep = self.selected()
        if sweep is not None:
            self.lifecycle(sweep.id, "retry", f"requeued {sweep.name}'s failed and stopped runs")

    def action_cancel_sweep(self) -> None:
        sweep = self.selected()
        if sweep is None:
            return
        running = sweep.counts.running

        def answered(answer: bool | None) -> None:
            if answer:
                self.lifecycle(sweep.id, "cancel", f"cancelled {sweep.name}")

        question = f"cancel sweep '{sweep.name}'"
        if running:
            question += f", stopping its {running} running run(s)"
        self.host.ask(ConfirmScreen(question + "?"), answered)

    def action_delete_sweep(self) -> None:
        sweep = self.selected()
        if sweep is None:
            return

        def answered(answer: bool | None) -> None:
            if answer:
                self.lifecycle(sweep.id, "delete", f"deleted {sweep.name}")

        self.host.ask(
            ConfirmScreen(
                f"delete sweep '{sweep.name}' and its {sweep.counts.total} run(s), with their data?"
            ),
            answered,
        )

    def action_extend_sweep(self) -> None:
        sweep = self.selected()
        if sweep is None:
            return

        def answered(axes: dict[str, object] | None) -> None:
            if axes:
                self.extend(sweep.id, axes)

        self.host.ask(ExtendSweepScreen(sweep), answered)

    @textual.work(thread=True, group="sweep-lifecycle")
    def lifecycle(self, sweep_id: str, op: str, done: str) -> None:
        """Start, pause, retry, cancel or delete, off the message loop.

        Start and retry check for the GPU toolkit, and cancel and delete stop
        runs, all of which shell out.
        """
        try:
            if op == "start":
                self.data.start_sweep(sweep_id)
            elif op == "pause":
                self.data.pause_sweep(sweep_id)
            elif op == "retry":
                self.data.retry_sweep(sweep_id)
            elif op == "cancel":
                self.data.cancel_sweep(sweep_id)
            else:
                self.data.delete_sweep(sweep_id)
        except exceptions.UI as e:
            self.host.from_thread(self.held_error, str(e))
            return
        self.host.from_thread(self.held_message, done)
        self.host.from_thread(self.refresh_data)

    @textual.work(thread=True, group="sweep-lifecycle")
    def extend(self, sweep_id: str, axes: dict[str, object]) -> None:
        """`sweep_extend` describes the image, which starts a container."""
        try:
            added = self.data.extend_sweep(sweep_id, axes)
        except exceptions.UI as e:
            self.host.from_thread(self.held_error, str(e))
            return
        self.host.from_thread(self.held_message, f"added {added} run(s)")
        self.host.from_thread(self.refresh_data)


class NewSweepScreen(textual.screen.ModalScreen[NewSweep | None]):
    """A new sweep's name, image, base config and computes.

    Everything but the axes, which need the image's config schema -- which is
    why the image comes first, here, and the axes on the next screen. From a
    run (`N` in Runs), the image and base are that run's and are not asked.
    """

    BINDINGS = [
        textual.binding.Binding("escape", "cancel", "cancel"),
    ]

    def __init__(self, choices: data.NewSweepChoices, base: types.RunRow | None = None) -> None:
        super().__init__()
        self.choices = choices
        self.base = base

    def compose(self) -> textual.app.ComposeResult:
        box = textual.containers.Vertical(id="new-sweep")
        box.border_title = "new sweep"
        with box:
            with textual.containers.Horizontal(classes="new-run-row"):
                yield textual.widgets.Label("name")
                yield textual.widgets.Input(id="new-sweep-name", compact=True)
            if self.base is not None:
                with textual.containers.Horizontal(classes="new-run-row"):
                    yield textual.widgets.Label("image")
                    yield textual.widgets.Label(self.base.image)
                with textual.containers.Horizontal(classes="new-run-row"):
                    yield textual.widgets.Label("from")
                    yield textual.widgets.Label(self.base.name)
            else:
                image = self.choices.images[0] if self.choices.images else ""
                with textual.containers.Horizontal(classes="new-run-row"):
                    yield textual.widgets.Label("image")
                    yield textual.widgets.Select[str](
                        [(name, name) for name in self.choices.images],
                        allow_blank=False,
                        compact=True,
                        id="new-sweep-image",
                    )
                with textual.containers.Horizontal(classes="new-run-row"):
                    yield textual.widgets.Label("from")
                    yield textual.widgets.Select[str](
                        self.base_options(image),
                        allow_blank=False,
                        compact=True,
                        id="new-sweep-base",
                    )
            yield textual.widgets.Label("compute (space picks; runs are spread across them)")
            default = render.default_compute(self.choices.compute)
            yield textual.widgets.SelectionList[str](
                *(
                    (label, value, value == default)
                    for label, value in render.compute_options(self.choices.compute)
                ),
                id="new-sweep-compute",
            )
            yield textual.widgets.Static("", id="new-sweep-error")
            with textual.containers.Horizontal(id="new-sweep-buttons"):
                yield textual.widgets.Button(
                    "Next", variant="primary", id="new-sweep-next", compact=True
                )
                yield textual.widgets.Button(
                    "Cancel", variant="default", id="new-sweep-cancel", compact=True
                )

    def base_options(self, image: str) -> list[tuple[str, str]]:
        """The configs a sweep of `image` can start from: its defaults, or a run's.

        The empty value stands for the defaults: a `Select` needs a string.
        """
        options = [("the image's defaults", "")]
        options += [(r.name, r.id) for r in self.choices.runs if r.image == image]
        return options

    def on_mount(self) -> None:
        self.query_one("#new-sweep-name", textual.widgets.Input).focus()

    @textual.on(textual.widgets.Select.Changed, "#new-sweep-image")
    def _image_changed(self) -> None:
        base: textual.widgets.Select[str] = self.query_one(
            "#new-sweep-base", textual.widgets.Select
        )
        base.set_options(self.base_options(self._picked("#new-sweep-image")))

    def _picked(self, identifier: str) -> str:
        picked: textual.widgets.Select[str] = self.query_one(identifier, textual.widgets.Select)
        return str(picked.value)

    def computes(self) -> list[str]:
        chosen: textual.widgets.SelectionList[str] = self.query_one(
            "#new-sweep-compute", textual.widgets.SelectionList
        )
        return list(chosen.selected)

    def _error(self, message: str) -> None:
        self.query_one("#new-sweep-error", textual.widgets.Static).update(message)

    @textual.on(textual.widgets.Input.Submitted, "#new-sweep-name")
    def _submitted(self) -> None:
        self._next()

    @textual.on(textual.widgets.Button.Pressed, "#new-sweep-next")
    def _next(self) -> None:
        name = self.query_one("#new-sweep-name", textual.widgets.Input).value.strip()
        if not name:
            self._error("a sweep needs a name")
            return
        problem = sweepsmod.name_problem(name)
        if problem is not None:
            self._error(problem)
            return
        computes = self.computes()
        if not computes:
            self._error("pick at least one compute")
            return
        if self.base is not None:
            image, base, base_name = self.base.image, self.base.id, self.base.name
        else:
            image = self._picked("#new-sweep-image")
            base = self._picked("#new-sweep-base") or None
            base_name = next((r.name for r in self.choices.runs if r.id == base), "")
        self.dismiss(NewSweep(name, image, base, base_name, computes))

    @textual.on(textual.widgets.Button.Pressed, "#new-sweep-cancel")
    def _cancel(self) -> None:
        self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


# What an axis field takes, shown in it while it is empty.
_AXIS_HINT = "a,b,c · lin:lo:hi:n · log:lo:hi:n · *"


class SweepFormScreen(textual.screen.ModalScreen[dict[str, object] | None]):
    """The axes of a new sweep: the image's config, with a sweep field on each.

    Each field shows the value every point starts from, and beside it what to
    sweep it over -- empty for a field held fixed. The grid's size is worked
    out as the fields are typed in, against the schema, so a value the field
    would refuse is said here rather than by the create. Comes back with the
    raw axes, `sweeps.create_sweep`'s argument.
    """

    BINDINGS = [
        textual.binding.Binding("escape", "cancel", "cancel"),
        textual.binding.Binding("down", "app.focus_next", "next", show=False),
        textual.binding.Binding("up", "app.focus_previous", "previous", show=False),
    ]

    DEFAULT_CSS = """
    SweepFormScreen {
        align: center middle;
    }
    /* As tall as the image's fields, up to most of the screen; past that the
       fields scroll and the preview and buttons stay put under them. */
    #sweep-form {
        width: 90%;
        max-width: 110;
        height: auto;
        max-height: 90%;
        padding: 0 1;
        border: round $accent;
        border-title-align: left;
        background: $surface;
    }
    #sweep-fields {
        height: auto;
        max-height: 30;
    }
    .sweep-section {
        color: $text-muted;
        text-style: bold;
        margin-top: 1;
    }
    .sweep-field {
        height: 1;
    }
    .sweep-field > Label {
        width: 28;
    }
    .sweep-field > .value {
        width: 16;
        color: $text-muted;
    }
    .sweep-field > Input {
        width: 1fr;
        height: 1;
    }
    #sweep-preview {
        height: auto;
        margin-top: 1;
    }
    #sweep-form-error {
        color: $error;
        height: auto;
    }
    #sweep-form-buttons {
        width: 100%;
        height: auto;
        align-horizontal: right;
    }
    #sweep-form-buttons Button {
        margin: 0 1;
        min-width: 0;
        padding: 0 2;
    }
    """

    def __init__(self, spec: NewSweep, form: data.SweepForm) -> None:
        super().__init__()
        self.spec = spec
        self.rows = render.config_rows(form.schema, form.values, form.phase_order)
        self.problems: list[str] = []

    def compose(self) -> textual.app.ComposeResult:
        box = textual.containers.Vertical(id="sweep-form")
        source = f"from {self.spec.base_name}" if self.spec.base else "from its defaults"
        box.border_title = f"new sweep {self.spec.name} · {self.spec.image} {source}"
        with box:
            with textual.containers.VerticalScroll(id="sweep-fields"):
                section = None
                for index, row in enumerate(self.rows):
                    if row.section != section:
                        section = row.section
                        yield textual.widgets.Static(section, classes="sweep-section")
                    with textual.containers.Horizontal(classes="sweep-field"):
                        label = textual.widgets.Label(row.field.label or row.field.key)
                        label.tooltip = row.field.description or None
                        yield label
                        yield textual.widgets.Static(
                            render.format_config_value(row.value), classes="value"
                        )
                        yield textual.widgets.Input(
                            placeholder=_AXIS_HINT, compact=True, id=f"axis-{index}"
                        )
            yield textual.widgets.Static("", id="sweep-preview")
            yield textual.widgets.Static("", id="sweep-form-error")
            with textual.containers.Horizontal(id="sweep-form-buttons"):
                yield textual.widgets.Button(
                    "Create", variant="primary", id="sweep-form-create", compact=True
                )
                yield textual.widgets.Button(
                    "Cancel", variant="default", id="sweep-form-cancel", compact=True
                )

    def on_mount(self) -> None:
        inputs = self.query(textual.widgets.Input)
        if inputs:
            inputs.first().focus()
        self.preview()

    def axes(self) -> tuple[dict[str, object], dict[str, int]]:
        """The raw axes typed so far, and how many values each expands to.

        Problems are kept in `self.problems`, one per field that has one.
        """
        raw_axes: dict[str, object] = {}
        sizes: dict[str, int] = {}
        self.problems = []
        for index, row in enumerate(self.rows):
            text = self.query_one(f"#axis-{index}", textual.widgets.Input).value.strip()
            if not text:
                continue
            path = ".".join(row.path)
            try:
                _, raw = sweepsmod.parse_axis_arg(f"{path}={text}")
                sizes[path] = len(sweepsmod.expand_axis(raw, row.field))
            except exceptions.UI as e:
                self.problems.append(f"{row.field.label or row.field.key}: {e}")
                continue
            raw_axes[path] = raw
        return raw_axes, sizes

    @textual.on(textual.widgets.Input.Changed)
    def preview(self) -> None:
        raw_axes, sizes = self.axes()
        preview = self.query_one("#sweep-preview", textual.widgets.Static)
        error = self.query_one("#sweep-form-error", textual.widgets.Static)
        error.update("\n".join(self.problems))
        if not raw_axes:
            preview.update(rich.text.Text("type values into a field to sweep it", style="dim"))
            return
        total = 1
        for n in sizes.values():
            total *= n
        factors = " × ".join(f"{path.rsplit('.', 1)[-1]} {n}" for path, n in sizes.items())
        computes = self.spec.compute
        spread = ", ".join(
            f"{c} {len(range(i, total, len(computes)))}" for i, c in enumerate(computes)
        )
        preview.update(f"{factors} = {total} run(s) · {spread}")

    @textual.on(textual.widgets.Input.Submitted)
    def _submitted(self) -> None:
        """`enter` in a field moves on to the next, as `down` does."""
        self.focus_next()

    @textual.on(textual.widgets.Button.Pressed, "#sweep-form-create")
    def _create(self) -> None:
        raw_axes, _ = self.axes()
        error = self.query_one("#sweep-form-error", textual.widgets.Static)
        if self.problems:
            error.update("\n".join(self.problems))
            return
        if not raw_axes:
            error.update("sweep at least one field")
            return
        self.dismiss(raw_axes)

    @textual.on(textual.widgets.Button.Pressed, "#sweep-form-cancel")
    def _cancel(self) -> None:
        self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class ExtendSweepScreen(textual.screen.ModalScreen[dict[str, object] | None]):
    """Values to add to a sweep's axes: one field per axis, empty to leave it.

    The values are checked by the extend itself, which has the image's schema;
    this only reads what was typed.
    """

    BINDINGS = [
        textual.binding.Binding("escape", "cancel", "cancel"),
    ]

    DEFAULT_CSS = """
    ExtendSweepScreen {
        align: center middle;
    }
    #extend-sweep {
        width: 80;
        height: auto;
        padding: 0 1;
        border: round $accent;
        border-title-align: left;
        background: $surface;
    }
    .extend-row {
        height: 2;
    }
    .extend-row > Label {
        width: 18;
    }
    .extend-row > Vertical > Input {
        height: 1;
    }
    .extend-row .current {
        color: $text-muted;
    }
    #extend-error {
        color: $error;
        height: auto;
    }
    #extend-buttons {
        width: 100%;
        height: auto;
        align-horizontal: right;
    }
    #extend-buttons Button {
        margin: 0 1;
        min-width: 0;
        padding: 0 2;
    }
    """

    def __init__(self, sweep: types.SweepRow) -> None:
        super().__init__()
        self.sweep = sweep
        self.paths = list(sweep.axes)

    def compose(self) -> textual.app.ComposeResult:
        box = textual.containers.Vertical(id="extend-sweep")
        box.border_title = f"extend {self.sweep.name}"
        with box:
            for index, path in enumerate(self.paths):
                with textual.containers.Horizontal(classes="extend-row"):
                    yield textual.widgets.Label(path.rsplit(".", 1)[-1])
                    with textual.containers.Vertical():
                        yield textual.widgets.Input(
                            placeholder=f"add: {_AXIS_HINT}", compact=True, id=f"extend-{index}"
                        )
                        yield textual.widgets.Static(
                            "now " + render.axis_text(path, self.sweep.axes[path]),
                            classes="current",
                        )
            yield textual.widgets.Static("", id="extend-error")
            with textual.containers.Horizontal(id="extend-buttons"):
                yield textual.widgets.Button(
                    "Extend", variant="primary", id="extend-go", compact=True
                )
                yield textual.widgets.Button(
                    "Cancel", variant="default", id="extend-cancel", compact=True
                )

    def on_mount(self) -> None:
        self.query(textual.widgets.Input).first().focus()

    @textual.on(textual.widgets.Input.Submitted)
    @textual.on(textual.widgets.Button.Pressed, "#extend-go")
    def _extend(self) -> None:
        axes: dict[str, object] = {}
        try:
            for index, path in enumerate(self.paths):
                text = self.query_one(f"#extend-{index}", textual.widgets.Input).value.strip()
                if text:
                    axes[path] = sweepsmod.parse_axis_arg(f"{path}={text}")[1]
        except exceptions.UI as e:
            self.query_one("#extend-error", textual.widgets.Static).update(str(e))
            return
        if not axes:
            self.query_one("#extend-error", textual.widgets.Static).update(
                "add values to at least one axis"
            )
            return
        self.dismiss(axes)

    @textual.on(textual.widgets.Button.Pressed, "#extend-cancel")
    def _cancel(self) -> None:
        self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


# -- comparing --------------------------------------------------------------


class PickScreen(textual.screen.ModalScreen[str | None]):
    """One choice from a short list: a phase, a metric, a sweep, a name.

    Opens on the current choice, and comes back with the picked option's
    value, or None on escape.
    """

    BINDINGS = [
        textual.binding.Binding("escape", "cancel", "cancel"),
    ]

    DEFAULT_CSS = """
    PickScreen {
        align: center middle;
    }
    #pick {
        width: auto;
        min-width: 30;
        max-width: 80;
        height: auto;
        max-height: 80%;
        border: round $accent;
        border-title-align: left;
        background: $surface;
    }
    """

    def __init__(self, title: str, options: list[tuple[str, str]], current: str | None) -> None:
        super().__init__()
        self.title_text = title
        self.options = options
        self.current = current

    def compose(self) -> textual.app.ComposeResult:
        box = textual.containers.Vertical(id="pick")
        box.border_title = self.title_text
        with box:
            yield textual.widgets.OptionList(
                *(
                    textual.widgets.option_list.Option(label, id=value)
                    for label, value in self.options
                ),
                id="pick-options",
            )

    def on_mount(self) -> None:
        options = self.query_one("#pick-options", textual.widgets.OptionList)
        values = [value for _, value in self.options]
        if self.current in values:
            options.highlighted = values.index(self.current)
        options.focus()

    def on_option_list_option_selected(
        self, event: textual.widgets.OptionList.OptionSelected
    ) -> None:
        self.dismiss(event.option.id)

    def action_cancel(self) -> None:
        self.dismiss(None)


class NameScreen(textual.screen.ModalScreen[str | None]):
    """A name for something: a saved comparison. The add-image dialog's shape."""

    BINDINGS = [
        textual.binding.Binding("escape", "cancel", "cancel"),
    ]

    DEFAULT_CSS = """
    NameScreen {
        align: center middle;
    }
    #name-box {
        width: 52;
        height: auto;
        padding: 0 1;
        border: round $accent;
        border-title-align: left;
        background: $surface;
    }
    #name-error {
        color: $error;
        height: auto;
    }
    """

    def __init__(self, title: str, value: str = "") -> None:
        super().__init__()
        self.title_text = title
        self.value = value

    def compose(self) -> textual.app.ComposeResult:
        box = textual.containers.Vertical(id="name-box")
        box.border_title = self.title_text
        with box:
            with textual.containers.Horizontal(classes="new-run-row"):
                yield textual.widgets.Label("name")
                yield textual.widgets.Input(self.value, id="name-field", compact=True)
            yield textual.widgets.Static("", id="name-error")

    def on_mount(self) -> None:
        self.query_one("#name-field", textual.widgets.Input).focus()

    @textual.on(textual.widgets.Input.Submitted, "#name-field")
    def _submitted(self) -> None:
        name = self.query_one("#name-field", textual.widgets.Input).value.strip()
        if not name:
            self.query_one("#name-error", textual.widgets.Static).update("it needs a name")
            return
        self.dismiss(name)

    def action_cancel(self) -> None:
        self.dismiss(None)


class CompareScreen(_Screen):
    """The Compare workspace: a set of runs, through one lens at a time.

    The set is the marked runs, a sweep's, or one saved under a name; the
    title says which, and through what phase, metric and reduction. `[` and
    `]` step through the lenses -- the curves over the table, the response of
    the metric to an axis, the sweep's grid as a heatmap, and the config
    fields that differ -- and `<` `>` sort the table. The table's cursor is
    the run the curves highlight, `enter` opens it in Runs and `space` marks
    it.

    What is on screen is kept: the comparison is saved as it changes, and is
    the one the app opens on next time.
    """

    WORKSPACE = "compare"
    HOME = "#compare-table"

    # Every run's config and phase list, and its curve's new points: heavier
    # than the other workspaces' fetches, and a comparison is read rather
    # than watched tick by tick.
    REFRESH_SECONDS = 3.0

    BINDINGS = commands.footer(
        [
            textual.binding.Binding(
                "left_square_bracket", "prev_lens", "lens", key_display="[", show=False
            ),
            textual.binding.Binding("right_square_bracket", "next_lens", "lens", key_display="]"),
            textual.binding.Binding(
                "less_than_sign", "prev_sort", "sort", key_display="<", show=False
            ),
            textual.binding.Binding("greater_than_sign", "next_sort", "sort", key_display=">"),
            textual.binding.Binding("1", "focus_table", "table", show=False),
            textual.binding.Binding("space", "toggle_mark", "mark"),
            textual.binding.Binding("m", "pick_metric", "metric"),
        ]
    )

    DEFAULT_CSS = """
    CompareScreen #lenses {
        height: 1;
        padding: 0 1;
    }
    CompareScreen #compare-view {
        height: 1fr;
    }
    CompareScreen #compare-legend {
        height: auto;
        padding: 0 1;
    }
    CompareScreen #compare-grid {
        padding: 0 1;
    }
    CompareScreen #compare-table {
        height: auto;
        max-height: 40%;
    }
    CompareScreen #compare-diff {
        height: 1fr;
    }
    CompareScreen #compare-empty {
        padding: 1 2;
        color: $text-muted;
    }
    """

    def __init__(self, host: Host, source: data.Data) -> None:
        super().__init__(host, source)
        # None until the saved one is read, which the first fetch does.
        self.state: comparemod.State | None = None
        self.snapshot: data.CompareSnapshot | None = None
        # Bumped by every change the viewer makes, so that a fetch started
        # before one does not put the old state back when it lands.
        self._generation = 0
        self.table_runs: list[str] = []
        self.cursor_run: str | None = None
        self._table_columns: list[str] = []
        self._diff_columns: list[str] = []

    def compose_body(self) -> textual.app.ComposeResult:
        yield textual.widgets.Static("", id="lenses")
        view = textual.containers.Vertical(id="compare-view", classes="pane")
        view.border_title = "compare"
        with view:
            yield widgets.ComparePlot(id="compare-plot")
            yield textual.widgets.Static("", id="compare-legend")
            with textual.containers.VerticalScroll(id="compare-heatmap"):
                yield textual.widgets.Static("", id="compare-grid")
            yield textual.widgets.Static("", id="compare-empty")
        runs = widgets.PaneTable(id="compare-table", classes="pane")
        runs.cursor_type = "row"
        runs.border_title = "1 runs"
        yield runs
        diff = textual.widgets.DataTable[render.Cell](id="compare-diff", classes="pane")
        diff.cursor_type = "row"
        diff.border_title = "1 config fields that differ"
        yield diff

    def on_mount(self) -> None:
        self.query_one("#compare-plot", widgets.ComparePlot).charset = self.host.charset()
        super().on_mount()
        self.table("#compare-table").focus()

    def refresh_data(self) -> None:
        self.fetch(self.state, self._generation)

    @textual.work(thread=True, exclusive=True, group="compare")
    def fetch(self, state: comparemod.State | None, generation: int) -> None:
        try:
            if state is None:
                state = comparemod.State.from_json(self.data.get_state(tuistate.COMPARE))
            snapshot = self.data.compare(state)
        except exceptions.UI as e:
            self.host.from_thread(self.show_error, str(e))
            return
        self.host.from_thread(self.apply, snapshot, generation)

    def apply(self, snapshot: data.CompareSnapshot, generation: int) -> None:
        self.clear_error()
        self.snapshot = snapshot
        if generation == self._generation:
            self.state = snapshot.state
        elif self.state is not None and self.state.phase != snapshot.state.phase:
            # The viewer changed what the fetch reads while it was reading;
            # what landed is about the old phase, so the next one is fetched.
            self.refresh_data()
        self.draw()
        self.refresh_bindings()

    def change(self, **changes: typing.Any) -> None:
        """Change the comparison: redraw now, keep it, and fetch what it needs."""
        state = self.state if self.state is not None else comparemod.State()
        self.state = dataclasses.replace(state, **changes)
        self._generation += 1
        self.draw()
        self.keep(self.state)
        self.refresh_data()
        self.refresh_bindings()

    @textual.work(thread=True, group="compare-keep")
    def keep(self, state: comparemod.State) -> None:
        try:
            self.data.put_state(tuistate.COMPARE, state.to_json())
        except exceptions.UI as e:
            self.host.from_thread(self.held_error, str(e))

    # -- drawing --------------------------------------------------------

    def reducer(self) -> str:
        state = self.state or comparemod.State()
        return state.reducer or comparemod.auto_reducer(state.metric)

    def axes(self) -> list[str]:
        snapshot = self.snapshot
        if snapshot is None:
            return []
        return comparemod.axes_of(snapshot.sweep, [r.id for r in snapshot.runs], snapshot.configs)

    def colour_axis(self) -> str | None:
        axes = self.axes()
        state = self.state or comparemod.State()
        if state.colour in axes:
            return state.colour
        return axes[0] if axes else None

    def reduced(self) -> dict[str, float]:
        snapshot, state = self.snapshot, self.state
        if snapshot is None or state is None or state.metric is None:
            return {}
        out: dict[str, float] = {}
        for run_id, points in snapshot.points.items():
            value = comparemod.reduce(points.get(state.metric, []), self.reducer())
            if value is not None:
                out[run_id] = value
        return out

    def set_label(self) -> str:
        snapshot, state = self.snapshot, self.state or comparemod.State()
        n = len(snapshot.runs) if snapshot is not None else 0
        if state.name:
            return f"{state.name} ({n})"
        if state.source == "sweep" and snapshot is not None and snapshot.sweep is not None:
            return f"sweep {snapshot.sweep.name} ({n})"
        return f"marked runs ({n})"

    def draw(self) -> None:
        snapshot, state = self.snapshot, self.state
        if snapshot is None or state is None:
            return
        view = self.query_one("#compare-view", textual.containers.Vertical)
        plot = self.query_one("#compare-plot", widgets.ComparePlot)
        legend = self.query_one("#compare-legend", textual.widgets.Static)
        heatmap = self.query_one("#compare-heatmap", textual.containers.VerticalScroll)
        empty = self.query_one("#compare-empty", textual.widgets.Static)
        runs_table = self.table("#compare-table")
        diff = self.table("#compare-diff")

        reducer = self.reducer()
        colour = self.colour_axis()
        title = [f"compare · {self.set_label()}"]
        if state.phase:
            title.append(f"phase {state.phase}")
        if state.metric:
            title.append(f"{state.metric} ({reducer})")
        if colour:
            title.append(f"colour {comparemod.short(colour)}")
        view.border_title = " · ".join(title)
        self.draw_lenses()

        lens = state.lens
        nothing = not snapshot.runs
        view.display = lens != "diff"
        diff.display = lens == "diff" and not nothing
        runs_table.display = lens != "diff" and not nothing
        plot.display = lens in ("curves", "response") and not nothing
        legend.display = plot.display
        heatmap.display = lens == "heatmap" and not nothing
        empty.display = nothing
        if not nothing and self.focused in (None, runs_table, diff):
            # A lens change hides one table and shows the other; the focus
            # follows to the one on screen rather than being left on nothing.
            self.action_focus_table()
        if nothing:
            view.display = True
            empty.update(
                "Nothing to compare yet. Mark runs with space in Runs or Sweeps, "
                "or pick a sweep: Compare menu (F3 F3), Compare a sweep..."
            )
            return

        reduced = self.reduced()
        steps: dict[str, int] = {}
        if state.metric is not None:
            for run_id, points in snapshot.points.items():
                series = points.get(state.metric)
                if series:
                    steps[run_id] = series[-1].step
        axes = self.axes()
        marked = set(snapshot.marked)
        table = comparemod.table(
            snapshot.runs,
            axes,
            snapshot.configs,
            reduced,
            steps,
            snapshot.durations,
            state.metric,
            reducer,
            state.sort,
            marked,
        )
        self.fill_runs(table)
        runs_table.border_title = "1 runs" + (
            f" · sorted by {state.sort}" if state.sort in table.columns else ""
        )
        if lens == "curves":
            self.draw_curves(colour)
        elif lens == "response":
            self.draw_response(axes, reduced, colour)
        elif lens == "heatmap":
            self.draw_heatmap(reduced, table.best)
        else:
            columns, rows = comparemod.diff_table(snapshot.runs, snapshot.configs)
            if columns != self._diff_columns:
                diff.clear(columns=True)
                diff.add_columns(*columns)
                self._diff_columns = columns
            _fill(diff, rows)
            if not rows:
                diff.border_title = "1 config fields that differ: none"
            else:
                diff.border_title = f"1 config fields that differ ({len(rows)})"

    def draw_lenses(self) -> None:
        state = self.state or comparemod.State()
        text = rich.text.Text()
        for lens in comparemod.LENSES:
            if lens == state.lens:
                text.append(f"[{lens}]", style="bold reverse")
            else:
                text.append(f" {lens} ", style="dim")
            text.append("  ")
        best = self.best_text()
        if best:
            text.append("   ")
            text.append(best)
        self.query_one("#lenses", textual.widgets.Static).update(text)

    def best_text(self) -> str:
        snapshot = self.snapshot
        reduced = self.reduced()
        best = comparemod.best_of(reduced, self.reducer())
        if snapshot is None or best is None:
            return ""
        name = next((r.name for r in snapshot.runs if r.id == best), best[:8])
        return f"best {name} · {render.format_value(reduced[best])}"

    def fill_runs(self, table: comparemod.Table) -> None:
        grid = self.table("#compare-table")
        if table.columns != self._table_columns:
            grid.clear(columns=True)
            grid.add_columns(*table.columns)
            self._table_columns = table.columns
        _fill(grid, table.rows)
        self.table_runs = table.run_ids
        # The cursor follows its run through a re-sort.
        if self.cursor_run in table.run_ids:
            row = table.run_ids.index(self.cursor_run)
            if grid.cursor_row != row:
                grid.move_cursor(row=row)
        elif table.run_ids:
            self.cursor_run = table.run_ids[min(max(grid.cursor_row, 0), len(table.run_ids) - 1)]

    def draw_curves(self, colour: str | None) -> None:
        snapshot, state = self.snapshot, self.state
        assert snapshot is not None and state is not None
        plot = self.query_one("#compare-plot", widgets.ComparePlot)
        legend = self.query_one("#compare-legend", textual.widgets.Static)
        by_run, key = comparemod.colours(snapshot.runs, colour, snapshot.configs)
        series: list[tuple[list[float], list[float], str]] = []
        on_top: tuple[list[float], list[float], str] | None = None
        highlighted: str | None = None
        for run in snapshot.runs:
            if state.metric is None:
                break
            built = render.build_plot(state.metric, snapshot.points.get(run.id, {}), render.X_STEP)
            if built is None or not built.xs:
                continue
            if run.id == self.cursor_run:
                on_top = (built.xs, built.ys, comparemod.HIGHLIGHT)
                highlighted = run.name
            else:
                series.append((built.xs, built.ys, by_run[run.id]))
        if on_top is not None:
            series.append(on_top)
        plot.curves = widgets.Curves(
            title=f"{state.metric} · {state.phase}", x_label="step", series=series
        )
        legend.update(comparemod.legend_text(key, highlighted))

    def draw_response(self, axes: list[str], reduced: dict[str, float], colour: str | None) -> None:
        snapshot, state = self.snapshot, self.state
        assert snapshot is not None and state is not None
        plot = self.query_one("#compare-plot", widgets.ComparePlot)
        legend = self.query_one("#compare-legend", textual.widgets.Static)
        x_axis, series = comparemod.response(snapshot.runs, axes, snapshot.configs, reduced, colour)
        if x_axis is None:
            plot.curves = None
            legend.update(
                rich.text.Text("the response needs an axis that is a number", style="dim")
            )
            return
        xs = [x for s in series for x in s.xs]
        plot.curves = widgets.Curves(
            title=f"{state.metric} ({self.reducer()})",
            x_label=comparemod.short(x_axis),
            series=[(s.xs, s.ys, s.colour) for s in series],
            x_log=comparemod.log_scale(xs),
        )
        legend.update(comparemod.legend_text([(s.label, s.colour) for s in series], None))

    def draw_heatmap(self, reduced: dict[str, float], best: str | None) -> None:
        snapshot = self.snapshot
        assert snapshot is not None
        grid = self.query_one("#compare-grid", textual.widgets.Static)
        if snapshot.sweep is None:
            grid.update(
                rich.text.Text(
                    "the heatmap is a sweep's grid: pick a sweep to compare", style="dim"
                )
            )
            return
        values = {
            run_id: render.format_value(value) + (f" {comparemod.BEST}" if run_id == best else "")
            for run_id, value in reduced.items()
        }
        grid.update(render.sweep_grid(snapshot.sweep, snapshot.runs, values))

    @textual.on(textual.widgets.DataTable.RowHighlighted, "#compare-table")
    def _row_highlighted(self, event: textual.widgets.DataTable.RowHighlighted) -> None:
        table = self.table("#compare-table")
        if _stale(event, table) or not 0 <= table.cursor_row < len(self.table_runs):
            return
        run_id = self.table_runs[table.cursor_row]
        if run_id != self.cursor_run:
            self.cursor_run = run_id
            if self.state is not None and self.state.lens == "curves":
                self.draw_curves(self.colour_axis())
            self.refresh_bindings()

    def action_drill_in(self) -> None:
        """`enter` on a run: that run, in Runs."""
        self.action_open_run()

    # -- what can be done ----------------------------------------------

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        snapshot = self.snapshot
        has_runs = snapshot is not None and bool(snapshot.runs)
        answers: dict[str, bool] = {
            "open_run": self.cursor_run is not None and has_runs,
            "toggle_mark": self.cursor_run is not None and has_runs,
            "compare_tray": True,
            "pick_sweep": snapshot is not None and bool(snapshot.sweeps),
            "open_saved": snapshot is not None and bool(snapshot.saved),
            "save_comparison": has_runs,
            "clear_tray": snapshot is not None and bool(snapshot.marked),
            "pick_phase": snapshot is not None and bool(snapshot.phases),
            "pick_metric": snapshot is not None and bool(snapshot.metric_names),
            "pick_reducer": has_runs,
            "pick_colour": bool(self.axes()),
            "next_lens": True,
            "prev_lens": True,
            "next_sort": has_runs,
            "prev_sort": has_runs,
        }
        if action in answers:
            return True if answers[action] else None
        return super().check_action(action, parameters)

    def action_focus_table(self) -> None:
        """`1`: the table on screen -- the runs, or in the diff lens, the diff."""
        lens = self.state.lens if self.state is not None else "curves"
        self.query_one("#compare-diff" if lens == "diff" else "#compare-table").focus()

    def action_back(self) -> None:
        """`escape`: back to the table on screen, whichever lens it is."""
        self.action_focus_table()

    def action_open_run(self) -> None:
        if self.cursor_run is not None:
            self.host.show_run(self.cursor_run)

    def action_toggle_mark(self) -> None:
        if self.cursor_run is not None:
            self.mark(self.cursor_run)

    @textual.work(thread=True, group="mark")
    def mark(self, run_id: str) -> None:
        try:
            self.data.toggle_mark(run_id)
        except exceptions.UI as e:
            self.host.from_thread(self.held_error, str(e))
            return
        self.host.from_thread(self.marked_changed)

    def marked_changed(self) -> None:
        self.refresh_data()
        refresh = getattr(self.host, "refresh_status", None)
        if callable(refresh):
            refresh()

    def action_compare_tray(self) -> None:
        self.change(source="tray", sweep=None, runs=(), name=None, sort=None, colour=None)

    def show_sweep(self, sweep_id: str) -> None:
        """Compare a sweep's runs: from the Sweeps workspace, or the picker."""
        self.change(source="sweep", sweep=sweep_id, runs=(), name=None, sort=None, colour=None)

    def pick(
        self,
        title: str,
        options: list[tuple[str, str]],
        current: str | None,
        then: collections.abc.Callable[[str], None],
    ) -> None:
        def answered(answer: str | None) -> None:
            if answer is not None:
                then(answer)

        self.host.ask(PickScreen(title, options, current), answered)

    def action_pick_sweep(self) -> None:
        if self.snapshot is None:
            return
        self.pick(
            "compare a sweep",
            [
                (f"{s.name}  ({s.counts.total} runs, {s.status})", s.id)
                for s in self.snapshot.sweeps
            ],
            self.state.sweep if self.state else None,
            self.show_sweep,
        )

    def action_pick_phase(self) -> None:
        if self.snapshot is None:
            return
        self.pick(
            "phase",
            [(p, p) for p in self.snapshot.phases],
            self.state.phase if self.state else None,
            lambda phase: self.change(phase=phase, metric=None),
        )

    def action_pick_metric(self) -> None:
        if self.snapshot is None:
            return
        self.pick(
            "metric",
            [(m, m) for m in self.snapshot.metric_names],
            self.state.metric if self.state else None,
            lambda metric: self.change(metric=metric),
        )

    def action_pick_reducer(self) -> None:
        auto = comparemod.auto_reducer(self.state.metric if self.state else None)
        options = [(f"automatic ({auto})", "")] + [
            (f"{r}: {what}", r)
            for r, what in (
                ("min", "the lowest value"),
                ("max", "the highest value"),
                ("last", "where it ended"),
            )
        ]
        self.pick(
            "reduce each curve to",
            options,
            (self.state.reducer or "") if self.state else "",
            lambda reducer: self.change(reducer=reducer or None),
        )

    def action_pick_colour(self) -> None:
        self.pick(
            "colour by",
            [(path, path) for path in self.axes()],
            self.colour_axis(),
            lambda path: self.change(colour=path),
        )

    def step_lens(self, delta: int) -> None:
        state = self.state or comparemod.State()
        lenses = comparemod.LENSES
        index = lenses.index(state.lens) if state.lens in lenses else 0
        self.change(lens=lenses[(index + delta) % len(lenses)])

    def action_next_lens(self) -> None:
        self.step_lens(1)

    def action_prev_lens(self) -> None:
        self.step_lens(-1)

    def step_sort(self, delta: int) -> None:
        if not self._table_columns:
            return
        state = self.state or comparemod.State()
        default = comparemod.metric_column(state.metric, self.reducer())
        self.change(sort=comparemod.next_sort(self._table_columns, state.sort, delta, default))

    def action_next_sort(self) -> None:
        self.step_sort(1)

    def action_prev_sort(self) -> None:
        self.step_sort(-1)

    def action_clear_tray(self) -> None:
        self.clear_tray()

    @textual.work(thread=True, group="mark")
    def clear_tray(self) -> None:
        try:
            self.data.set_tray([])
        except exceptions.UI as e:
            self.host.from_thread(self.held_error, str(e))
            return
        self.host.from_thread(self.held_message, "cleared the marks")
        self.host.from_thread(self.marked_changed)

    def action_save_comparison(self) -> None:
        state, snapshot = self.state, self.snapshot
        if state is None or snapshot is None:
            return

        def answered(name: str | None) -> None:
            if not name:
                return
            saved = dataclasses.replace(state, name=name)
            if saved.source == "tray":
                # A tray changes; what was saved is these runs.
                saved = dataclasses.replace(
                    saved, source="runs", runs=tuple(r.id for r in snapshot.runs)
                )
            self.save(name, saved)

        self.host.ask(NameScreen("save comparison as", state.name or ""), answered)

    @textual.work(thread=True, group="compare-keep")
    def save(self, name: str, state: comparemod.State) -> None:
        try:
            self.data.save_comparison(name, state.to_json())
        except exceptions.UI as e:
            self.host.from_thread(self.held_error, str(e))
            return
        self.host.from_thread(self.saved, name, state)

    def saved(self, name: str, state: comparemod.State) -> None:
        self.held_message(f"saved as {name}")
        self.change(**{f.name: getattr(state, f.name) for f in dataclasses.fields(state)})

    def action_open_saved(self) -> None:
        if self.snapshot is None:
            return

        def picked(name: str) -> None:
            self.open_saved(name)

        self.pick(
            "open a saved comparison",
            [(n, n) for n in self.snapshot.saved],
            self.state.name if self.state else None,
            picked,
        )

    @textual.work(thread=True, group="compare-keep")
    def open_saved(self, name: str) -> None:
        try:
            value = self.data.saved_comparison(name)
        except exceptions.UI as e:
            self.host.from_thread(self.held_error, str(e))
            return
        state = dataclasses.replace(comparemod.State.from_json(value), name=name)
        self.host.from_thread(
            self.change, **{f.name: getattr(state, f.name) for f in dataclasses.fields(state)}
        )


class GotoScreen(textual.screen.ModalScreen[str | None]):
    """The `:` line: where to go, typed.

    A run by name or id prefix, a phase as `run/phase`, a sweep as `@name`,
    a workspace by name, or `menu` for the menu of the one in front -- the
    way to everything the function keys reach, for a terminal that does not
    send them. `right` accepts the completion shown in grey.
    """

    BINDINGS = [
        textual.binding.Binding("escape", "cancel", "cancel"),
    ]

    DEFAULT_CSS = """
    GotoScreen {
        align: left bottom;
        background: transparent;
    }
    #goto {
        width: 100%;
        height: auto;
        background: $surface;
        border-top: solid $accent;
        padding: 0 1;
    }
    #goto-help {
        color: $text-muted;
        height: 1;
    }
    #goto-line {
        height: 1;
    }
    #goto-line > Label {
        width: 2;
        color: $accent;
        text-style: bold;
    }
    #goto-line > Input {
        width: 1fr;
        height: 1;
    }
    """

    # The words that are not names: the workspaces, and the menu.
    WORDS = (*(ws.name for ws in commands.WORKSPACES), "menu", "quit")

    def __init__(self, names: list[str]) -> None:
        super().__init__()
        self.names = names

    def compose(self) -> textual.app.ComposeResult:
        with textual.containers.Vertical(id="goto"):
            yield textual.widgets.Static(
                "a run, run/phase, @sweep, a workspace, or menu"
                "  ·  right completes, enter goes, escape closes",
                id="goto-help",
            )
            with textual.containers.Horizontal(id="goto-line"):
                yield textual.widgets.Label(":")
                yield textual.widgets.Input(
                    compact=True,
                    id="goto-field",
                    suggester=textual.suggester.SuggestFromList(
                        [*self.WORDS, *self.names], case_sensitive=False
                    ),
                )

    def on_mount(self) -> None:
        self.query_one("#goto-field", textual.widgets.Input).focus()

    @textual.on(textual.widgets.Input.Submitted, "#goto-field")
    def _submitted(self, event: textual.widgets.Input.Submitted) -> None:
        self.dismiss(event.value.strip() or None)

    def action_cancel(self) -> None:
        self.dismiss(None)
