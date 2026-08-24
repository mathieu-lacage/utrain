"""Widgets specific to the dashboard: plots, the metrics pane, the config list.

Each pane carries the keys that act on it, so a shortcut is live only while its
pane has focus and the footer lists exactly what the focused pane can do.
Textual consults the focused widget's bindings before the screen's, which is
what makes that work; the actions themselves stay on `MainScreen`, named through
the `screen.` prefix, because they are about the view as a whole rather than
about one widget.
"""

import collections.abc
import dataclasses

import rich.text
import textual.app
import textual.binding
import textual.containers
import textual.css.query
import textual.message
import textual.reactive
import textual.widget
import textual.widgets
import textual.widgets.select
import uniplot

from .. import container as containermod
from . import render

# The class every in-place editor carries, so a row can find the one it opened
# without caring which of the two widgets it is.
_EDITOR = "editor"


def ancestors(
    node: textual.widget.Widget | None,
) -> collections.abc.Iterator[textual.widget.Widget]:
    """`node` and every widget it sits in, innermost first."""
    while node is not None:
        yield node
        node = node.parent if isinstance(node.parent, textual.widget.Widget) else None


def in_config_field(node: textual.widget.Widget | None) -> bool:
    """True while `node` is the widget a config value is being typed into.

    Strictly inside a row, which is what tells an open editor -- or the option
    list a `Select` drops -- from the row itself holding the cursor. Both the
    screen and the app ask, and neither can tell from the focused widget's
    class: an `Input` in a config row and an `Input` anywhere else are the same
    widget, so the answer is in where it sits.
    """
    return any(isinstance(parent, FieldRow) for parent in ancestors(node) if parent is not node)


class PaneTable(textual.widgets.DataTable[render.Cell]):
    """A sidebar list, with `enter` meaning "show me this one".

    `DataTable` spends `enter` on `select_cursor`, whose `RowSelected` message
    nothing here listens for -- moving the cursor is already the selection. So
    the key is free, and taking it is what lets the whole app be walked with
    the arrows, `enter` and `escape` rather than with the pane numbers.
    """

    BINDINGS = [
        textual.binding.Binding("enter", "screen.drill_in", "open"),
    ]


class MetricPlot(textual.widgets.Static):
    """One metric drawn in the terminal, sized to whatever box it is given.

    `uniplot.plot_gen` is a Rich renderable -- it implements `__rich_console__`
    and `__rich_measure__` for exactly this -- and Textual renders Rich, so the
    plot goes in as an object rather than as a pre-rendered string. That is what
    keeps the ANSI colours intact and lets Rich do the measuring.

    The plot is rebuilt on resize because uniplot rasterises to a fixed
    character grid: `width` and `height` are part of the plot, not of the
    viewport it is shown in.
    """

    DEFAULT_CSS = """
    MetricPlot {
        height: auto;
        min-height: 12;
        padding: 0 1;
    }
    """

    # Leaves room for uniplot's own y-axis labels and border within the
    # widget's content box.
    _Y_LABEL_WIDTH = 12
    _FRAME_HEIGHT = 4

    plot: textual.reactive.reactive[render.Plot | None] = textual.reactive.reactive(
        None, layout=True
    )
    log_y: textual.reactive.reactive[bool] = textual.reactive.reactive(False)
    # uniplot's name for how a point is drawn: half-blocks pack 2x2 points into
    # a cell, braille dots 2x4, so braille is finer where the font has it and
    # blocks are safer where it does not.
    charset: textual.reactive.reactive[str] = textual.reactive.reactive(render.CHARSET_BLOCK)

    def __init__(self, plot: render.Plot | None = None, id: str | None = None) -> None:
        super().__init__(id=id)
        self._drawn: tuple[int, int, int, bool, str] | None = None
        self.plot = plot

    def watch_plot(self) -> None:
        self._redraw()

    def watch_log_y(self) -> None:
        self._redraw()

    def watch_charset(self) -> None:
        self._redraw()

    def on_resize(self) -> None:
        self._redraw()

    def _redraw(self) -> None:
        plot = self.plot
        if plot is None:
            self.update("no data")
            return

        width = max(20, self.content_size.width - self._Y_LABEL_WIDTH)
        height = max(5, self.content_size.height - self._FRAME_HEIGHT)

        # Rendering a plot sets the content of a height:auto widget, which
        # resizes it, which arrives back here as a resize. Without this the
        # widget would redraw itself for as long as the app was running.
        drawn = (width, height, id(plot), self.log_y, self.charset)
        if drawn == self._drawn:
            return
        self._drawn = drawn

        # A log axis cannot show a non-positive value, and metrics legitimately
        # hit zero (an mfu of 0.0 on the first step). Falling back rather than
        # letting uniplot drop the points keeps the curve honest.
        log_y = self.log_y and all(y > 0 for y in plot.ys)

        self.update(
            uniplot.plot_gen(
                xs=plot.xs,
                ys=plot.ys,
                lines=True,
                title=f"{plot.title} vs {plot.x_label}",
                width=width,
                height=height,
                y_as_log=log_y,
                character_set=self.charset,
                # uniplot draws a gridline at zero on both axes by default,
                # which for a training curve is never where a gridline helps:
                # x is a step or an elapsed time, so it starts at zero and the
                # line lands on the left border, and a metric that bottoms out
                # at zero -- an mfu on its first step -- puts the other one on
                # the bottom border. Either way it reads as a doubled frame.
                x_gridlines=[],
                y_gridlines=[],
            )
        )


class PlotPane(textual.containers.VerticalScroll):
    """The stack of plots, and the keys for how they are drawn.

    A class of its own rather than a bare `VerticalScroll` in the screen's
    `compose`, because a container cannot carry bindings without one.
    """

    BINDINGS = [
        textual.binding.Binding("l", "screen.log_y", "log y"),
        textual.binding.Binding("b", "screen.charset", "braille"),
        # Shifted, the convention `S` and `R` already follow: `e` is the
        # screen's `edit_config`, and a pane-local `e` would shadow it here and
        # nowhere else. Offered on the metrics pane too -- see `MetricList`.
        textual.binding.Binding("E", "screen.export_plot", "export"),
    ]

    DEFAULT_CSS = """
    PlotPane {
        height: 2fr;
        border: round $panel;
        border-title-align: left;
    }
    """


class MetricList(textual.widgets.OptionList):
    """The metrics a phase has logged, with their latest values.

    The names are not a query of their own: `metrics.Tail` accumulates them as
    it reads, so this pane is fed the same `MetricUpdate` the plots are.

    Which metrics are drawn lives here rather than on the screen, because the
    rule for a newly seen one -- it arrives checked -- is a property of the
    list. Which one is soloed does not: that is the plot's y axis, and the
    screen owns the axes.
    """

    BINDINGS = [
        textual.binding.Binding("space", "screen.toggle_metric", "plot metric"),
        textual.binding.Binding("y", "screen.solo_metric", "y axis"),
        textual.binding.Binding("x", "screen.cycle_x", "x axis"),
        # The same key `PlotPane` offers: which metric is exported is decided by
        # this pane's cursor, so having to cross to the plots to say "write that
        # one" would be a hop with nothing to decide in it.
        textual.binding.Binding("E", "screen.export_plot", "export"),
    ]

    DEFAULT_CSS = """
    MetricList {
        height: 1fr;
        border: round $panel;
        border-title-align: left;
        padding: 0;
    }
    """

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)
        self.border_title = "3 metrics"
        self._columns: list[str] = []
        self._selected: set[str] = set()
        self._rows: list[render.MetricRow] = []

    @property
    def columns(self) -> list[str]:
        return list(self._columns)

    @property
    def selected(self) -> list[str]:
        """Checked metrics, in the order the phase first logged them."""
        return [c for c in self._columns if c in self._selected]

    @property
    def highlighted_name(self) -> str | None:
        """The metric under the cursor, defaulting to the first.

        `highlighted` is None until the viewer has moved the cursor, and a
        keypress before then should still do the obvious thing.
        """
        index = self.highlighted if self.highlighted is not None else 0
        if index >= len(self._columns):
            return None
        return self._columns[index]

    def reset(self) -> None:
        """Forget everything: the viewer has moved to a different phase."""
        self._columns = []
        self._selected = set()
        self._rows = []
        self.clear_options()

    def sync(self, columns: list[str]) -> bool:
        """Adopt any newly seen metric names. True when the list changed.

        New metrics arrive checked: a phase that starts logging a metric
        mid-run should show it without the viewer having to notice and opt in.
        """
        fresh = [c for c in columns if c not in self._columns]
        if not fresh:
            return False
        self._columns.extend(fresh)
        self._selected.update(fresh)
        return True

    def show(self, rows: list[render.MetricRow]) -> None:
        """Redraw, keeping the cursor where the viewer left it.

        A no-op when nothing changed: the pane is rebuilt every refresh, and an
        unchanged rebuild would fight the viewer for the cursor once a second.
        """
        if rows == self._rows:
            return
        self._rows = list(rows)
        index = self.highlighted
        self.clear_options()
        for row in rows:
            self.add_option(self._line(row))
        if rows:
            # `highlighted` is None until the viewer moves the cursor, which
            # leaves the pane with no visible cursor and `enter` -- the list's
            # own binding, which needs one -- doing nothing. The keys that read
            # the cursor already default to the first row, so this only makes
            # that visible.
            self.highlighted = 0 if index is None else min(index, len(rows) - 1)

    def _line(self, row: render.MetricRow) -> rich.text.Text:
        """`mark name value`, with the value pushed to the right margin.

        Built by hand rather than left to a table: an `OptionList` option is one
        renderable, and the pane is narrow enough that the value has to be
        allowed to collide with a long name rather than wrap.
        """
        width = max(20, self.content_size.width - 2)
        name = row.name[: max(4, width - len(row.value) - 3)]
        pad = max(1, width - len(name) - len(row.value) - 2)
        line = rich.text.Text()
        line.append(row.mark, style="bold cyan")
        line.append(f" {name}{' ' * pad}")
        line.append(row.value, style="bold")
        return line

    def toggle_highlighted(self) -> str | None:
        """Draw or stop drawing the metric under the cursor."""
        name = self.highlighted_name
        if name is None:
            return None
        if name in self._selected:
            self._selected.discard(name)
        else:
            self._selected.add(name)
        return name


class LogTail(textual.widgets.RichLog):
    """The tail of a phase's stdout, replaced wholesale on each refresh."""

    DEFAULT_CSS = """
    LogTail {
        height: 1fr;
        border: round $panel;
        border-title-align: left;
    }
    """

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)
        self.border_title = "5 logs"
        self._shown: list[str] = []

    @property
    def shown(self) -> list[str]:
        """What was last written. RichLog exposes no readable line count."""
        return list(self._shown)

    def show(self, lines: list[str]) -> None:
        if lines == self._shown:
            # The log is re-read every refresh; rewriting an unchanged tail
            # would scroll the pane out from under the viewer once a second.
            return
        # Whether to jump to the newest line is decided before the rewrite:
        # a viewer who has scrolled up to read something is not moved, and one
        # sitting at the bottom keeps following.
        following = self.is_vertical_scroll_end
        self._shown = list(lines)
        self.clear()
        for line in lines:
            self.write(line, scroll_end=False)
        if following:
            self.scroll_end(animate=False)


class ChatLog(textual.containers.VerticalScroll):
    """The transcript of a chat: one `Static` per turn, the last one growing.

    Not a `RichLog`, which is what the log pane uses and what a transcript
    superficially resembles. A `RichLog` can only append lines, and a streaming
    reply is the opposite of that -- it is one turn rewritten as each delta
    lands, wrapping over more lines as it goes. A widget per turn makes the
    open one addressable, and leaves the finished ones alone.
    """

    DEFAULT_CSS = """
    ChatLog {
        height: 1fr;
        border: round $panel;
        border-title-align: left;
        padding: 0 1;
    }
    ChatLog > Static {
        margin-bottom: 1;
    }
    ChatLog > .chat-user {
        color: $accent;
    }
    ChatLog > .chat-note {
        color: $text-muted;
    }
    """

    # What each turn is prefixed with, and the class it wears. A prefix rather
    # than a name in the border or a column: the reply wraps, and a hanging
    # indent would put the model's words in two different places depending on
    # how long they were.
    _PREFIX = {"user": "> ", "assistant": "", "note": ""}
    _CLASS = {"user": "chat-user", "assistant": "chat-reply", "note": "chat-note"}

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)
        self.border_title = "chat"
        self._open: textual.widgets.Static | None = None
        self._role = "note"
        # The transcript as text, kept alongside the widgets: a `Static` does
        # not hand back what it was told to display, and both the streaming
        # append and the tests want to read it.
        self._turns: list[str] = []

    @property
    def turns(self) -> list[str]:
        """Every turn as it stands, newest last, prefixes and all."""
        return list(self._turns)

    def start_turn(self, role: str, text: str = "") -> None:
        """Open a turn. Anything appended from here on belongs to it."""
        self._role = role
        line = self._PREFIX.get(role, "") + text
        self._turns.append(line)
        self._open = textual.widgets.Static(line, classes=self._CLASS.get(role, ""))
        self.mount(self._open)
        self.scroll_end(animate=False)

    def append(self, delta: str) -> None:
        """Add to the open turn. Silently ignored if there is none."""
        if self._open is None:
            return
        self._turns[-1] += delta
        self._open.update(self._turns[-1])
        self.scroll_end(animate=False)

    def end_turn(self) -> None:
        self._open = None

    def note(self, text: str) -> None:
        """A line the transcript says for itself -- a status, a log tail."""
        self.start_turn("note", text)
        self.end_turn()

    def clear_all(self) -> None:
        """Forget the conversation, for `/reset`."""
        self.end_turn()
        self._turns.clear()
        self.remove_children()


class FieldRow(textual.containers.Horizontal):
    """A label and a value, as two columns -- and, on `e`, the editor for it.

    What every config field looks like, whatever the run's status, and what the
    run summary above them is made of. A row that stands for a config field
    carries its `render.ConfigRow` and is a cursor stop; the summary rows have
    none and are skipped, because there is nothing to do to a run id but read
    it.

    The editor replaces the value in the row rather than opening over it. Both
    `Input` and `Select` have a compact form -- no border, one cell tall --
    which is exactly the height of the row, so opening one moves nothing: the
    rest of the config stays where it was and stays readable while you type.
    """

    DEFAULT_CSS = """
    FieldRow {
        height: 1;
        padding: 0 1;
    }
    FieldRow > Label {
        width: 28;
    }
    FieldRow > .value {
        width: 1fr;
        color: $text;
    }
    FieldRow:focus {
        background: $accent 25%;
    }
    FieldRow:focus > .value {
        text-style: bold;
    }
    FieldRow > .editor {
        width: 1fr;
        height: 1;
    }
    """

    # Reactive rather than a plain attribute because a value can be set before
    # the row has composed -- `mount` is deferred, and the run summary is
    # updated in the same pass that mounts it.
    value: textual.reactive.reactive[str] = textual.reactive.reactive("")

    def __init__(
        self,
        label: str,
        value: str,
        description: str = "",
        row: render.ConfigRow | None = None,
    ) -> None:
        super().__init__()
        self._label = label
        self._description = description
        self.row = row
        # Per instance rather than per class: the same widget is both a config
        # row and a summary row, and only the first is somewhere the cursor can
        # stand.
        self.can_focus = row is not None
        self.value = value

    @property
    def path(self) -> tuple[str, ...] | None:
        """Where this row's value lives in `config.yaml`, if it is one."""
        return None if self.row is None else self.row.path

    @property
    def label(self) -> str:
        """The left column's text. A `Label`'s content is not readable back."""
        return self._label

    def compose(self) -> textual.app.ComposeResult:
        label = textual.widgets.Label(self._label)
        label.tooltip = self._description or None
        yield label
        yield textual.widgets.Static(self.value, classes="value")

    def watch_value(self, value: str) -> None:
        try:
            self.query_one(".value", textual.widgets.Static).update(value)
        except textual.css.query.NoMatches:
            # Not composed yet; `compose` will read the new value itself.
            pass

    # -- the editor -------------------------------------------------------

    @property
    def editing(self) -> bool:
        return bool(self.query(f".{_EDITOR}"))

    def open_editor(self) -> object | None:
        """Show the editor for this field, or -- for a bool -- flip it.

        A bool has no editor: `Switch` is three cells tall and has no compact
        form, and a value with two states does not need a widget to pick
        between them. The flipped value comes back for the caller to commit,
        where every other type commits later, when the editor says so.
        """
        if self.row is None or self.editing:
            return None
        field = self.row.field
        if field.type == "bool":
            return not bool(self.row.value)
        self.query_one(".value", textual.widgets.Static).display = False
        editor = self._editor(field)
        editor.add_class(_EDITOR)
        self.mount(editor)
        editor.focus()
        if isinstance(editor, textual.widgets.Select):
            # Straight to the options: opening the editor and then having to
            # ask for the list is a keypress that says nothing. After the
            # refresh, because a `Select` that has not composed yet has no
            # overlay to show and would silently stay shut.
            self.call_after_refresh(editor.action_show_overlay)
        return None

    def _editor(self, field: containermod.schema.FieldSchema) -> textual.widget.Widget:
        """The widget for a field, from its declared type.

        The type rather than what the value happens to look like, so an enum is
        a menu even before it has been set.
        """
        value = self.row.value if self.row is not None else None
        if field.type == "enum":
            chosen: str | textual.widgets.select.NoSelection = (
                textual.widgets.Select.NULL if value is None else str(value)
            )
            select: textual.widgets.Select[str] = textual.widgets.Select(
                [(o, o) for o in field.options],
                value=chosen,
                compact=True,
            )
            return select
        return textual.widgets.Input(
            value="" if value is None else str(value),
            placeholder="" if field.default is None else str(field.default),
            compact=True,
        )

    def editor_value(self) -> object:
        """What the viewer has put in, in the shape `runs.write_config` takes.

        Strings are handed over uncoerced: the query layer validates against the
        same schema this was built from, and doing the conversion twice is how
        the two would come to disagree about what "3e-4" means.
        """
        editor = self.query_one(f".{_EDITOR}")
        if isinstance(editor, textual.widgets.Input):
            return editor.value
        select: textual.widgets.Select[str] = self.query_one(textual.widgets.Select)
        return (
            None if isinstance(select.value, textual.widgets.select.NoSelection) else select.value
        )

    def close_editor(self) -> bool:
        """Put the value back and take the focus. False when none was open."""
        editors = list(self.query(f".{_EDITOR}"))
        if not editors:
            return False
        for editor in editors:
            editor.remove()
        self.query_one(".value", textual.widgets.Static).display = True
        self.focus()
        return True

    def set_value(self, value: object) -> None:
        """Show `value` as the read-only row shows it."""
        self.value = render.format_config_value(value)


class ConfigPane(textual.containers.VerticalScroll):
    """A run's config, as a list of values you can put a cursor on.

    Every run shows the same list, whatever its status, because reading a
    config and comparing two of them is what the pane is mostly for. Editing is
    one field at a time and in place: `e` opens an editor on the row under the
    cursor, `enter` commits it, `escape` abandons it, and the pane is a list
    again either way.

    A committed field is written through `runs.write_config` straight away, so
    the pane never holds anything the file does not -- there is nothing to save
    and nothing to discard. While an editor holds the focus the screen's
    single-key shortcuts are switched off (see `MainScreen.check_action`),
    because the option list a `Select` drops does not swallow printable keys
    the way an `Input` does, and `q` would otherwise quit the app out from under
    the edit.

    The view refreshes once a second. The mount is keyed on the run and the
    fields so the rows are built once; their values are updated in place, which
    is what lets a row keep an open editor across a refresh.
    """

    # On the pane rather than on a row, because the focus is on a row or inside
    # its editor and a binding is looked up from the focused widget outwards.
    # `e` is the exception and lives on the screen: it has to work from the runs
    # pane too.
    BINDINGS = [
        # The list's own cursor. These displace `VerticalScroll`'s scrolling,
        # which `pageup`, `pagedown`, `home` and `end` still do -- and focusing
        # a row scrolls it into view anyway.
        textual.binding.Binding("down", "screen.next_row", "next", show=False),
        textual.binding.Binding("up", "screen.prev_row", "previous", show=False),
        textual.binding.Binding("enter", "screen.edit_config", "edit", show=False),
        # An `Input` binds `enter` but not `escape`, so this is reachable from
        # inside an editor while `enter` stays the editor's own.
        textual.binding.Binding("escape", "screen.back", "back", show=False),
    ]

    DEFAULT_CSS = """
    ConfigPane {
        height: 1fr;
        border: round $panel;
        border-title-align: left;
    }
    ConfigPane > Static.section {
        padding: 1 1 0 1;
        text-style: bold;
        color: $accent;
    }
    """

    class Edited(textual.message.Message):
        """One field was given a new value, and it should be written."""

        def __init__(self, path: tuple[str, ...], value: object) -> None:
            super().__init__()
            self.path = path
            self.value = value

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)
        self.border_title = "4 config"
        self._key: tuple[str, tuple[str, ...], tuple[tuple[str, ...], ...]] | None = None
        self._summary: dict[str, FieldRow] = {}
        self._can_edit = False

    @property
    def editable(self) -> bool:
        """The run can still be configured."""
        return self._can_edit

    def show(
        self,
        run_id: str,
        summary: list[tuple[str, str]],
        rows: list[render.ConfigRow],
        editable: bool,
    ) -> None:
        """Show a run's summary and config, mounting only when it has to.

        The view refreshes once a second. Re-mounting on every tick would take
        away whatever editor was open, so the mount is keyed on the run, the
        labels and the fields, and everything that merely changes value -- a
        status, an attempt count, a config value edited from elsewhere -- is
        updated in place. A row with an editor open is left alone: what is in
        the editor is the viewer's, not the file's.
        """
        self._can_edit = editable
        key = (run_id, tuple(label for label, _ in summary), tuple(r.path for r in rows))
        if key != self._key:
            self._key = key
            self._mount(summary, rows)
        for label, value in summary:
            row = self._summary.get(label)
            if row is not None:
                row.value = value
        by_path = {row.path: row for row in self.rows()}
        for config_row in rows:
            row = by_path.get(config_row.path)
            if row is not None and not row.editing:
                row.row = config_row
                row.set_value(config_row.value)

    def _mount(
        self,
        summary: list[tuple[str, str]],
        rows: list[render.ConfigRow],
    ) -> None:
        self.remove_children()
        self._summary = {}
        children: list[textual.widget.Widget] = []
        if summary:
            children.append(textual.widgets.Static("run", classes="section"))
            for label, value in summary:
                row = FieldRow(label, value)
                self._summary[label] = row
                children.append(row)
        if not rows:
            children.append(
                textual.widgets.Static("this image declares no config", classes="section")
            )
            self.mount_all(children)
            return
        section = ""
        for config_row in rows:
            if config_row.section != section:
                section = config_row.section
                children.append(textual.widgets.Static(section, classes="section"))
            field = config_row.field
            children.append(
                FieldRow(
                    field.label or field.key,
                    render.format_config_value(config_row.value),
                    field.description,
                    config_row,
                )
            )
        self.mount_all(children)

    # -- the cursor -------------------------------------------------------

    def rows(self) -> list[FieldRow]:
        """The config rows, top to bottom."""
        return [row for row in self.query(FieldRow) if row.row is not None]

    def cursor(self) -> FieldRow | None:
        """The row the cursor is on, or the one whose editor is open."""
        return next((row for row in self.rows() if row.has_focus_within), None)

    def focus_row(self, delta: int) -> FieldRow | None:
        """Move the cursor `delta` rows, wrapping around. The row it lands on.

        From nowhere -- the pane itself has the focus, which is where `enter`
        from the runs list leaves it -- the first row is `delta` steps in, so
        `down` lands on the top one and `up` on the bottom one.

        The row comes back rather than being read off the focus afterwards:
        `focus` is not answered until the message loop runs, so `has_focus` is
        still the old answer to the caller.
        """
        rows = self.rows()
        if not rows:
            return None
        index = next((i for i, row in enumerate(rows) if row.has_focus), None)
        start = (0 if delta > 0 else -1) if index is None else index + delta
        row = rows[start % len(rows)]
        row.focus()
        return row

    # -- editing one field ------------------------------------------------

    def edit_here(self) -> bool:
        """Edit the field under the cursor, finding it a row if it has none.

        One rule, so `e` always ends with something open: from the runs list,
        where there is no cursor in the pane yet, it edits the first field.

        A bool has no editor: `open_editor` hands back the flipped value, and
        it is committed on the spot.
        """
        if not self._can_edit:
            return False
        row = self.cursor() or self.focus_row(1)
        if row is None:
            return False
        value = row.open_editor()
        if value is not None and row.path is not None:
            self.post_message(self.Edited(row.path, value))
        return True

    def commit(self, *, only_if_changed: bool = False) -> bool:
        """Send what the open editor holds to be written. False when none is.

        `only_if_changed` is for a `Select`, which announces its value as soon
        as it is mounted with one. Nothing was picked there, and committing it
        would shut the menu before it had been looked at.
        """
        row = self.cursor()
        if row is None or not row.editing or row.path is None or row.row is None:
            return False
        value = row.editor_value()
        if only_if_changed and value == row.row.value:
            return False
        self.post_message(self.Edited(row.path, value))
        return True

    def close_editor(self) -> bool:
        """Shut the open editor, keeping the file's value. False when none is."""
        row = self.cursor()
        return row is not None and row.close_editor()

    def applied(self, path: tuple[str, ...], value: object) -> None:
        """Take on a value that has just been written.

        The row carries the value the next edit is built from, so this cannot
        wait for the next refresh: two edits in the same second would send the
        second one alongside a stale copy of the first, undoing it.
        """
        for row in self.rows():
            if row.path == path and row.row is not None:
                row.row = dataclasses.replace(row.row, value=value)
                row.set_value(value)

    def values(self, replacing: tuple[tuple[str, ...], object] | None = None) -> dict[str, object]:
        """The config as the rows have it, with one field replaced.

        Read off the rows' `ConfigRow`s rather than off the screen: what is
        displayed has been through `format_config_value`, and formatting is not
        meant to be read back.
        """
        path, value = replacing if replacing is not None else ((), None)
        return render.config_values(
            [
                (row.row.path, value if row.row.path == path else row.row.value)
                for row in self.rows()
                if row.row is not None
            ]
        )
