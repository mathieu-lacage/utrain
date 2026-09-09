"""The `utrain tui` application.

The app owns navigation and the one thing screens cannot do for themselves:
marshalling a worker thread's result back onto the message loop. Screens ask for
both through the `screens.Host` protocol, which this implements.
"""

import collections.abc
import os
import typing

import textual.app
import textual.binding
import textual.screen

from . import data, render, screens, widgets


class _Modes(typing.Protocol):
    """`App.add_mode`, with the screen its factory returns spelled out.

    Textual types it against a bare `Screen`, whose result type is
    unparameterised; every screen here is a `Screen[None]`. Saying so here is
    what lets the call site stay typed.
    """

    def add_mode(
        self,
        mode: str,
        base_screen: collections.abc.Callable[[], textual.screen.Screen[None]],
    ) -> None: ...


class UtrainApp(textual.app.App[None]):
    """Browses runs, phases and their metrics."""

    TITLE = "utrain"

    # Two columns: a fixed-width sidebar of context lists, and the content the
    # selected one is about. The sidebar is sized in cells rather than in `fr`
    # because its widest column -- a run name -- does not get more readable with
    # a wider terminal, whereas a plot does.
    CSS = """
    #main {
        height: 1fr;
    }
    #sidebar {
        width: 42;
    }
    #content {
        width: 1fr;
    }
    #runs, #phases {
        height: 1fr;
        border: round $panel;
        border-title-align: left;
    }
    /* The metric picker is a drawer rather than a fourth list in the sidebar.
       It is a control for the plots -- it decides which curves are drawn, not
       what the screen is about -- so it is there while it is being used and
       gone the rest of the time. Each metric's latest value rides in its
       plot's title, which is what lets it be transient at all.

       Docked rather than laid over the plots: it takes its width from them, so
       the curve a viewer has just checked stays whole while they are still
       standing in the list that checked it. */
    #metrics {
        display: none;
        dock: right;
        /* Clears the header above and the status line and footer below. */
        margin: 1 0 2 0;
        width: 34;
        border: round $accent;
        border-title-align: left;
        background: $surface;
    }
    #log {
        height: 1fr;
    }
    /* The tab strip: one line above the content panes, naming which of the
       three is up and the number that gets to the others. */
    #tabs {
        height: 1;
        padding: 0 1;
    }
    /* The destinations, one line under the header on every one of them. */
    #destinations {
        height: 1;
        padding: 0 1;
    }
    #sidebar > *:focus, #content > *:focus, #content > *:focus-within {
        border: round $accent;
    }
    #empty {
        padding: 1 2;
        color: $text-muted;
    }
    #error {
        color: $error;
        height: auto;
        padding: 0 1;
    }
    /* The same line, for a message that is not an error -- "wrote loss.png".
       See `_Screen.show_message`. */
    #error.-ok {
        color: $success;
    }
    HelpScreen {
        align: center middle;
    }
    #help {
        width: 78;
        height: auto;
        max-height: 90%;
        padding: 0 2 1 2;
        border: round $accent;
        border-title-align: left;
        background: $surface;
    }
    ConfirmScreen {
        align: center middle;
    }
    #confirm {
        width: auto;
        height: auto;
        padding: 1 2;
        border: round $accent;
        background: $surface;
    }
    /* Without an explicit width the dialog comes up empty: a container sized
       `auto` takes its width from its children, and a `Static` -- or a
       `Horizontal` -- is `1fr` by default, which takes its width from the
       container. The two resolve each other to nothing, border and all. */
    #confirm-question {
        width: auto;
        padding-bottom: 1;
    }
    #confirm-buttons {
        width: auto;
        height: auto;
        align-horizontal: center;
    }
    /* `compact` drops the button's borders; the width is a separate default
       (`min-width: 16`), which on a two-word dialog is mostly empty padding. */
    #confirm-buttons Button {
        margin: 0 1;
        min-width: 0;
        padding: 0 2;
    }
    NewRunScreen {
        align: center middle;
    }
    /* Unlike the confirm dialog, this one is sized rather than shrunk to fit:
       an `Input` and two `Select`s have no natural width, and a dialog whose
       width followed the longest image name would move under the viewer. */
    #new-run {
        width: 52;
        height: auto;
        padding: 0 1;
        border: round $accent;
        border-title-align: left;
        background: $surface;
    }
    /* A label and a field on one line, the shape `FieldRow` gives the config
       form -- so the dialog looks like the form the viewer lands in next. The
       compact `Input` and `Select` are one cell tall and borderless, which is
       what lets the row be a single line. */
    .new-run-row {
        height: 1;
    }
    .new-run-row > Label {
        width: 10;
        color: $text-muted;
    }
    .new-run-row > Input, .new-run-row > Select {
        width: 1fr;
        height: 1;
    }
    /* Empty, this is still a line, which is the gap between the fields and the
       buttons -- so a message appears in space the dialog already had and the
       buttons do not jump out from under the cursor. */
    #new-run-error {
        color: $error;
        height: auto;
    }
    #new-run-buttons {
        width: 100%;
        height: auto;
        align-horizontal: right;
    }
    #new-run-buttons Button {
        margin: 0 1;
        min-width: 0;
        padding: 0 2;
    }
    ExportScreen {
        align: center middle;
    }
    /* Wider than the new-run dialog: its longest field is an image name, and
       this one's is a path, which a viewer may well have typed a directory
       into. It reuses `.new-run-row` for the label-and-field lines, because
       the two dialogs should not be subtly different shapes. */
    #export {
        width: 66;
        height: auto;
        padding: 0 1;
        border: round $accent;
        border-title-align: left;
        background: $surface;
    }
    #export-error {
        color: $error;
        height: auto;
    }
    #export-buttons {
        width: 100%;
        height: auto;
        align-horizontal: right;
    }
    #export-buttons Button {
        margin: 0 1;
        min-width: 0;
        padding: 0 2;
    }
    AddImageScreen {
        align: center middle;
    }
    /* As wide as the export dialog and for the same reason: its one field is a
       URL, and a registry ref with a tag on it is long. It reuses
       `.new-run-row` for the label-and-field line, so the three dialogs are
       not subtly different shapes. */
    #add-image {
        width: 66;
        height: auto;
        padding: 0 1;
        border: round $accent;
        border-title-align: left;
        background: $surface;
    }
    #add-image-error {
        color: $error;
        height: auto;
    }
    #add-image-buttons {
        width: 100%;
        height: auto;
        align-horizontal: right;
    }
    #add-image-buttons Button {
        margin: 0 1;
        min-width: 0;
        padding: 0 2;
    }
    /* The transcript takes the screen and the prompt is one line under it,
       the shape a terminal chat already has. Borderless and compact, so the
       line the viewer types on is a line and not a box. */
    #chat-input {
        height: 1;
        border: none;
        padding: 0 1;
    }
    """

    BINDINGS = [
        textual.binding.Binding("q", "quit", "quit"),
    ]

    def __init__(self, source: data.Data | None = None) -> None:
        super().__init__()
        self.data = source if source is not None else data.Data()

    def on_mount(self) -> None:
        # A mode per destination rather than a screen pushed over the last one.
        # Each keeps its own screen and its own stack, so going to look at an
        # image and coming back lands on the run that was selected -- and the
        # metric readers that run had open are still open.
        modes = typing.cast(_Modes, self)
        modes.add_mode("runs", lambda: screens.MainScreen(self, self.data))
        modes.add_mode("images", lambda: screens.ImagesScreen(self, self.data))
        modes.add_mode("compute", lambda: screens.ComputeScreen(self, self.data))
        self.switch_mode("runs")

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """`q` is off while a config field's editor or a confirm prompt is open.

        An `Input` swallows the key, but the option list a `Select` drops does
        not, so without this a `q` aimed at an enum would quit the app instead
        of picking an option. The screen switches off its own keys the same way.
        """
        if action != "quit":
            return True
        # App bindings are global, so `q` at a dialog would quit instead of
        # being read as "not n" -- or, in the new-run dialog, as a letter of the
        # name being typed.
        if isinstance(self.screen, textual.screen.ModalScreen):
            return False
        # `q` at a chat prompt is a letter of what is being said, and there is
        # no modal and no config field to notice: the whole screen is an
        # editor. Escape is how it is left.
        if isinstance(self.screen, screens.ChatScreen):
            return False
        return not widgets.in_config_field(self.focused)

    # -- screens.Host -----------------------------------------------------

    def open(self, screen: textual.screen.Screen[None]) -> None:
        self.push_screen(screen)

    def ask(
        self,
        screen: textual.screen.ModalScreen[screens.Answer],
        answer: collections.abc.Callable[[screens.Answer | None], None],
    ) -> None:
        self.push_screen(screen, answer)

    def charset(self) -> str:
        return render.default_charset(
            self.data.settings.tui_charset,
            self.console.encoding,
            os.environ.get("TERM", ""),
            self.console.legacy_windows,
        )

    def close(self) -> None:
        self.pop_screen()

    def go(self, destination: str) -> None:
        self.switch_mode(destination)

    def from_thread(
        self,
        callback: collections.abc.Callable[..., None],
        *args: object,
    ) -> None:
        self.call_from_thread(callback, *args)


def run(source: data.Data | None = None) -> None:
    UtrainApp(source).run()
