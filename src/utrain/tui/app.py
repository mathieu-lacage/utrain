"""The `utrain tui` application.

The app owns navigation and the one thing screens cannot do for themselves:
marshalling a worker thread's result back onto the message loop. Screens ask for
both through the `screens.Host` protocol, which this implements.
"""

import base64
import collections.abc
import os

import textual
import textual.app
import textual.binding
import textual.screen

from .. import exceptions
from . import commands, data, menus, render, screens, widgets


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
    #runs {
        height: 1fr;
        border: round $panel;
        border-title-align: left;
    }
    /* A sweep's status grid, which the right side shows for a sweep row. */
    #sweep {
        height: 1fr;
        border: round $panel;
        border-title-align: left;
        padding: 0 1;
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
    /* Textual draws a selection with a half-alpha primary background and a
       transparent foreground. A terminal does not blend -- the block reads as
       solid -- and a transparent foreground renders as the background it sits
       on, so the selected text vanished into the highlight. A solid primary
       with $text on top stays readable in the truecolor theme, and the ansi
       theme's own pair is dark blue under a light text as well. */
    Screen > .screen--selection {
        background: $primary;
        color: $text;
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
    /* The resource panels -- images, compute -- are modals over whatever the
       viewer was looking at, not destinations: `i` pushes one, `escape`
       closes it, and the screen beneath is suspended rather than replaced, so
       the run being watched is still there when it closes. A surface under
       the panes rather than a box around them: the list is framed and titled
       like every other pane, and a second border would say what the first
       already says.

       Width rather than `auto`, for the trap `#confirm` dodges: the table
       inside is `1fr` wide, and an `auto` container around a `1fr` child
       resolves each other to nothing. Height is `auto`, which works because
       a `DataTable` hugs its rows -- but capped, so a long list scrolls
       inside the table instead of pushing the error line and the footer off
       the bottom of the terminal.

       The dim has to be restated here. ModalScreen would supply a
       translucent background, but Textual gathers DEFAULT_CSS by walking
       __bases__ and following only the first DOMNode base -- which from
       `_Popover` is `_Screen`, taking the chain to Screen and its opaque
       `background: $background`, and never to ModalScreen at all. The
       translucent colour is what tells the compositor to draw the screen
       beneath; without it the panel floats on a blank one. In an ANSI
       terminal, where alpha is not to be had, ModalScreen's answer is no
       background at all and a dim text-style on the suspended screen
       beneath -- which Screen's own DEFAULT_CSS supplies. */
    ImagesScreen, ComputeScreen {
        align: center middle;
        background: $background 60%;
        &:ansi {
            background: transparent;
        }
    }
    #panel {
        width: 70%;
        max-width: 100;
        min-width: 44;
        height: auto;
        max-height: 85%;
        padding: 1;
        background: $surface;
    }
    #panel DataTable {
        max-height: 20;
    }
    #panel > *:focus, #panel > *:focus-within {
        border: round $accent;
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
    /* A menu's box. Here rather than in `MenuScreen`'s own CSS because an
       `OptionList` brings a border of its own, and the app's stylesheet is
       what outranks a widget's defaults. */
    MenuScreen > OptionList {
        height: auto;
        max-height: 90%;
        border: round $accent;
        background: $surface;
        padding: 0;
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
        # Textual's own copy key is ctrl+c (or super+c). ctrl+shift+c is what
        # a desktop terminal uses to copy its own selection, and the fingers
        # bring it here -- where there is no terminal selection to copy, only
        # the one the app made, so it copies that.
        textual.binding.Binding("ctrl+shift+c", "screen.copy_text", show=False),
        # The workspaces, and the menus. No `alt` shortcuts: over ssh an
        # `alt+x` arrives as `escape` then `x`, which is also "back" then a
        # key, split by a timeout that network latency makes unreliable.
        *(
            textual.binding.Binding(ws.key, f"workspace_key('{ws.name}')", ws.title, show=False)
            for ws in commands.WORKSPACES
        ),
        textual.binding.Binding("f10", "menu", "menu", show=False),
    ]

    # The workspace the app opens on, before a saved session says otherwise.
    DEFAULT_MODE = "runs"

    # How often the top row's status is read: the marked-run count and the
    # GPUs' utilisation. `nvidia-smi` forks, so not every second.
    _STATUS_SECONDS = 3.0

    def __init__(self, source: data.Data | None = None) -> None:
        super().__init__()
        self.data = source if source is not None else data.Data()
        # The panels, built once and kept. A screen popped off the stack is
        # unmounted -- its widgets are gone -- but `ImagesScreen.add` starts a
        # pull that outlives the panel being closed, and its callbacks are
        # written on the instance; keeping it here means the pull is not lost
        # with the view, and the next `i` mounts the same screen again. See
        # `_Popover` for the `_closed` latch that makes that safe.
        self._panels: dict[str, textual.screen.Screen[None]] = {}
        # One mode per workspace, each with its own screen stack: a dialog or
        # a chat opened in one stays there while another is looked at.
        self.workspaces: dict[str, collections.abc.Callable[[], textual.screen.Screen[None]]] = {
            "runs": lambda: screens.MainScreen(self, self.data),
            "system": lambda: screens.SystemScreen(self, self.data),
        }
        for name, factory in self.workspaces.items():
            # Textual types a mode's factory as returning `Screen[Unknown]`.
            self.add_mode(name, factory)  # pyright: ignore[reportUnknownMemberType]
        # What the top row's right-hand side says; see `refresh_status`.
        self.status_line = ""

    def on_mount(self) -> None:
        self.set_interval(self._STATUS_SECONDS, self.refresh_status)
        self.refresh_status()

    @textual.work(thread=True, exclusive=True, group="status")
    def refresh_status(self) -> None:
        try:
            line = self.data.status_line()
        except exceptions.UI:
            return
        self.call_from_thread(self.show_status, line)

    def show_status(self, line: str) -> None:
        self.status_line = line
        for bar in self.screen.query(menus.TopBar):
            bar.show_status(line)

    def action_workspace(self, name: str) -> None:
        """Go to a workspace. From a menu's "Go to" item, or its function key."""
        menus.close_menus(self)
        if name in self.workspaces and name != self.current_mode:
            self.switch_mode(name)

    def action_workspace_key(self, name: str) -> None:
        """`F1`-`F4`: go to that workspace; in it already, drop its menu.

        So `F2 F2` reads "go to Sweeps and show me what I can do there".
        """
        if name == self.current_mode and not isinstance(self.screen, menus.MenuScreen):
            menus.open_menu(self, name)
            return
        self.action_workspace(name)

    def action_menu(self) -> None:
        """`F10`: the menu of the workspace in front of the viewer."""
        menus.open_menu(self, self.current_mode)

    def on_top_bar_menu_requested(self, event: menus.TopBar.MenuRequested) -> None:
        menus.open_menu(self, event.workspace)

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """`q` is off while a config field's editor or a confirm prompt is open.

        An `Input` swallows the key, but the option list a `Select` drops does
        not, so without this a `q` aimed at an enum would quit the app instead
        of picking an option. The screen switches off its own keys the same way.

        The function keys are off under a dialog, which has to be answered
        before the viewer goes anywhere, but not under a menu, which they
        close on the way.
        """
        if action in ("workspace_key", "menu"):
            screen = self.screen
            return not isinstance(screen, textual.screen.ModalScreen) or isinstance(
                screen, menus.MenuScreen
            )
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

    def copy_to_clipboard(self, text: str) -> None:
        """Copy the selection to the terminal's clipboard, and its primary too.

        Textual's copy speaks OSC 52 to the terminal, which is the only way a
        TUI running over ssh reaches the machine the terminal runs on. It asks
        for the clipboard target alone, and the middle mouse button pastes the
        primary selection -- a different target -- so the same text is offered
        there as well. Whether either lands is the terminal emulator's call:
        one that honors neither leaves these keys copying into the void, and
        holding shift while dragging is then the way to select text with the
        terminal's own machinery instead.
        """
        super().copy_to_clipboard(text)
        if self._driver is None:
            return
        encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
        self._driver.write(f"\x1b]52;p;{encoded}\a")

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

    def go(self, which: str) -> None:
        """Show a panel -- `images` or `compute` -- over the screen showing.

        The same instance every time, so that a pull in flight survives the
        panel being closed; the remount refetches the list, which is all the
        state the panels have.
        """
        panel = self._panels.get(which)
        if panel is None:
            factory = {"images": screens.ImagesScreen, "compute": screens.ComputeScreen}[which]
            panel = factory(self, self.data)
            self._panels[which] = panel
        self.push_screen(panel)

    def from_thread(
        self,
        callback: collections.abc.Callable[..., None],
        *args: object,
    ) -> None:
        self.call_from_thread(callback, *args)


def run(source: data.Data | None = None) -> None:
    UtrainApp(source).run()
