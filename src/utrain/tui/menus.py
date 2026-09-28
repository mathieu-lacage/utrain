"""The top row and the menus it drops.

The row names the four workspaces, with the one you are in picked out, and
carries the state that matters wherever you are: how many runs are marked for
comparison, and how busy the GPUs are. It is the app's only row of chrome
above the panes, where the old header was.

Each title is also a menu. A menu is a modal screen holding an `OptionList`,
drawn under its title over the workspace -- which does not move, keeps
refreshing underneath, and is not dimmed, so the live state the viewer may be
about to act on stays visible around it. A submenu is a second one, pushed
beside the item that opened it. Textual has no menu widget; this is the whole
of one.

What the items are is `commands.MENUS`. Whether one is live is the screen's own
`check_action`, asked the same way the footer asks it, so a greyed item and a
greyed key always agree.
"""

import typing

import rich.text
import textual.app
import textual.binding
import textual.containers
import textual.events
import textual.message
import textual.screen
import textual.widget
import textual.widgets
import textual.widgets.option_list

from . import commands

# How wide a menu is at the least, and the gap it keeps between an item's
# label and its key.
_MIN_WIDTH = 24
_KEY_GAP = 3


def title_text(ws: commands.Workspace) -> rich.text.Text:
    """A workspace's title with the letter Alt goes with underlined, as a DOS
    menu bar marks its shortcuts."""
    text = rich.text.Text(ws.title)
    text.stylize("underline", ws.mnemonic, ws.mnemonic + 1)
    return text


class Title(textual.widgets.Static):
    """One workspace's name in the top row. A click drops its menu."""

    def __init__(self, ws: commands.Workspace, active: bool) -> None:
        super().__init__(title_text(ws), classes="title -active" if active else "title")
        self.workspace = ws

    def on_click(self, event: textual.events.Click) -> None:
        event.stop()
        self.post_message(TopBar.MenuRequested(self.workspace.name))


class TopBar(textual.containers.Horizontal):
    """The row of workspace titles, and the status that follows the viewer."""

    DEFAULT_CSS = """
    TopBar {
        height: 1;
        dock: top;
        background: $panel;
    }
    TopBar > .title {
        width: auto;
        padding: 0 1;
    }
    TopBar > .title.-active {
        background: $accent;
        color: $text;
        text-style: bold;
    }
    TopBar > #topbar-status {
        width: 1fr;
        content-align: right middle;
        padding: 0 1;
        color: $text-muted;
    }
    """

    class MenuRequested(textual.message.Message):
        """A title was clicked: drop that workspace's menu."""

        def __init__(self, workspace: str) -> None:
            super().__init__()
            self.workspace = workspace

    def __init__(self, active: str, workspaces: typing.Sequence[commands.Workspace]) -> None:
        super().__init__(id="topbar")
        self.active = active
        self.workspaces = list(workspaces)
        self._status = ""

    def compose(self) -> textual.app.ComposeResult:
        for ws in self.workspaces:
            yield Title(ws, ws.name == self.active)
        yield textual.widgets.Static(self._status, id="topbar-status")

    @property
    def status(self) -> str:
        """What the right-hand side says. A `Static`'s content is not readable back."""
        return self._status

    def show_status(self, text: str) -> None:
        if text == self._status:
            return
        self._status = text
        for status in self.query("#topbar-status"):
            typing.cast(textual.widgets.Static, status).update(text)

    def title_x(self, workspace: str) -> int:
        """Where a workspace's title starts, for its menu to drop from."""
        for title in self.query(Title):
            if title.workspace.name == workspace:
                return title.region.x
        return 0


def app_of(node: textual.widget.Widget) -> textual.app.App[typing.Any]:
    """The app a widget belongs to, typed.

    Textual types `Widget.app` as `App[Unknown]`, which strict checking will
    not let through; the screens reach theirs through `Host` for the same
    reason. A menu genuinely needs the app itself -- it pushes and pops screens
    -- so it asks for it once, here.
    """
    return typing.cast(textual.app.App[typing.Any], node.app)  # pyright: ignore[reportUnknownMemberType]


def enabled(screen: textual.screen.Screen[typing.Any], action: str) -> bool:
    """Whether a menu item would do anything, asked of the screen it would run on.

    An `app.` action is always live. Anything else must be an action the
    screen has, and one its `check_action` does not switch off or grey out --
    `True` exactly, since `None` is Textual's "shown but greyed".
    """
    if action.startswith("app."):
        return True
    name = action.split("(", 1)[0]
    if not hasattr(screen, f"action_{name}"):
        return False
    return screen.check_action(name, ()) is True


def _prompt(label: str, key: str, width: int) -> rich.text.Text:
    text = rich.text.Text(label)
    pad = max(_KEY_GAP, width - len(label) - len(key))
    text.append(" " * pad)
    text.append(key, style="bold")
    return text


def menu_width(entries: typing.Sequence[commands.Entry]) -> int:
    """Wide enough for the longest label and its key, with the gap between."""
    longest = _MIN_WIDTH
    for entry in entries:
        if isinstance(entry, commands.Item):
            longest = max(longest, len(entry.label) + len(entry.key) + _KEY_GAP)
        elif isinstance(entry, commands.Submenu):
            longest = max(longest, len(entry.label) + 1 + _KEY_GAP)
    return longest


class MenuScreen(textual.screen.ModalScreen[None]):
    """One drop-down: a workspace's menu, or a submenu of one.

    `target` is the screen the items act on -- the workspace in front of the
    viewer, whichever menu this is -- and `workspace` is whose menu it is,
    which is what `left` and `right` step from.
    """

    DEFAULT_CSS = """
    MenuScreen {
        align: left top;
        background: transparent;
    }
    """

    BINDINGS = [
        textual.binding.Binding("escape", "close", "close", show=False),
        textual.binding.Binding("f10", "close_all", "close", show=False),
        textual.binding.Binding("left", "left", "", show=False),
        textual.binding.Binding("right", "right", "", show=False),
    ]

    def __init__(
        self,
        workspace: str,
        target: textual.screen.Screen[typing.Any],
        entries: typing.Sequence[commands.Entry],
        x: int,
        y: int,
        submenu: bool = False,
    ) -> None:
        super().__init__()
        self.workspace = workspace
        self.target = target
        self.entries = list(entries)
        self.x = x
        self.y = y
        self.submenu = submenu
        self.width = menu_width(entries)

    def compose(self) -> textual.app.ComposeResult:
        options: list[textual.widgets.option_list.Option | None] = []
        for index, entry in enumerate(self.entries):
            if entry is None:
                options.append(None)
            elif isinstance(entry, commands.Submenu):
                options.append(
                    textual.widgets.option_list.Option(
                        _prompt(entry.label, "▸", self.width), id=str(index)
                    )
                )
            else:
                options.append(
                    textual.widgets.option_list.Option(
                        _prompt(entry.label, entry.key, self.width),
                        id=str(index),
                        disabled=not enabled(self.target, entry.action),
                    )
                )
        yield textual.widgets.OptionList(*options)

    def on_mount(self) -> None:
        menu = self.query_one(textual.widgets.OptionList)
        # Two cells for the border, two for the option list's own padding.
        menu.styles.width = self.width + 4
        menu.styles.offset = (self.x, self.y)
        menu.focus()

    @property
    def items(self) -> list[commands.Entry]:
        return list(self.entries)

    def labels(self) -> list[tuple[str, bool]]:
        """Each item's label and whether it is live, for tests and for help."""
        out: list[tuple[str, bool]] = []
        for entry in self.entries:
            if isinstance(entry, commands.Item):
                out.append((entry.label, enabled(self.target, entry.action)))
            elif isinstance(entry, commands.Submenu):
                out.append((entry.label, True))
        return out

    def _highlighted(self) -> commands.Entry:
        menu = self.query_one(textual.widgets.OptionList)
        if menu.highlighted is None:
            return None
        option = menu.get_option_at_index(menu.highlighted)
        return self.entries[int(option.id or 0)]

    def on_option_list_option_selected(
        self, event: textual.widgets.OptionList.OptionSelected
    ) -> None:
        event.stop()
        self.choose(self.entries[int(event.option.id or 0)])

    def choose(self, entry: commands.Entry) -> None:
        if isinstance(entry, commands.Submenu):
            self.open_submenu(entry)
        elif isinstance(entry, commands.Item):
            run(app_of(self), self.target, entry.action)

    def open_submenu(self, entry: commands.Submenu) -> None:
        # Drawn so that its first item sits on the row of the item that opened
        # it: separators take a row each but are not options, so the row is
        # the entry's place in the menu, not the option's.
        row = self.entries.index(entry)
        app_of(self).push_screen(
            MenuScreen(
                self.workspace,
                self.target,
                entry.items,
                self.x + self.width + 3,
                self.y + row,
                submenu=True,
            )
        )

    def action_close(self) -> None:
        app_of(self).pop_screen()

    def action_close_all(self) -> None:
        close_menus(app_of(self))

    def action_right(self) -> None:
        entry = self._highlighted()
        if isinstance(entry, commands.Submenu):
            self.open_submenu(entry)
            return
        self.step(1)

    def action_left(self) -> None:
        if self.submenu:
            app_of(self).pop_screen()
            return
        self.step(-1)

    def step(self, delta: int) -> None:
        """Close every menu and drop the neighbouring workspace's, wrapping.

        The workspace does not change: only which menu is open.
        """
        names = [ws.name for ws in available(app_of(self))]
        index = names.index(self.workspace) if self.workspace in names else 0
        close_menus(app_of(self))
        open_menu(app_of(self), names[(index + delta) % len(names)])


def available(app: object) -> list[commands.Workspace]:
    """The workspaces this app has, in top-row order.

    Any object: the screens pass their `Host`, which is the app but is not
    typed as one.
    """
    modes = getattr(app, "workspaces", None)
    if modes is None:
        return list(commands.WORKSPACES)
    return [ws for ws in commands.WORKSPACES if ws.name in modes]


def close_menus(app: textual.app.App[typing.Any]) -> None:
    """Take every open menu off the stack, submenus first."""
    while isinstance(app.screen, MenuScreen):
        app.pop_screen()


def open_menu(app: textual.app.App[typing.Any], workspace: str) -> None:
    """Drop `workspace`'s menu under its title, acting on the screen in front.

    Nothing opens over a dialog: whatever it is asking has to be answered
    first, and a menu over it could only act on the screen behind it.
    """
    close_menus(app)
    target = app.screen
    if isinstance(target, textual.screen.ModalScreen):
        return
    bars = target.query(TopBar)
    x = bars.first().title_x(workspace) if bars else 0
    app.push_screen(MenuScreen(workspace, target, commands.MENUS[workspace], x, 1))


def run(
    app: textual.app.App[typing.Any], target: textual.screen.Screen[typing.Any], action: str
) -> None:
    """Close the menus and run an item's action where it belongs."""
    close_menus(app)
    if action.startswith("app."):
        app.call_later(app.run_action, action.removeprefix("app."))
    elif enabled(target, action):
        app.call_later(target.run_action, action)
