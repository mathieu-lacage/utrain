"""What the app can be asked to do, written down once.

The workspaces, and each one's menu of commands, are declared here as data.
The top row is drawn from `WORKSPACES`; the menu `Alt+M` drops is drawn from
`MENUS`; and `?` writes its key map from both. The keys themselves stay where
Textual looks for them -- `BINDINGS` on the screen or the pane that owns them --
and an item names the action a binding runs, so a menu item and its key cannot
come to do two different things: they are the same action, and the same
`check_action` decides whether either is live.

The menu shown is always the workspace in front's, and its items run on that
workspace's screen. Actions prefixed `app.` run on the app instead -- quitting
-- and are always live.
"""

import collections.abc
import dataclasses

import textual.binding


@dataclasses.dataclass(frozen=True)
class Workspace:
    """A top-level place: its mode name, its title, and the keys that go there.

    Two keys each. `alt+<letter>` is the one shown: a letter is on every
    keyboard layout and reaches the app from every terminal, where a function
    key is often kept by the terminal for itself -- F1 for its help, F10 for
    its menu bar. The function key stays, for the terminals that pass it on.
    """

    name: str
    title: str
    # The letter underlined in the title: `alt` with it goes there.
    letter: str
    fkey: str

    @property
    def alt(self) -> str:
        return f"alt+{self.letter}"

    @property
    def key_display(self) -> str:
        return f"Alt+{self.letter.upper()}"

    @property
    def mnemonic(self) -> int:
        """Where `letter` is in the title, for the top row to underline it."""
        return self.title.lower().index(self.letter)


# In the order the top row shows them, and the function keys follow it.
WORKSPACES = (
    Workspace("runs", "Runs", "r", "f1"),
    Workspace("sweeps", "Sweeps", "s", "f2"),
    Workspace("compare", "Compare", "c", "f3"),
    # `s` is Sweeps'.
    Workspace("system", "System", "y", "f4"),
)


def workspace(name: str) -> Workspace:
    for ws in WORKSPACES:
        if ws.name == name:
            return ws
    raise KeyError(name)


@dataclasses.dataclass(frozen=True)
class Item:
    """One command: what the menu calls it, what runs it, what key it is on."""

    label: str
    action: str
    key: str = ""


@dataclasses.dataclass(frozen=True)
class Submenu:
    """A menu inside a menu, so a long one stays short enough to fit the screen."""

    label: str
    items: tuple["Entry", ...]


# A line between groups of items.
SEPARATOR = None

Entry = Item | Submenu | None


_REFRESH = Item("Refresh", "force_refresh", "r")
_QUIT = Item("Quit", "app.quit", "q")


MENUS: dict[str, tuple[Entry, ...]] = {
    "runs": (
        Item("New run...", "new_run", "n"),
        Item("New sweep from run...", "new_sweep_from_run", "N"),
        SEPARATOR,
        Submenu(
            "Run",
            (
                Item("Edit config", "edit_config", "e"),
                Item("Start", "start_run", "s"),
                Item("Stop", "stop_run", "S"),
                Item("Restart...", "restart_run", "R"),
                Item("Delete...", "delete_run", "d"),
                Item("Chat with model", "chat_run", "t"),
            ),
        ),
        Submenu(
            "Plot",
            (
                Item("Metrics...", "toggle_metrics", "m"),
                Item("Cycle x axis", "cycle_x", "x"),
                Item("Log y axis", "log_y", "l"),
                Item("Braille / blocks", "charset", "b"),
                Item("Export plot...", "export_plot", "E"),
            ),
        ),
        SEPARATOR,
        Item("Mark for compare", "toggle_mark", "space"),
        Item("Show in Sweeps", "show_in_sweeps"),
        SEPARATOR,
        _REFRESH,
        _QUIT,
    ),
    "sweeps": (
        Item("New sweep...", "new_sweep", "N"),
        Item("Start / resume", "start_sweep", "s"),
        Item("Pause dispatch", "pause_sweep", "p"),
        Item("Extend grid...", "extend_sweep", "+"),
        Item("Retry failed runs", "retry_sweep", "R"),
        Item("Cancel sweep...", "cancel_sweep", "S"),
        Item("Delete sweep...", "delete_sweep", "d"),
        SEPARATOR,
        Item("Mark / unmark", "toggle_mark", "space"),
        Item("Compare this sweep", "compare_sweep", "C"),
        SEPARATOR,
        _REFRESH,
        _QUIT,
    ),
    "compare": (
        Item("Compare the marked runs", "compare_tray"),
        Item("Compare a sweep...", "pick_sweep"),
        Item("Open saved...", "open_saved"),
        Item("Save as...", "save_comparison"),
        Item("Clear the marks", "clear_tray"),
        SEPARATOR,
        Item("Phase...", "pick_phase"),
        Item("Metric...", "pick_metric", "m"),
        Item("Reduce by...", "pick_reducer"),
        Item("Colour by...", "pick_colour"),
        Item("Sort by next column", "next_sort", ">"),
        Item("Sort by previous column", "prev_sort", "<"),
        SEPARATOR,
        Item("Mark / unmark", "toggle_mark", "space"),
        SEPARATOR,
        _REFRESH,
        _QUIT,
    ),
    "system": (
        Item("Add image...", "add_image", "a"),
        Item("Delete image...", "delete_image", "d"),
        SEPARATOR,
        Item("Check data store", "check_store"),
        Item("Clean up store...", "gc_store", "G"),
        SEPARATOR,
        _REFRESH,
        _QUIT,
    ),
}


def _actions(entries: tuple[Entry, ...]) -> set[str]:
    out: set[str] = set()
    for entry in entries:
        if isinstance(entry, Item):
            out.add(_bare(entry.action))
        elif isinstance(entry, Submenu):
            out |= _actions(entry.items)
    return out


def _bare(action: str) -> str:
    """An action's name without its namespace or arguments: `screen.zoom()` is `zoom`."""
    return action.split("(", 1)[0].rsplit(".", 1)[-1]


# Every action some menu offers.
MENU_ACTIONS = frozenset(set[str]().union(*(_actions(items) for items in MENUS.values())))


def footer(
    bindings: collections.abc.Sequence[textual.binding.Binding],
) -> list[textual.binding.BindingType]:
    """`bindings`, with every command a menu offers kept out of the footer.

    The one rule for what the footer shows: keys that move around the
    workspace you are in -- panes, opening what is under the cursor, views --
    and the few that go anywhere else (the menu, goto, help). A command, which
    acts on something, is in the menu with its key beside it; its key still
    works, it is only not listed twice. Applied to every workspace's bindings,
    so a key cannot be in both places or drift between them.
    """
    return [
        dataclasses.replace(b, show=False) if _bare(b.action) in MENU_ACTIONS else b
        for b in bindings
    ]


# What `?` adds to the menus: the keys that move between places rather than
# act on anything, which no menu lists.
NAVIGATION = (
    ("Alt+R / S / C / Y", "go to Runs, Sweeps, Compare, System"),
    ("F1 - F4", "the same, where the terminal passes function keys on"),
    ("Alt+M / F10", "every command of the workspace you are in, as a menu"),
    ("right / left", "in a menu: open a submenu, and close it"),
    ("tab / shift+tab", "the next pane on screen, and the previous"),
    ("1 / 2 / 3", "a pane by the number in its title"),
    ("enter", "open what is under the cursor: a tree row, a sweep, a run"),
    ("z", "in Runs: give the focused pane the whole right side"),
    ("[ / ]", "in Compare: the previous lens, and the next"),
    (":", "go to a run, phase, sweep or workspace by name"),
    ("escape", "close a menu, a dialog or a panel; out of a pane"),
)

# Keys that belong to one pane and act on the row under its cursor, which is
# why they are in no menu: the metric picker's.
PANE_KEYS = (
    ("space / enter", "in the metric picker: draw this metric, or stop drawing it"),
    ("shift+up / down", "in the metric picker: extend the range space acts on"),
    ("y", "in the metric picker: solo this metric, or stop"),
)
