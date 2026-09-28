"""What the app can be asked to do, written down once.

The workspaces, and the menu each one's title drops, are declared here as data.
The top row is drawn from `WORKSPACES`; a menu is drawn from `MENUS`; and `?`
writes its key map from both. The keys themselves stay where Textual looks for
them -- `BINDINGS` on the screen or the pane that owns them -- and an item names
the action a binding runs, so a menu item and its key cannot come to do two
different things: they are the same action, and the same `check_action` decides
whether either is live.

An item's action runs on the screen in front of the viewer, whichever menu it
was picked from: a command acts on the selection wherever it is. An action the
screen does not have is shown greyed, which is how a menu that belongs to
another workspace says "not from here". Actions prefixed `app.` run on the app
instead -- going to a workspace, quitting -- and are always live.
"""

import dataclasses


@dataclasses.dataclass(frozen=True)
class Workspace:
    """A top-level place: its mode name, its title, and the key that goes there."""

    name: str
    title: str
    key: str

    @property
    def key_display(self) -> str:
        return self.key.upper()


# In the order the top row shows them, and the function keys follow it.
WORKSPACES = (
    Workspace("runs", "Runs", "f1"),
    Workspace("sweeps", "Sweeps", "f2"),
    Workspace("compare", "Compare", "f3"),
    Workspace("system", "System", "f4"),
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


def _go(name: str) -> Item:
    ws = workspace(name)
    return Item(f"Go to {ws.title}", f"app.workspace('{name}')", ws.key_display)


MENUS: dict[str, tuple[Entry, ...]] = {
    "runs": (
        _go("runs"),
        SEPARATOR,
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
        Item("Expand / collapse", "drill_in", "enter"),
        Item("Zoom pane", "zoom", "z"),
    ),
    "sweeps": (
        _go("sweeps"),
        SEPARATOR,
        Item("New sweep...", "new_sweep", "N"),
        Item("Start / resume", "start_sweep", "s"),
        Item("Pause dispatch", "pause_sweep", "p"),
        Item("Extend grid...", "extend_sweep", "+"),
        Item("Retry failed runs", "retry_sweep", "R"),
        Item("Cancel sweep...", "cancel_sweep", "S"),
        Item("Delete sweep...", "delete_sweep", "d"),
        SEPARATOR,
        Item("Open run in Runs", "open_run", "enter"),
        Item("Mark / unmark", "toggle_mark", "space"),
        Item("Compare this sweep", "compare_sweep", "C"),
    ),
    "compare": (
        _go("compare"),
        SEPARATOR,
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
        Item("Next lens", "next_lens", "]"),
        Item("Sort by next column", "next_sort", ">"),
        SEPARATOR,
        Item("Open run in Runs", "open_run", "enter"),
        Item("Mark / unmark", "toggle_mark", "space"),
    ),
    "system": (
        _go("system"),
        SEPARATOR,
        Item("Add image...", "add_image", "a"),
        Item("Delete image...", "delete_image", "d"),
        SEPARATOR,
        Item("Check data store", "check_store"),
        Item("Clean up store...", "gc_store", "G"),
        SEPARATOR,
        Item("Quit", "app.quit", "q"),
    ),
}


# What `?` adds to the menus: the keys that move between places rather than
# act on anything, which no menu lists.
NAVIGATION = (
    ("F1 - F4", "go to Runs, Sweeps, Compare, System; again opens its menu"),
    ("F10", "open the menu of the workspace you are in"),
    ("left / right", "in a menu: the neighbouring menu"),
    ("tab / shift+tab", "the next pane on screen, and the previous"),
    ("1 / 2 / 3", "a pane by the number in its title"),
    (":", "go to a run, phase, sweep or workspace by name"),
    ("escape", "close a menu, a dialog or a panel; out of a pane"),
    ("r", "refresh now, dropping the image caches"),
)

# Keys that belong to one pane and act on the row under its cursor, which is
# why they are in no menu: the metric picker's.
PANE_KEYS = (
    ("space / enter", "in the metric picker: draw this metric, or stop drawing it"),
    ("shift+up / down", "in the metric picker: extend the range space acts on"),
    ("y", "in the metric picker: solo this metric, or stop"),
)
