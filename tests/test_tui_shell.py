"""Pilot tests for the app's shell: the top row, the workspaces and the menus.

The seeded runs and the recording data source are `test_tui_app`'s, so these
drive the same app the rest of the TUI tests do.
"""

import pathlib
import typing

import pytest
import textual.widget
import textual.widgets

import utrain.compute
import utrain.config
import utrain.tui.app
import utrain.tui.commands
import utrain.tui.data
import utrain.tui.menus
import utrain.tui.screens
import utrain.types

from . import test_tui_app as base

pytestmark = pytest.mark.anyio

# The same autouse fixtures, so a cursor move fetches straight away here too.
no_settle_delay = base.no_settle_delay
anyio_backend = base.anyio_backend


class _ShellData(base._RecordingData):  # pyright: ignore[reportPrivateUsage]
    """Also stubs the machine: its compute, and the image and store calls."""

    def __init__(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        super().__init__(*args, **kwargs)
        self.removed: list[str] = []
        self.gc_runs = 0

    def compute(self) -> utrain.compute.ComputeInfo:
        return self.compute_info

    def remove_image(self, name: str) -> None:
        self.removed.append(name)
        self.images.remove(name)

    def store_gc(self) -> utrain.types.GcResult:
        self.gc_runs += 1
        return utrain.types.GcResult(removed=2, reclaimed_bytes=2048)


@pytest.fixture()
def app(tmp_path: pathlib.Path) -> utrain.tui.app.UtrainApp:
    base._seed(tmp_path)  # pyright: ignore[reportPrivateUsage]
    source = _ShellData(
        settings=utrain.config.Settings(data_dir=tmp_path, tui_charset="block"),
        describe_cache={base.IMAGE_ID: base._describe()},  # pyright: ignore[reportPrivateUsage]
    )
    return utrain.tui.app.UtrainApp(source)


def _bar(app: utrain.tui.app.UtrainApp) -> utrain.tui.menus.TopBar:
    return app.screen.query_one(utrain.tui.menus.TopBar)


def _active_title(app: utrain.tui.app.UtrainApp) -> str:
    return next(
        str(t.render()) for t in app.screen.query(utrain.tui.menus.Title) if t.has_class("-active")
    )


def _menu(app: utrain.tui.app.UtrainApp) -> utrain.tui.menus.MenuScreen:
    screen = app.screen
    assert isinstance(screen, utrain.tui.menus.MenuScreen)
    return screen


async def _settle(app: utrain.tui.app.UtrainApp, pilot: typing.Any) -> None:
    await base._settle(app, pilot)  # pyright: ignore[reportPrivateUsage]


# -- the top row ----------------------------------------------------------


async def test_the_top_row_names_the_workspaces_and_marks_the_one_you_are_in(
    app: utrain.tui.app.UtrainApp,
) -> None:
    async with app.run_test(size=base.SIZE) as pilot:
        await _settle(app, pilot)
        titles = [str(t.render()) for t in app.screen.query(utrain.tui.menus.Title)]
        assert titles == ["Runs", "Sweeps", "Compare", "System"]
        assert _active_title(app) == "Runs"


async def test_a_function_key_goes_to_its_workspace_and_keeps_the_other(
    app: utrain.tui.app.UtrainApp,
) -> None:
    async with app.run_test(size=base.SIZE) as pilot:
        await _settle(app, pilot)
        runs = app.screen
        await pilot.press("f4")
        await _settle(app, pilot)
        assert isinstance(app.screen, utrain.tui.screens.SystemScreen)
        assert _active_title(app) == "System"

        await pilot.press("f1")
        await _settle(app, pilot)
        # The same screen, not a new one: the workspace kept its state.
        assert app.screen is runs


async def test_the_status_counts_the_marked_runs(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=base.SIZE) as pilot:
        await _settle(app, pilot)
        app.data.set_tray([base.RUN_ID])
        app.refresh_status()
        await _settle(app, pilot)
        assert "1 marked" in _bar(app).status
        assert "gpu0 0%" in _bar(app).status


# -- menus ----------------------------------------------------------------


@pytest.mark.parametrize("key", ["alt+m", "f10"])
async def test_the_menu_key_drops_the_menu_of_the_workspace_you_are_in(
    app: utrain.tui.app.UtrainApp, key: str
) -> None:
    async with app.run_test(size=base.SIZE) as pilot:
        await _settle(app, pilot)
        await pilot.press(key)
        await _settle(app, pilot)
        menu = _menu(app)
        assert menu.workspace == "runs"
        assert menu.labels()[0] == ("New run...", True)

        await pilot.press(key)  # the key that opened it closes it
        await _settle(app, pilot)
        assert isinstance(app.screen, utrain.tui.screens.MainScreen)


async def test_escape_closes_the_menu(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=base.SIZE) as pilot:
        await _settle(app, pilot)
        await pilot.press("alt+m")
        await _settle(app, pilot)
        await pilot.press("escape")
        await _settle(app, pilot)
        assert isinstance(app.screen, utrain.tui.screens.MainScreen)


async def test_the_key_of_the_workspace_you_are_in_does_nothing(
    app: utrain.tui.app.UtrainApp,
) -> None:
    async with app.run_test(size=base.SIZE) as pilot:
        await _settle(app, pilot)
        runs = app.screen
        await pilot.press("alt+r")
        await _settle(app, pilot)
        assert app.screen is runs


async def test_the_menu_drops_under_the_active_title_over_the_workspace(
    app: utrain.tui.app.UtrainApp,
) -> None:
    async with app.run_test(size=base.SIZE) as pilot:
        await _settle(app, pilot)
        await pilot.press("alt+y")
        await _settle(app, pilot)
        x = _bar(app).title_x("system")
        await pilot.press("alt+m")
        await _settle(app, pilot)
        menu = _menu(app)
        assert menu.workspace == "system"
        box = menu.query_one(textual.widgets.OptionList)
        assert box.region.x == x
        assert box.region.y == 1
        # Over the workspace, which is still there underneath.
        assert isinstance(app.screen_stack[-2], utrain.tui.screens.SystemScreen)


async def test_left_and_right_on_the_menu_itself_go_nowhere(
    app: utrain.tui.app.UtrainApp,
) -> None:
    """There is one menu, so there is no neighbouring one to step to."""
    async with app.run_test(size=base.SIZE) as pilot:
        await _settle(app, pilot)
        await pilot.press("alt+m")
        await _settle(app, pilot)
        menu = _menu(app)
        await pilot.press("right", "left")
        await _settle(app, pilot)
        assert _menu(app) is menu
        assert app.current_mode == "runs"


async def test_switching_workspace_closes_the_menu(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=base.SIZE) as pilot:
        await _settle(app, pilot)
        await pilot.press("alt+m")
        await _settle(app, pilot)
        await pilot.press("alt+s")
        await _settle(app, pilot)
        assert isinstance(app.screen, utrain.tui.screens.SweepsScreen)


async def test_an_item_that_does_not_apply_is_greyed(app: utrain.tui.app.UtrainApp) -> None:
    """The seeded run is done: it can be restarted, not started or stopped."""
    async with app.run_test(size=base.SIZE) as pilot:
        await _settle(app, pilot)
        await pilot.press("f10")
        await _settle(app, pilot)
        runs_menu = _menu(app)
        run = next(
            e
            for e in runs_menu.items
            if isinstance(e, utrain.tui.commands.Submenu) and e.label == "Run"
        )
        live = {
            item.label: utrain.tui.menus.enabled(runs_menu.target, item.action)
            for item in run.items
            if isinstance(item, utrain.tui.commands.Item)
        }
        assert live["Start"] is False
        assert live["Stop"] is False
        assert live["Restart..."] is True


async def test_every_menu_ends_with_quit(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=base.SIZE) as pilot:
        await _settle(app, pilot)
        for key in ("alt+r", "alt+s", "alt+c", "alt+y"):
            await pilot.press(key, "alt+m")
            await _settle(app, pilot)
            assert _menu(app).labels()[-1] == ("Quit", True)
            await pilot.press("escape")
            await _settle(app, pilot)


async def test_right_opens_a_submenu_beside_its_item_and_left_closes_it(
    app: utrain.tui.app.UtrainApp,
) -> None:
    async with app.run_test(size=base.SIZE) as pilot:
        await _settle(app, pilot)
        await pilot.press("f10")
        await _settle(app, pilot)
        top = _menu(app)
        # Down to "Run", past the separators OptionList skips.
        box = top.query_one(textual.widgets.OptionList)
        run_index = next(
            i
            for i in range(box.option_count)
            if top.entries[int(box.get_option_at_index(i).id or 0)]
            == next(
                e
                for e in top.entries
                if isinstance(e, utrain.tui.commands.Submenu) and e.label == "Run"
            )
        )
        box.highlighted = run_index
        await pilot.press("right")
        await _settle(app, pilot)
        sub = _menu(app)
        assert sub is not top and sub.submenu
        assert [label for label, _ in sub.labels()][:2] == ["Edit config", "Start"]
        sub_box = sub.query_one(textual.widgets.OptionList)
        assert sub_box.region.x > box.region.x

        await pilot.press("left")
        await _settle(app, pilot)
        assert _menu(app) is top


async def test_picking_an_item_runs_its_command_on_the_screen_in_front(
    app: utrain.tui.app.UtrainApp,
) -> None:
    async with app.run_test(size=base.SIZE) as pilot:
        await _settle(app, pilot)
        await pilot.press("alt+m", "enter")  # New run...
        await _settle(app, pilot)
        assert isinstance(app.screen, utrain.tui.screens.NewRunScreen)


def _title(app: utrain.tui.app.UtrainApp, name: str) -> utrain.tui.menus.Title:
    return next(t for t in app.screen.query(utrain.tui.menus.Title) if t.workspace.name == name)


async def test_clicking_a_title_goes_there(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=base.SIZE) as pilot:
        await _settle(app, pilot)
        await pilot.click(_title(app, "system"))
        await _settle(app, pilot)
        assert isinstance(app.screen, utrain.tui.screens.SystemScreen)


async def test_clicking_the_title_you_are_on_drops_its_menu(
    app: utrain.tui.app.UtrainApp,
) -> None:
    async with app.run_test(size=base.SIZE) as pilot:
        await _settle(app, pilot)
        await pilot.click(_title(app, "runs"))
        await _settle(app, pilot)
        assert _menu(app).workspace == "runs"


async def test_function_keys_are_off_under_a_dialog(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=base.SIZE) as pilot:
        await _settle(app, pilot)
        await pilot.press("d")  # the delete prompt
        await _settle(app, pilot)
        await pilot.press("f4")
        await _settle(app, pilot)
        assert isinstance(app.screen, utrain.tui.screens.ConfirmScreen)
        assert app.current_mode == "runs"


async def test_the_key_map_lists_every_workspace_menu(app: utrain.tui.app.UtrainApp) -> None:
    sections = dict(utrain.tui.screens.help_sections(list(utrain.tui.commands.WORKSPACES)))
    assert ("s", "Run > Start") in sections["Runs (Alt+R)"]
    assert ("G", "Clean up store") in sections["System (Alt+Y)"]
    assert (
        "Alt+M / F10",
        "every command of the workspace you are in, as a menu",
    ) in sections["getting around"]


# -- the System workspace -------------------------------------------------


async def test_system_shows_devices_images_and_the_store(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=base.SIZE) as pilot:
        await _settle(app, pilot)
        await pilot.press("f4")
        await _settle(app, pilot)
        screen = app.screen
        assert isinstance(screen, utrain.tui.screens.SystemScreen)
        compute_table = screen.table("#compute")
        assert compute_table.row_count == 2  # the cpu and one gpu
        assert screen.table("#images").row_count == len(base.CHOICE_IMAGES)
        store = str(screen.query_one("#store", textual.widgets.Static).render())
        assert "file(s)" in store


async def test_d_deletes_the_image_under_the_cursor_once_asked(
    app: utrain.tui.app.UtrainApp,
) -> None:
    async with app.run_test(size=base.SIZE) as pilot:
        await _settle(app, pilot)
        await pilot.press("f4")
        await _settle(app, pilot)
        await pilot.press("d")
        await _settle(app, pilot)
        assert isinstance(app.screen, utrain.tui.screens.ConfirmScreen)
        await pilot.press("left", "enter")  # Yes
        await _settle(app, pilot)
        data = app.data
        assert isinstance(data, _ShellData)
        assert data.removed == [base.CHOICE_IMAGES[0]]


async def test_g_cleans_the_store_once_asked(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=base.SIZE) as pilot:
        await _settle(app, pilot)
        await pilot.press("f4")
        await _settle(app, pilot)
        await pilot.press("G")
        await _settle(app, pilot)
        await pilot.press("left", "enter")
        await _settle(app, pilot)
        data = app.data
        assert isinstance(data, _ShellData)
        assert data.gc_runs == 1
        screen = app.screen
        assert isinstance(screen, utrain.tui.screens.SystemScreen)
        assert "reclaimed 2.0 KB" in screen.error


async def test_system_without_podman_says_so_and_shows_the_rest(
    app: utrain.tui.app.UtrainApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_podman() -> list[object]:
        raise FileNotFoundError(2, "No such file or directory", "podman")

    monkeypatch.setattr(app.data, "list_images", no_podman)
    async with app.run_test(size=base.SIZE) as pilot:
        await _settle(app, pilot)
        await pilot.press("f4")
        await _settle(app, pilot)
        screen = app.screen
        assert isinstance(screen, utrain.tui.screens.SystemScreen)
        assert screen.error.startswith("cannot list images:")
        assert screen.table("#compute").row_count > 0


# -- Alt with a letter ---------------------------------------------------------


async def test_alt_with_the_underlined_letter_goes_to_each_workspace(
    app: utrain.tui.app.UtrainApp,
) -> None:
    async with app.run_test(size=base.SIZE) as pilot:
        await _settle(app, pilot)
        for key, mode in (
            ("alt+s", "sweeps"),
            ("alt+c", "compare"),
            ("alt+y", "system"),
            ("alt+r", "runs"),
        ):
            await pilot.press(key)
            await _settle(app, pilot)
            assert app.current_mode == mode


def test_each_title_underlines_its_letter() -> None:
    for ws in utrain.tui.commands.WORKSPACES:
        text = utrain.tui.menus.title_text(ws)
        underlined = [
            text.plain[span.start : span.end]
            for span in text.spans
            if "underline" in str(span.style)
        ]
        assert [u.lower() for u in underlined] == [ws.letter]


def test_the_raw_bytes_a_terminal_sends_for_alt_reach_the_binding() -> None:
    """Escape then the letter, the way xterm and friends send Alt+letter."""
    from textual import events
    from textual._xterm_parser import XTermParser  # pyright: ignore[reportPrivateUsage]

    parser = XTermParser()
    keys = [e.key for e in parser.feed("\x1by") if isinstance(e, events.Key)]
    keys += [e.key for e in parser.feed("") if isinstance(e, events.Key)]
    assert keys == ["alt+y"]


# -- escape --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "home", "other"),
    [
        ("alt+r", "#runs", "2"),
        ("alt+s", "#sweeps", "2"),
        ("alt+c", "#compare-table", None),
        ("alt+y", "#images", "1"),
    ],
)
async def test_escape_on_a_workspace_comes_back_to_its_list_and_stays(
    app: utrain.tui.app.UtrainApp, key: str, home: str, other: str | None
) -> None:
    """A workspace's own screen has nothing under it: escape must not close it."""
    async with app.run_test(size=base.SIZE) as pilot:
        await _settle(app, pilot)
        await pilot.press(key)
        await _settle(app, pilot)
        screen = app.screen
        if other is not None:
            await pilot.press(other)  # into another pane
            await _settle(app, pilot)
        for _ in range(3):
            await pilot.press("escape")
            await _settle(app, pilot)
        assert app.screen is screen
        assert app.focused is screen.query_one(home)


# -- focus looks the same everywhere ------------------------------------------


def _border(widget: textual.widget.Widget) -> tuple[str, object]:
    edge = widget.styles.border_top
    return edge[0], edge[1]


async def test_every_workspace_lights_its_focused_pane_as_runs_does(
    app: utrain.tui.app.UtrainApp,
) -> None:
    """The focused pane's frame, and an unfocused one's, match the Runs panes'."""
    async with app.run_test(size=base.SIZE) as pilot:
        await _settle(app, pilot)
        app.data.set_tray([base.RUN_ID])  # so Compare has a table to show
        runs = app.screen
        focused = _border(runs.query_one("#runs"))
        blurred = _border(runs.query_one("#log"))
        assert focused != blurred

        for key, lit, unlit in (
            ("alt+s", "#sweeps", "#matrix"),
            ("alt+c", "#compare-table", "#compare-view"),
            ("alt+y", "#images", "#compute"),
        ):
            await pilot.press(key)
            await _settle(app, pilot)
            await _settle(app, pilot)
            screen = app.screen
            assert app.focused is screen.query_one(lit), key
            assert _border(screen.query_one(lit)) == focused, key
            assert _border(screen.query_one(unlit)) == blurred, key
