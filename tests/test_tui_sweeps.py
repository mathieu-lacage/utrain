"""Pilot tests for the Sweeps workspace, and for making a sweep.

The seeding is `test_tui_runs`': the base app's runs, and a draft sweep "lr" of
two runs over them, one finished and one still queued. The sweep calls that
would spawn a dispatcher or describe an image are recorded instead of made.
"""

import pathlib
import typing

import pytest
import textual.widgets

import utrain.compute
import utrain.config
import utrain.container.podman
import utrain.exceptions
import utrain.tui.app
import utrain.tui.menus
import utrain.tui.render
import utrain.tui.screens

from . import test_tui_app as base
from . import test_tui_runs as runs_tests

pytestmark = pytest.mark.anyio

no_settle_delay = base.no_settle_delay
anyio_backend = base.anyio_backend


class _SweepData(base._RecordingData):  # pyright: ignore[reportPrivateUsage]
    """Records the sweep lifecycle, and stubs what shells out."""

    def __init__(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        super().__init__(*args, **kwargs)
        self.sweep_ops: list[tuple[str, str]] = []
        self.extended: list[tuple[str, dict[str, object]]] = []
        self.new_sweeps: list[tuple[str, str, str | None, dict[str, object], list[str]]] = []

    def compute(self) -> utrain.compute.ComputeInfo:
        return self.compute_info

    def presets(self) -> dict[str, str]:
        return {name: base.IMAGE_ID for name in self.images}

    def start_sweep(self, sweep_id: str) -> None:
        if self.fail:
            raise utrain.exceptions.UI(self.fail)
        self.sweep_ops.append(("start", sweep_id))

    def pause_sweep(self, sweep_id: str) -> None:
        self.sweep_ops.append(("pause", sweep_id))

    def cancel_sweep(self, sweep_id: str) -> None:
        self.sweep_ops.append(("cancel", sweep_id))

    def retry_sweep(self, sweep_id: str) -> int:
        self.sweep_ops.append(("retry", sweep_id))
        return 0

    def delete_sweep(self, sweep_id: str) -> int:
        self.sweep_ops.append(("delete", sweep_id))
        return 2

    def extend_sweep(self, sweep_id: str, axes: dict[str, object]) -> int:
        self.extended.append((sweep_id, axes))
        return 1

    def create_sweep(
        self,
        name: str,
        image: str,
        base: str | None,
        axes: dict[str, object],
        compute_specs: list[str],
    ) -> str:
        self.new_sweeps.append((name, image, base, axes, compute_specs))
        return runs_tests.SWEEP_ID


@pytest.fixture()
def app(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> utrain.tui.app.UtrainApp:
    # The presets already name the image by its id, which is what the new-sweep
    # form resolves them to.
    monkeypatch.setattr(utrain.container.podman, "image_id", lambda ref: ref)
    base._seed(tmp_path)  # pyright: ignore[reportPrivateUsage]
    runs_tests._seed_sweep(tmp_path)  # pyright: ignore[reportPrivateUsage]
    source = _SweepData(
        settings=utrain.config.Settings(data_dir=tmp_path, tui_charset="block"),
        describe_cache={base.IMAGE_ID: base._describe()},  # pyright: ignore[reportPrivateUsage]
    )
    return utrain.tui.app.UtrainApp(source)


def _data(app: utrain.tui.app.UtrainApp) -> _SweepData:
    source = app.data
    assert isinstance(source, _SweepData)
    return source


async def _settle(app: utrain.tui.app.UtrainApp, pilot: typing.Any) -> None:
    await base._settle(app, pilot)  # pyright: ignore[reportPrivateUsage]


def _sweeps(app: utrain.tui.app.UtrainApp) -> utrain.tui.screens.SweepsScreen:
    screen = app.screen
    assert isinstance(screen, utrain.tui.screens.SweepsScreen)
    return screen


def _text(app: utrain.tui.app.UtrainApp, identifier: str) -> str:
    return str(app.screen.query_one(identifier, textual.widgets.Static).render())


async def _open(app: utrain.tui.app.UtrainApp, pilot: typing.Any) -> None:
    await _settle(app, pilot)
    await pilot.press("f2")
    await _settle(app, pilot)


async def test_f2_shows_the_sweeps_with_the_grid_queue_and_spec(
    app: utrain.tui.app.UtrainApp,
) -> None:
    async with app.run_test(size=base.SIZE) as pilot:
        await _open(app, pilot)
        screen = _sweeps(app)
        assert screen.selected_sweep == runs_tests.SWEEP_ID
        listed = screen.table("#sweeps")
        assert [str(c) for c in listed.get_row_at(0)] == ["lr", "draft", "1/2"]

        grid = screen.table("#matrix")
        assert grid.row_count == 2
        assert "done" in str(grid.get_row_at(0)[1])
        assert "queued" in str(grid.get_row_at(1)[1])

        assert "next  cpu lr-02" in _text(app, "#queue")
        spec = _text(app, "#spec")
        assert "lr 0.001,0.01 ×2 → 2 runs" in spec
        assert "the image's defaults" in spec


async def test_the_keys_follow_the_sweeps_state(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=base.SIZE) as pilot:
        await _open(app, pilot)
        keys = {key: b.enabled for key, b in _sweeps(app).active_bindings.items()}
        # A draft: it can be started, not paused; nothing has failed.
        assert keys["s"] and not keys["p"] and not keys["R"]
        assert keys["d"] and keys["N"]
        # The list has the focus, so there is no cell to open or mark.
        assert not keys["space"]


async def test_s_starts_the_sweep(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=base.SIZE) as pilot:
        await _open(app, pilot)
        await pilot.press("s")
        await _settle(app, pilot)
        assert _data(app).sweep_ops == [("start", runs_tests.SWEEP_ID)]
        assert _sweeps(app).error == "started lr"


async def test_a_refused_start_says_why(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=base.SIZE) as pilot:
        await _open(app, pilot)
        _data(app).fail = "no GPU toolkit"
        await pilot.press("s")
        await _settle(app, pilot)
        assert _sweeps(app).error == "no GPU toolkit"


async def test_delete_asks_first(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=base.SIZE) as pilot:
        await _open(app, pilot)
        await pilot.press("d")
        await _settle(app, pilot)
        assert isinstance(app.screen, utrain.tui.screens.ConfirmScreen)
        await pilot.press("left", "enter")
        await _settle(app, pilot)
        assert _data(app).sweep_ops == [("delete", runs_tests.SWEEP_ID)]


async def test_enter_on_a_cell_opens_its_run_in_runs(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=base.SIZE) as pilot:
        await _open(app, pilot)
        await pilot.press("enter")  # into the grid, on the first run
        await _settle(app, pilot)
        assert app.screen.focused is _sweeps(app).table("#matrix")

        await pilot.press("enter")
        await _settle(app, pilot)
        await _settle(app, pilot)
        assert app.current_mode == "runs"
        main = base._main(app)  # pyright: ignore[reportPrivateUsage]
        assert main.selected_run == runs_tests.SWEPT_DONE
        # Its sweep was opened so that the run is there to be on.
        assert utrain.tui.render.sweep_key(runs_tests.SWEEP_ID) in main.expanded


async def test_space_on_a_cell_marks_its_run(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=base.SIZE) as pilot:
        await _open(app, pilot)
        await pilot.press("2", "down", "space")
        await _settle(app, pilot)
        await _settle(app, pilot)
        assert app.data.tray() == [runs_tests.SWEPT_QUEUED]
        cell = str(_sweeps(app).table("#matrix").get_row_at(1)[1])
        assert cell.startswith(utrain.tui.render.MARK)


async def test_extend_adds_values_to_an_axis(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=base.SIZE) as pilot:
        await _open(app, pilot)
        await pilot.press("plus")
        await _settle(app, pilot)
        assert isinstance(app.screen, utrain.tui.screens.ExtendSweepScreen)
        await pilot.press(*"0.1", "enter")
        await _settle(app, pilot)
        assert _data(app).extended == [(runs_tests.SWEEP_ID, {"phases.pretrain.lr": ["0.1"]})]
        assert _sweeps(app).error == "added 1 run(s)"


async def test_show_in_sweeps_goes_from_a_sweep_run_to_its_sweep(
    app: utrain.tui.app.UtrainApp,
) -> None:
    async with app.run_test(size=base.SIZE) as pilot:
        await _settle(app, pilot)
        await pilot.press("enter", "down")  # onto lr-01
        await _settle(app, pilot)
        base._main(app).action_show_in_sweeps()  # pyright: ignore[reportPrivateUsage]
        await _settle(app, pilot)
        assert app.current_mode == "sweeps"
        assert _sweeps(app).selected_sweep == runs_tests.SWEEP_ID


async def test_n_from_a_run_makes_a_sweep_around_it(app: utrain.tui.app.UtrainApp) -> None:
    """The image and base are the run's; the form is its config, with the
    grid's size worked out as the axes are typed."""
    async with app.run_test(size=base.SIZE) as pilot:
        await _settle(app, pilot)
        await pilot.press("down")  # onto tiny-shakespeare
        await _settle(app, pilot)
        await pilot.press("N")
        await _settle(app, pilot)
        dialog = app.screen
        assert isinstance(dialog, utrain.tui.screens.NewSweepScreen)
        # The run's image and base are shown, not asked.
        assert not dialog.query("#new-sweep-image")
        await pilot.press(*"lr2", "enter")
        await _settle(app, pilot)

        form = app.screen
        assert isinstance(form, utrain.tui.screens.SweepFormScreen)
        labels = [str(label.render()) for label in form.query(".sweep-field > Label")]
        assert labels == ["Layers", "Precision", "Learning rate", "Resume"]

        # Layers takes 4 and 6; Learning rate is two fields further down.
        await pilot.press(*"4,6", "enter", "down")
        await pilot.press(*"log:1e-4:1e-2:3")
        await _settle(app, pilot)
        assert "n_layer 2 × lr 3 = 6 run(s) · gpu0 6" in _text(app, "#sweep-preview")

        form.query_one("#sweep-form-create", textual.widgets.Button).press()
        await _settle(app, pilot)
        await _settle(app, pilot)
        assert _data(app).new_sweeps == [
            (
                "lr2",
                base.IMAGE,
                base.RUN_ID,
                {
                    "globals.model.n_layer": ["4", "6"],
                    "phases.pretrain.lr": {"log": ["1e-4", "1e-2"], "num": "3"},
                },
                ["gpu0"],
            )
        ]
        # And the viewer is taken to it.
        assert app.current_mode == "sweeps"


async def test_the_form_says_what_a_field_refuses(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=base.SIZE) as pilot:
        await _open(app, pilot)
        await pilot.press("N")
        await _settle(app, pilot)
        assert isinstance(app.screen, utrain.tui.screens.NewSweepScreen)
        await pilot.press(*"depth", "enter")
        await _settle(app, pilot)
        form = app.screen
        assert isinstance(form, utrain.tui.screens.SweepFormScreen)
        await pilot.press(*"0,100")  # Layers is 1..48
        await _settle(app, pilot)
        assert "Layers" in _text(app, "#sweep-form-error")
        form.query_one("#sweep-form-create", textual.widgets.Button).press()
        await _settle(app, pilot)
        assert app.screen is form
        assert _data(app).new_sweeps == []


async def test_a_sweep_name_is_checked_in_the_dialog(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=base.SIZE) as pilot:
        await _open(app, pilot)
        await pilot.press("N")
        await _settle(app, pilot)
        await pilot.press(*"no good", "enter")
        await _settle(app, pilot)
        assert isinstance(app.screen, utrain.tui.screens.NewSweepScreen)
        assert "letters, digits" in _text(app, "#new-sweep-error")


async def test_quitting_keeps_where_you_were(
    app: utrain.tui.app.UtrainApp, tmp_path: pathlib.Path
) -> None:
    """The workspace, the tree's open rows and its cursor, back on restart."""
    async with app.run_test(size=base.SIZE) as pilot:
        await _settle(app, pilot)
        await pilot.press("enter", "down")  # open the sweep, onto lr-01
        await _settle(app, pilot)
        await pilot.press("f2")
        await _settle(app, pilot)
        await pilot.press("q")
        await _settle(app, pilot)

    again = base._app(tmp_path)  # pyright: ignore[reportPrivateUsage]
    async with again.run_test(size=base.SIZE) as pilot:
        await _settle(again, pilot)
        await _settle(again, pilot)
        assert again.current_mode == "sweeps"
        assert _sweeps(again).selected_sweep == runs_tests.SWEEP_ID
        await pilot.press("f1")
        await _settle(again, pilot)
        main = base._main(again)  # pyright: ignore[reportPrivateUsage]
        assert main.selected_run == runs_tests.SWEPT_DONE
        assert runs_tests._names(again)[:3] == ["lr", "  lr-01", "  lr-02"]  # pyright: ignore[reportPrivateUsage]


# -- the goto line -------------------------------------------------------------


async def _goto(app: utrain.tui.app.UtrainApp, pilot: typing.Any, text: str) -> None:
    await pilot.press("colon")
    await _settle(app, pilot)
    assert isinstance(app.screen, utrain.tui.screens.GotoScreen)
    await pilot.press(*text, "enter")
    await _settle(app, pilot)
    await _settle(app, pilot)


async def test_goto_a_workspace_by_name(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=base.SIZE) as pilot:
        await _settle(app, pilot)
        await _goto(app, pilot, "system")
        assert app.current_mode == "system"


async def test_goto_a_sweep(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=base.SIZE) as pilot:
        await _settle(app, pilot)
        await _goto(app, pilot, "@lr")
        assert app.current_mode == "sweeps"
        assert _sweeps(app).selected_sweep == runs_tests.SWEEP_ID


async def test_goto_a_run_inside_a_sweep_opens_the_sweep(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=base.SIZE) as pilot:
        await _settle(app, pilot)
        await pilot.press("f2")
        await _settle(app, pilot)
        await _goto(app, pilot, "lr-02")
        assert app.current_mode == "runs"
        main = base._main(app)  # pyright: ignore[reportPrivateUsage]
        assert main.selected_run == runs_tests.SWEPT_QUEUED
        assert runs_tests._names(app)[:3] == ["lr", "  lr-01", "  lr-02"]  # pyright: ignore[reportPrivateUsage]


async def test_goto_a_phase(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=base.SIZE) as pilot:
        await _settle(app, pilot)
        await _goto(app, pilot, "tiny-shakespeare/tokenizer")
        main = base._main(app)  # pyright: ignore[reportPrivateUsage]
        assert main.selected_run == base.RUN_ID
        assert main.selected_phase == "tokenizer"
        assert main.cursor_key == utrain.tui.render.phase_key(base.RUN_ID, "tokenizer")


async def test_goto_somewhere_unknown_says_so(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=base.SIZE) as pilot:
        await _settle(app, pilot)
        await _goto(app, pilot, "nowhere")
        assert base._main(app).error == "run 'nowhere' not found"  # pyright: ignore[reportPrivateUsage]


async def test_goto_menu_is_f10_for_a_terminal_without_function_keys(
    app: utrain.tui.app.UtrainApp,
) -> None:
    async with app.run_test(size=base.SIZE) as pilot:
        await _settle(app, pilot)
        await _goto(app, pilot, "menu sweeps")
        screen = app.screen
        assert isinstance(screen, utrain.tui.menus.MenuScreen)
        assert screen.workspace == "sweeps"


async def test_goto_a_phase_the_run_lacks_says_so(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=base.SIZE) as pilot:
        await _settle(app, pilot)
        await _goto(app, pilot, "tiny-shakespeare/finetune")
        await _settle(app, pilot)
        main = base._main(app)  # pyright: ignore[reportPrivateUsage]
        assert main.selected_run == base.RUN_ID
        assert main.error == "no phase 'finetune' in that run"
