"""Pilot tests for the Compare workspace, and the comparison logic under it.

Seeds a finished sweep of four runs over two axes, each with a pretrain curve,
on top of `test_tui_app`'s runs -- so a comparison has real metrics files to
read, through the same readers the app uses.
"""

import json
import pathlib
import typing

import pytest
import sqlalchemy
import textual.widgets

import utrain.config
import utrain.db
import utrain.metrics
import utrain.tui.app
import utrain.tui.compare as compare
import utrain.tui.screens
import utrain.tui.widgets
import utrain.types

from . import test_tui_app as base

pytestmark = pytest.mark.anyio

no_settle_delay = base.no_settle_delay
anyio_backend = base.anyio_backend

SWEEP_ID = "9" * 32
LR = "phases.pretrain.lr"
DEPTH = "globals.model.n_layer"
# The four points, and the val/loss each ends on: lr=0.01 with 8 layers is best.
POINTS = [
    ("1" * 32, 0.001, 4, 2.0),
    ("2" * 32, 0.001, 8, 1.8),
    ("3" * 32, 0.01, 4, 1.6),
    ("4" * 32, 0.01, 8, 1.2),
]


def _seed_grid(data_dir: pathlib.Path) -> None:
    settings = utrain.config.Settings(data_dir=data_dir)
    spec = {
        "axes": {LR: [0.001, 0.01], DEPTH: [4, 8]},
        "replicate": [],
        "compute": ["cpu"],
        "base": None,
    }
    with utrain.db.with_db(settings) as session:
        session.execute(
            sqlalchemy.insert(utrain.db.sweeps).values(
                id=SWEEP_ID,
                name="grid",
                image=base.IMAGE,
                image_id=base.IMAGE_ID,
                spec=json.dumps(spec),
                state="done",
                pid=None,
                created_at=1000.0,
            )
        )
        for index, (run_id, lr, depth, final) in enumerate(POINTS):
            run_dir = data_dir / "runs" / run_id
            attempt_dir = run_dir / "attempt" / "1"
            (attempt_dir / "logs").mkdir(parents=True)
            (run_dir / "config.yaml").write_text(
                f"run_id: {run_id}\ncompute: cpu\n"
                f"globals:\n  model:\n    n_layer: {depth}\n    dtype: fp32\n"
                f"phases:\n  pretrain:\n    lr: {lr}\n    resume: false\n"
            )
            base._write_metrics(  # pyright: ignore[reportPrivateUsage]
                attempt_dir,
                "pretrain",
                run_id,
                [{"val/loss": final + 1.0 - step * 0.1} for step in range(11)],
            )
            session.execute(
                sqlalchemy.insert(utrain.db.runs).values(
                    id=run_id,
                    name=f"grid-{index + 1:02d}",
                    image=base.IMAGE,
                    image_id=base.IMAGE_ID,
                    compute="cpu",
                    status="done",
                    config_hash=None,
                    created_at=1001.0 + index,
                    sweep_id=SWEEP_ID,
                    sweep_point=json.dumps({LR: lr, DEPTH: depth}),
                )
            )
            session.execute(
                sqlalchemy.insert(utrain.db.run_attempts).values(
                    run_id=run_id,
                    attempt=1,
                    from_phase=None,
                    status="done",
                    pid=None,
                    started_at=1000.0,
                    ended_at=1100.0,
                )
            )
            for order, phase in enumerate(["tokenizer", "pretrain"]):
                session.execute(
                    sqlalchemy.insert(utrain.db.run_phases).values(
                        run_id=run_id,
                        attempt=1,
                        phase=phase,
                        phase_order=order,
                        status="done",
                        started_at=1000.0,
                        ended_at=1100.0,
                    )
                )


@pytest.fixture()
def app(tmp_path: pathlib.Path) -> utrain.tui.app.UtrainApp:
    base._seed(tmp_path)  # pyright: ignore[reportPrivateUsage]
    _seed_grid(tmp_path)
    return base._app(tmp_path)  # pyright: ignore[reportPrivateUsage]


async def _settle(app: utrain.tui.app.UtrainApp, pilot: typing.Any) -> None:
    await base._settle(app, pilot)  # pyright: ignore[reportPrivateUsage]


def _compare(app: utrain.tui.app.UtrainApp) -> utrain.tui.screens.CompareScreen:
    screen = app.screen
    assert isinstance(screen, utrain.tui.screens.CompareScreen)
    return screen


def _rows(app: utrain.tui.app.UtrainApp) -> list[list[str]]:
    table = _compare(app).table("#compare-table")
    return [[str(c) for c in table.get_row_at(i)] for i in range(table.row_count)]


async def _compare_grid(app: utrain.tui.app.UtrainApp, pilot: typing.Any) -> None:
    await _settle(app, pilot)
    await pilot.press("f2")
    await _settle(app, pilot)
    await pilot.press("C")
    await _settle(app, pilot)
    await _settle(app, pilot)


async def test_with_nothing_marked_it_says_how_to_get_something(
    app: utrain.tui.app.UtrainApp,
) -> None:
    async with app.run_test(size=base.SIZE) as pilot:
        await _settle(app, pilot)
        await pilot.press("f3")
        await _settle(app, pilot)
        screen = _compare(app)
        assert screen.query_one("#compare-empty").display
        assert "Mark runs with space" in str(
            screen.query_one("#compare-empty", textual.widgets.Static).render()
        )


async def test_c_in_sweeps_compares_the_sweep_best_first(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=base.SIZE) as pilot:
        await _compare_grid(app, pilot)
        screen = _compare(app)
        assert app.current_mode == "compare"
        assert screen.state is not None
        assert screen.state.source == "sweep" and screen.state.phase == "pretrain"
        assert screen.state.metric == "val/loss"
        rows = _rows(app)
        assert rows[0][:3] == [" grid-04", "0.01", "8"]
        assert rows[0][4] == f"1.2 {compare.BEST}"
        assert [r[0] for r in rows] == [" grid-04", " grid-03", " grid-02", " grid-01"]
        # The curves are drawn, one per run, the cursor's on top in the highlight.
        curves = screen.query_one("#compare-plot", utrain.tui.widgets.ComparePlot).curves
        assert curves is not None
        assert len(curves.series) == 4
        assert curves.series[-1][2] == compare.HIGHLIGHT
        view = screen.query_one("#compare-view")
        assert "sweep grid (4)" in str(view.border_title)


async def test_the_comparison_is_kept_across_a_restart(
    app: utrain.tui.app.UtrainApp, tmp_path: pathlib.Path
) -> None:
    async with app.run_test(size=base.SIZE) as pilot:
        await _compare_grid(app, pilot)
        await pilot.press("right_square_bracket")  # to the response
        await _settle(app, pilot)

    again = base._app(tmp_path)  # pyright: ignore[reportPrivateUsage]
    async with again.run_test(size=base.SIZE) as pilot:
        await _settle(again, pilot)
        await pilot.press("f3")
        await _settle(again, pilot)
        await _settle(again, pilot)
        state = _compare(again).state
        assert state is not None
        assert (state.source, state.sweep, state.lens) == ("sweep", SWEEP_ID, "response")


async def test_lenses_step_with_the_brackets(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=base.SIZE) as pilot:
        await _compare_grid(app, pilot)
        screen = _compare(app)
        seen: list[str] = []
        for _ in compare.LENSES:
            assert screen.state is not None
            seen.append(screen.state.lens)
            await pilot.press("right_square_bracket")
            await _settle(app, pilot)
        assert seen == list(compare.LENSES)
        assert screen.state is not None and screen.state.lens == "curves"

        await pilot.press("left_square_bracket")  # back round to the diff
        await _settle(app, pilot)
        assert screen.query_one("#compare-diff").display
        diff = screen.table("#compare-diff")
        fields = [str(diff.get_row_at(i)[0]) for i in range(diff.row_count)]
        assert fields == [DEPTH, LR]


async def test_the_heatmap_is_the_grid_with_values(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=base.SIZE) as pilot:
        await _compare_grid(app, pilot)
        await pilot.press("right_square_bracket", "right_square_bracket")
        await _settle(app, pilot)
        grid = str(_compare(app).query_one("#compare-grid", textual.widgets.Static).render())
        assert f"● 1.2 {compare.BEST}" in grid
        assert "● 2" in grid


async def test_sort_moves_to_another_column(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=base.SIZE) as pilot:
        await _compare_grid(app, pilot)
        await pilot.press("less_than_sign", "less_than_sign", "less_than_sign")  # to "lr"
        await _settle(app, pilot)
        screen = _compare(app)
        assert screen.state is not None and screen.state.sort == "lr"
        assert [r[1] for r in _rows(app)] == ["0.001", "0.001", "0.01", "0.01"]


async def test_enter_opens_the_cursor_run_in_runs(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=base.SIZE) as pilot:
        await _compare_grid(app, pilot)
        await pilot.press("enter")
        await _settle(app, pilot)
        await _settle(app, pilot)
        assert app.current_mode == "runs"
        assert base._main(app).selected_run == POINTS[3][0]  # pyright: ignore[reportPrivateUsage]


async def test_a_saved_comparison_of_marks_keeps_its_runs(
    app: utrain.tui.app.UtrainApp,
) -> None:
    async with app.run_test(size=base.SIZE) as pilot:
        await _settle(app, pilot)
        app.data.set_tray([POINTS[0][0], POINTS[1][0]])
        await pilot.press("f3")
        await _settle(app, pilot)
        await _settle(app, pilot)
        screen = _compare(app)
        assert len(_rows(app)) == 2

        screen.action_save_comparison()
        await _settle(app, pilot)
        await pilot.press(*"pair", "enter")
        await _settle(app, pilot)
        await _settle(app, pilot)
        assert screen.state is not None
        assert screen.state.source == "runs" and screen.state.name == "pair"

        # The marks change; the saved comparison does not.
        app.data.set_tray([])
        screen.refresh_data()
        await _settle(app, pilot)
        assert len(_rows(app)) == 2


# -- the logic ---------------------------------------------------------------


def _points(values: list[float]) -> list[utrain.metrics.MetricPoint]:
    return [
        utrain.metrics.MetricPoint(step=i, timestamp=float(i), value=v)
        for i, v in enumerate(values)
    ]


def test_a_loss_is_reduced_by_its_minimum_and_the_rest_by_their_last() -> None:
    assert compare.auto_reducer("val/loss") == "min"
    assert compare.auto_reducer("accuracy") == "last"
    assert compare.reduce(_points([3.0, 1.0, 2.0]), "min") == 1.0
    assert compare.reduce(_points([3.0, 1.0, 2.0]), "last") == 2.0
    assert compare.reduce(_points([float("nan")]), "min") is None


def test_the_default_metric_is_a_validation_loss() -> None:
    assert compare.default_metric(["lr", "train/loss", "val/loss"]) == "val/loss"
    assert compare.default_metric(["lr", "loss"]) == "loss"
    assert compare.default_metric(["mfu", "lr"]) == "lr"
    assert compare.default_metric([]) is None


def test_state_survives_json_and_rejects_nonsense() -> None:
    state = compare.State(source="runs", runs=("a", "b"), lens="diff", name="x")
    assert compare.State.from_json(json.loads(json.dumps(state.to_json()))) == state
    assert compare.State.from_json({"source": "bogus", "lens": "3d"}) == compare.State()
    assert compare.State.from_json("nope") == compare.State()


def test_the_response_averages_what_lands_on_one_point() -> None:
    def run(run_id: str, lr: float, seed: int) -> utrain.types.RunRow:
        return utrain.types.RunRow(
            id=run_id,
            name=run_id,
            image="i",
            image_id="x",
            compute="cpu",
            status="done",
            created_at=0.0,
            attempt=1,
            phase=None,
            sweep_id="s",
            point={"p.a.lr": lr, "p.a.seed": seed},
        )

    runs = [run("a", 0.1, 0), run("b", 0.1, 1), run("c", 1.0, 0)]
    reduced = {"a": 1.0, "b": 3.0, "c": 5.0}
    x_axis, series = compare.response(runs, ["p.a.lr", "p.a.seed"], {}, reduced, None)
    assert x_axis == "p.a.lr"
    assert [(s.xs, s.ys) for s in series] == [([0.1, 1.0], [2.0, 5.0])]


async def test_the_sweeps_grid_scores_each_run_and_stars_the_best(
    app: utrain.tui.app.UtrainApp,
) -> None:
    async with app.run_test(size=base.SIZE) as pilot:
        await _settle(app, pilot)
        await pilot.press("f2")
        await _settle(app, pilot)
        await _settle(app, pilot)
        screen = app.screen
        assert isinstance(screen, utrain.tui.screens.SweepsScreen)
        assert screen.selected_sweep == SWEEP_ID
        grid = screen.table("#matrix")
        assert "cell: min val/loss" in str(grid.border_title)
        assert str(grid.get_row_at(1)[2]) == f" ● 1.2 {compare.BEST}"
        assert str(grid.get_row_at(0)[1]) == " ● 2"
