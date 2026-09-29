"""Pilot tests for what the Runs workspace's tree adds: sweeps, and marks.

The seeding and the recording data source are `test_tui_app`'s; this adds a
sweep of two runs, one finished and one still queued, over it.
"""

import json
import pathlib
import typing

import pytest
import sqlalchemy
import textual.widgets

import utrain.config
import utrain.db
import utrain.tui.app
import utrain.tui.render
import utrain.tui.screens
import utrain.tui.widgets

from . import test_tui_app as base

pytestmark = pytest.mark.anyio

no_settle_delay = base.no_settle_delay
anyio_backend = base.anyio_backend

SWEEP_ID = "5" * 32
SWEPT_DONE = "6" * 32
SWEPT_QUEUED = "7" * 32


def _seed_sweep(data_dir: pathlib.Path) -> None:
    """A draft sweep over the learning rate: one run finished, one queued.

    A draft, so that reading it starts no dispatcher.
    """
    settings = utrain.config.Settings(data_dir=data_dir)
    spec = {
        "axes": {"phases.pretrain.lr": [0.001, 0.01]},
        "replicate": [],
        "compute": ["cpu"],
        "base": None,
    }
    with utrain.db.with_db(settings) as session:
        session.execute(
            sqlalchemy.insert(utrain.db.sweeps).values(
                id=SWEEP_ID,
                name="lr",
                image=base.IMAGE,
                image_id=base.IMAGE_ID,
                spec=json.dumps(spec),
                state="draft",
                created_at=900.0,
            )
        )
        for index, (run_id, lr, status) in enumerate(
            ((SWEPT_DONE, 0.001, "done"), (SWEPT_QUEUED, 0.01, "queued"))
        ):
            run_dir = data_dir / "runs" / run_id
            run_dir.mkdir(parents=True)
            (run_dir / "config.yaml").write_text(
                f"run_id: {run_id}\ncompute: cpu\n"
                "globals:\n  model:\n    n_layer: 4\n    dtype: fp32\n"
                f"phases:\n  pretrain:\n    lr: {lr}\n    resume: false\n"
            )
            session.execute(
                sqlalchemy.insert(utrain.db.runs).values(
                    id=run_id,
                    name=f"lr-{index + 1:02d}",
                    image=base.IMAGE,
                    image_id=base.IMAGE_ID,
                    compute="cpu",
                    status=status,
                    config_hash=None,
                    created_at=901.0 + index,
                    sweep_id=SWEEP_ID,
                    sweep_point=json.dumps({"phases.pretrain.lr": lr}),
                )
            )
        # The finished one has an attempt, so it can be opened on its phases.
        session.execute(
            sqlalchemy.insert(utrain.db.run_attempts).values(
                run_id=SWEPT_DONE,
                attempt=1,
                from_phase=None,
                status="done",
                pid=None,
                started_at=901.0,
                ended_at=950.0,
            )
        )
        (data_dir / "runs" / SWEPT_DONE / "attempt" / "1" / "logs").mkdir(parents=True)


@pytest.fixture()
def app(tmp_path: pathlib.Path) -> utrain.tui.app.UtrainApp:
    base._seed(tmp_path)  # pyright: ignore[reportPrivateUsage]
    _seed_sweep(tmp_path)
    return base._app(tmp_path)  # pyright: ignore[reportPrivateUsage]


def _main(app: utrain.tui.app.UtrainApp) -> utrain.tui.screens.MainScreen:
    return base._main(app)  # pyright: ignore[reportPrivateUsage]


async def _settle(app: utrain.tui.app.UtrainApp, pilot: typing.Any) -> None:
    await base._settle(app, pilot)  # pyright: ignore[reportPrivateUsage]


def _names(app: utrain.tui.app.UtrainApp) -> list[str]:
    return base._tree_names(_main(app))  # pyright: ignore[reportPrivateUsage]


async def test_a_sweep_is_a_row_that_opens_on_its_runs(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=base.SIZE) as pilot:
        await _settle(app, pilot)
        assert _names(app) == ["lr", "tiny-shakespeare"]

        await pilot.press("enter")
        await _settle(app, pilot)
        assert _names(app) == ["lr", "  lr-01", "  lr-02", "tiny-shakespeare"]


async def test_a_sweep_row_shows_its_status_grid(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=base.SIZE) as pilot:
        await _settle(app, pilot)
        screen = _main(app)
        assert screen.selected_sweep == SWEEP_ID
        assert screen.query_one("#sweep").display
        assert not screen.query_one("#plots").display
        grid = str(screen.query_one("#sweep-grid", textual.widgets.Static).render())
        assert "lr=0.001" in grid and "lr=0.01" in grid
        assert "● done" in grid and "○ queued" in grid


async def test_a_queued_sweep_run_shows_its_point_and_its_config(
    app: utrain.tui.app.UtrainApp,
) -> None:
    """It has not started, so its config is what there is to see -- read-only,
    since the sweep decided it."""
    async with app.run_test(size=base.SIZE) as pilot:
        await _settle(app, pilot)
        await pilot.press("enter", "down", "down")
        await _settle(app, pilot)
        screen = _main(app)
        assert screen.selected_run == SWEPT_QUEUED
        table = screen.table("#runs")
        assert table.get_row_at(table.cursor_row)[1] == "lr=0.01"
        assert screen.query_one("#config").display
        form = screen.query_one("#config", utrain.tui.widgets.ConfigPane)
        assert not form.editable


async def test_space_marks_a_run_for_comparison(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=base.SIZE) as pilot:
        await _settle(app, pilot)
        await pilot.press("down")  # onto tiny-shakespeare
        await _settle(app, pilot)
        await pilot.press("space")
        await _settle(app, pilot)
        await _settle(app, pilot)

        assert app.data.tray() == [base.RUN_ID]
        screen = _main(app)
        table = screen.table("#runs")
        assert str(table.get_row_at(1)[0]).startswith(utrain.tui.render.MARK)
        assert "1 marked" in app.status_line

        await pilot.press("space")
        await _settle(app, pilot)
        assert app.data.tray() == []


async def test_space_on_a_sweep_row_marks_nothing(app: utrain.tui.app.UtrainApp) -> None:
    async with app.run_test(size=base.SIZE) as pilot:
        await _settle(app, pilot)
        assert _main(app).selected_sweep == SWEEP_ID
        assert not _main(app).active_bindings["space"].enabled


async def test_the_read_only_config_marks_what_the_sweep_varied(
    app: utrain.tui.app.UtrainApp,
) -> None:
    async with app.run_test(size=base.SIZE) as pilot:
        await _settle(app, pilot)
        await pilot.press("enter", "down")  # onto lr-01, which finished
        await _settle(app, pilot)
        assert _main(app).selected_run == SWEPT_DONE

        await pilot.press("e")
        await _settle(app, pilot)
        popup = app.screen
        assert isinstance(popup, utrain.tui.screens.ConfigViewScreen)
        labels = [row.label for row in popup.query_one(utrain.tui.widgets.ConfigPane).rows()]
        assert "Learning rate ◂ swept" in labels
        assert "Layers" in labels
