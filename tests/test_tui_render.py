"""Tests for the TUI's formatting and plot-building.

These are the TUI's equivalent of the render tests in `test_query_layer.py`:
pure functions, no terminal, no podman, so they run in CI.
"""

import pytest

import utrain.compute
import utrain.container.schema
import utrain.metrics
import utrain.runs
import utrain.tui.render as render
import utrain.types


def _points(pairs: list[tuple[int, float]], t0: float = 1000.0) -> list[utrain.metrics.MetricPoint]:
    return [
        utrain.metrics.MetricPoint(step=step, timestamp=t0 + step, value=value)
        for step, value in pairs
    ]


# -- cells ----------------------------------------------------------------


def test_run_cells_truncate_the_id() -> None:
    run = utrain.types.RunRow(
        id="abcdef0123",
        name="shake",
        image="img",
        image_id=None,
        compute="cpu",
        status="running",
        created_at=0.0,
        attempt=None,
        phase=None,
    )
    cells = render.run_cells(run, prefix_len=3)

    assert cells[0] == "abc"
    assert cells[1] == "shake"
    assert str(cells[2]) == "running"


def test_phase_cells_keep_an_inherited_phases_own_status() -> None:
    """A phase this attempt skipped still finished, in the attempt that ran it.

    Its status is reported plainly, with nothing about which attempt earned it:
    that is in the address, and in the column it was only noise.
    """
    entry = utrain.types.PhaseListEntry(
        phase="tokenizer",
        phase_order=0,
        address="ab/1/tokenizer",
        status="done",
        started_at=100.0,
        ended_at=200.0,
        inherited_from=1,
    )
    assert str(render.phase_cells(entry, now=0.0)[1]) == "done"


def test_phase_cells_of_an_inherited_phase_that_left_no_row() -> None:
    entry = utrain.types.PhaseListEntry(
        phase="tokenizer",
        phase_order=0,
        address="ab/2/tokenizer",
        status=None,
        started_at=None,
        ended_at=None,
        inherited_from=1,
    )
    assert str(render.phase_cells(entry, now=0.0)[1]) == "--"


def test_phase_cells_of_a_phase_that_never_ran() -> None:
    entry = utrain.types.PhaseListEntry(
        phase="pretrain",
        phase_order=1,
        address="ab/1/pretrain",
        status=None,
        started_at=None,
        ended_at=None,
        inherited_from=None,
    )
    cells = render.phase_cells(entry, now=0.0)
    assert str(cells[1]) == "--"
    assert cells[2] == "--"
    assert cells[3] == "--"


def test_phase_cells_time_a_running_phase_against_now() -> None:
    entry = utrain.types.PhaseListEntry(
        phase="pretrain",
        phase_order=1,
        address="ab/1/pretrain",
        status="running",
        started_at=1000.0,
        ended_at=None,
        inherited_from=None,
    )
    assert render.phase_cells(entry, now=1090.0)[3] == "1m30s"


# -- status colours -------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "expected"),
    [("running", "bold yellow"), ("done", "green"), ("failed", "bold red")],
)
def test_status_style_of_a_known_status(status: str, expected: str) -> None:
    assert render.status_style(status) == expected


def test_status_style_of_an_unknown_status_is_no_style() -> None:
    assert render.status_style("wedged") == ""


def test_status_style_of_an_inherited_phase_is_dimmed() -> None:
    assert render.status_style(None) == "dim"


def test_format_duration_of_a_finished_phase() -> None:
    assert render.format_duration(1000.0, 1100.0, now=9999.0) == "1m40s"


def test_format_duration_of_a_phase_that_never_started() -> None:
    assert render.format_duration(None, None, now=1.0) == "--"


def test_format_duration_of_hours() -> None:
    assert render.format_duration(0.0, 3 * 3600.0 + 120.0, now=0.0) == "3h02m"


def test_format_time_of_nothing() -> None:
    assert render.format_time(None) == "--"


@pytest.mark.parametrize(
    ("ago", "expected"),
    [(0, "0s ago"), (5, "5s ago"), (90, "1m ago"), (7200, "2h ago"), (172800, "2d ago")],
)
def test_format_age(ago: float, expected: str) -> None:
    assert render.format_age(1000.0 - ago, now=1000.0) == expected


def test_format_age_does_not_go_backwards_on_clock_skew() -> None:
    assert render.format_age(1010.0, now=1000.0) == "0s ago"


# -- x axis ---------------------------------------------------------------


def test_x_axis_choices_always_offers_step_and_elapsed() -> None:
    assert render.x_axis_choices([]) == [render.X_STEP, render.X_ELAPSED]


def test_x_axis_choices_offers_every_logged_metric() -> None:
    assert render.x_axis_choices(["loss", "mfu"]) == [
        render.X_STEP,
        render.X_ELAPSED,
        "loss",
        "mfu",
    ]


# -- downsampling ---------------------------------------------------------


def test_downsample_leaves_a_short_series_alone() -> None:
    xs, ys = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
    assert render.downsample(xs, ys, limit=10) == (xs, ys)


def test_downsample_respects_the_limit() -> None:
    xs = [float(i) for i in range(10_000)]
    thinned_x, thinned_y = render.downsample(xs, list(xs), limit=100)

    assert len(thinned_x) <= 101
    assert len(thinned_x) == len(thinned_y)


def test_downsample_keeps_the_first_and_last_points() -> None:
    """The last point is the one a viewer watching a live run cares about."""
    xs = [float(i) for i in range(1001)]
    thinned_x, _ = render.downsample(xs, list(xs), limit=100)

    assert thinned_x[0] == 0.0
    assert thinned_x[-1] == 1000.0


def test_downsample_keeps_x_and_y_aligned() -> None:
    xs = [float(i) for i in range(5000)]
    ys = [x * 2 for x in xs]
    thinned_x, thinned_y = render.downsample(xs, ys, limit=50)

    assert all(y == x * 2 for x, y in zip(thinned_x, thinned_y))


# -- plots ----------------------------------------------------------------


def test_build_plot_against_step() -> None:
    plot = render.build_plot("loss", {"loss": _points([(0, 3.0), (1, 2.0)])}, render.X_STEP)

    assert plot is not None
    assert plot.xs == [0.0, 1.0]
    assert plot.ys == [3.0, 2.0]
    assert plot.x_label == "step"


def test_build_plot_against_step_uses_a_logged_step_column() -> None:
    """A phase that logs a ``step`` of its own means the axis to be that column.

    wandb's ``_step`` counts log calls; nanochat logs its training step alongside
    its losses, and 1, 2 on the axis where the run was at 100, 200 is not what a
    viewer asking for `step` asked for.
    """
    points = {
        "loss": _points([(0, 3.0), (1, 2.0)]),
        "step": _points([(0, 100.0), (1, 200.0)]),
    }
    plot = render.build_plot("loss", points, render.X_STEP)

    assert plot is not None
    assert plot.xs == [100.0, 200.0]
    assert plot.ys == [3.0, 2.0]
    assert plot.x_label == "step"


def test_build_plot_against_step_falls_back_when_the_column_does_not_pair() -> None:
    """A step column none of whose rows carry the metric leaves `_step`.

    Rare -- a phase logs its step on the same rows as its losses -- but the
    curve surviving beats the plot disappearing over it.
    """
    points = {
        "loss": _points([(1, 3.0), (3, 2.0)]),
        "step": _points([(0, 100.0), (2, 200.0)]),
    }
    plot = render.build_plot("loss", points, render.X_STEP)

    assert plot is not None
    assert plot.xs == [1.0, 3.0]
    assert plot.ys == [3.0, 2.0]


def test_build_plot_against_elapsed_starts_at_zero() -> None:
    """Elapsed is measured from the phase's own first point, not the epoch."""
    points = _points([(0, 3.0), (5, 2.0)], t0=1_700_000_000.0)
    plot = render.build_plot("loss", {"loss": points}, render.X_ELAPSED)

    assert plot is not None
    assert plot.xs == [0.0, 5.0]
    assert plot.x_label == "elapsed (s)"


def test_build_plot_of_an_unlogged_metric_is_nothing() -> None:
    assert render.build_plot("loss", {}, render.X_STEP) is None


def test_build_plot_against_another_metric_joins_on_step() -> None:
    """Two metrics sampled at different rates pair only where they share a step."""
    points = {
        "loss": _points([(0, 3.0), (1, 2.5), (2, 2.0), (3, 1.5)]),
        "mfu": _points([(0, 0.1), (2, 0.3)]),
    }
    plot = render.build_plot("loss", points, "mfu")

    assert plot is not None
    assert plot.xs == [0.1, 0.3]
    assert plot.ys == [3.0, 2.0]
    assert plot.x_label == "mfu"


def test_build_plot_against_a_metric_with_no_shared_steps() -> None:
    points = {
        "loss": _points([(1, 3.0), (3, 2.0)]),
        "val": _points([(0, 9.0), (2, 8.0)]),
    }
    assert render.build_plot("loss", points, "val") is None


def test_build_plot_against_a_metric_that_was_never_logged() -> None:
    assert render.build_plot("loss", {"loss": _points([(0, 1.0)])}, "absent") is None


def test_build_plot_downsamples() -> None:
    points = _points([(i, float(i)) for i in range(5000)])
    plot = render.build_plot("loss", {"loss": points}, render.X_STEP, limit=100)

    assert plot is not None
    assert len(plot.xs) <= 101
    assert plot.xs[-1] == 4999.0


# -- the metrics pane -----------------------------------------------------


def test_metric_rows_mark_the_axes() -> None:
    rows = render.metric_rows(
        ["loss", "mfu", "lr"],
        {"loss": 1.834, "mfu": 0.4123, "lr": 0.0003},
        plotted=[("loss", "lr"), ("mfu", "lr"), ("lr", "lr")],
        solo="loss",
    )

    assert [(r.mark, r.name, r.value) for r in rows] == [
        ("y", "loss", "1.834"),
        (" ", "mfu", "0.4123"),
        ("x", "lr", "0.0003"),
    ]


def test_metric_rows_mark_every_drawn_metric_when_nothing_is_soloed() -> None:
    rows = render.metric_rows(
        ["loss", "mfu"],
        {"loss": 1.0, "mfu": 0.5},
        plotted=[("loss", render.X_STEP)],
        solo=None,
    )

    assert [r.mark for r in rows] == ["*", " "]


def test_metric_rows_mark_every_x_axis_in_use() -> None:
    rows = render.metric_rows(
        ["loss", "mfu", "lr"],
        {},
        plotted=[("loss", "lr"), ("loss", "mfu")],
        solo=None,
    )

    assert [r.mark for r in rows] == ["*", "x", "x"]


def test_metric_rows_mark_a_metric_drawn_against_two_x_axes_once() -> None:
    rows = render.metric_rows(
        ["loss", "mfu"],
        {},
        plotted=[("loss", render.X_STEP), ("loss", "mfu")],
        solo=None,
    )

    assert [r.mark for r in rows] == ["*", "x"]


def test_metric_rows_of_a_metric_with_no_value_yet() -> None:
    rows = render.metric_rows(["loss"], {}, plotted=[], solo=None)
    assert rows[0].value == "--"


# -- the config pane ------------------------------------------------------


def _schema() -> utrain.container.schema.ConfigSchema:
    return utrain.container.schema.ConfigSchema(
        globals=utrain.container.schema.GlobalConfigSchema(
            groups=[
                utrain.container.schema.FieldGroup(
                    name="model",
                    label="Model",
                    fields=[
                        utrain.container.schema.FieldSchema(
                            key="n_layer", label="Layers", type="int", default=4
                        )
                    ],
                )
            ]
        ),
        phases={
            "pretrain": utrain.container.schema.PhaseConfigSchema(
                groups=[
                    utrain.container.schema.FieldGroup(
                        name="optim",
                        label="Optimiser",
                        fields=[
                            utrain.container.schema.FieldSchema(
                                key="lr", label="Learning rate", type="float", default=0.001
                            )
                        ],
                    )
                ]
            )
        },
    )


def test_config_rows_read_the_value_on_disk() -> None:
    rows = render.config_rows(
        _schema(),
        {"globals": {"model": {"n_layer": 12}}, "phases": {"pretrain": {"lr": 0.02}}},
        phase_order=["tokenizer", "pretrain"],
    )

    assert [(r.path, r.value) for r in rows] == [
        (("globals", "model", "n_layer"), 12),
        (("phases", "pretrain", "lr"), 0.02),
    ]


def test_config_rows_fall_back_to_the_declared_default() -> None:
    rows = render.config_rows(_schema(), {}, phase_order=["pretrain"])
    assert [r.value for r in rows] == [4, 0.001]


def test_config_rows_are_labelled_by_section() -> None:
    rows = render.config_rows(_schema(), {}, phase_order=["pretrain"])
    assert [r.section for r in rows] == ["globals: Model", "phase: pretrain"]


def test_config_values_nest_the_paths_back_into_a_config() -> None:
    assert render.config_values(
        [
            (("globals", "model", "n_layer"), 12),
            (("globals", "model", "dtype"), "bf16"),
            (("phases", "pretrain", "lr"), 0.02),
        ]
    ) == {
        "globals": {"model": {"n_layer": 12, "dtype": "bf16"}},
        "phases": {"pretrain": {"lr": 0.02}},
    }


def _compute(gpus: int) -> utrain.compute.ComputeInfo:
    return utrain.compute.ComputeInfo(
        cpu=utrain.compute.CpuInfo(
            name="Fake CPU", cores=16, mem_total_gb=64.0, mem_available_gb=32.0
        ),
        gpus=[
            utrain.compute.GpuInfo(
                index=index,
                name=f"Fake GPU {index}",
                power_draw=10.0,
                power_limit=350.0,
                util=0,
                mem_used_mb=0.0,
                mem_total_mb=24576.0,
            )
            for index in range(gpus)
        ],
    )


def test_compute_options_offer_the_cpu_and_every_gpu() -> None:
    assert render.compute_options(_compute(2)) == [
        ("cpu (Fake CPU, 16 cores)", "cpu"),
        ("gpu0 (Fake GPU 0)", "gpu0"),
        ("gpu1 (Fake GPU 1)", "gpu1"),
    ]


def test_compute_options_always_offer_the_cpu() -> None:
    """A host with no GPU can still have a run made on it."""
    assert [value for _, value in render.compute_options(_compute(0))] == ["cpu"]


def test_compute_option_values_are_what_the_query_layer_accepts() -> None:
    """The dialog must not offer a spec `runs._resolve_compute` would reject.

    Only the cpu is checked against it: the gpu branch asks the host what it
    has, and CI has no GPU to agree with. The shape of the gpu values is the
    previous test's business.
    """
    assert utrain.runs._resolve_compute("cpu") == "cpu"
    assert [value for _, value in render.compute_options(_compute(2))][1:] == ["gpu0", "gpu1"]


# -- which character set a curve is drawn with ----------------------------


def test_the_setting_wins_over_every_guess() -> None:
    """The escape hatch for a viewer whose font the guess is wrong about."""
    assert render.default_charset("block", "utf-8", "xterm-256color") == render.CHARSET_BLOCK
    assert render.default_charset("braille", "ascii", "linux", True) == render.CHARSET_BRAILLE


def test_a_modern_terminal_gets_braille() -> None:
    """Braille is in every monospace font in common use, so it is the default
    unless something visible says otherwise."""
    for term in ("xterm-256color", "screen-256color", "tmux-256color", "alacritty", ""):
        assert render.default_charset("auto", "utf-8", term) == render.CHARSET_BRAILLE


def test_a_non_unicode_encoding_gets_blocks() -> None:
    """Not a question about the font: braille has no representation at all."""
    assert render.default_charset("auto", "ascii", "xterm") == render.CHARSET_BLOCK
    assert render.default_charset("auto", "latin-1", "xterm") == render.CHARSET_BLOCK
    assert render.default_charset("auto", None, "xterm") == render.CHARSET_BLOCK


def test_utf8_is_recognised_however_it_is_spelled() -> None:
    for encoding in ("utf-8", "UTF-8", "utf8", "UTF8"):
        assert render.default_charset("auto", encoding, "xterm") == render.CHARSET_BRAILLE


def test_a_terminal_with_no_font_to_configure_gets_blocks() -> None:
    """The Linux virtual console draws from a framebuffer font of a few hundred
    glyphs, and the rest of these predate the block entirely."""
    for term in ("linux", "dumb", "vt100", "vt220", "ansi"):
        assert render.default_charset("auto", "utf-8", term) == render.CHARSET_BLOCK


def test_the_terminal_name_is_matched_on_its_first_word() -> None:
    """`linux-16color` is still the console; `xterm-kitty` is still xterm."""
    assert render.default_charset("auto", "utf-8", "linux-16color") == render.CHARSET_BLOCK
    assert render.default_charset("auto", "utf-8", "XTERM-KITTY") == render.CHARSET_BRAILLE


def test_the_old_windows_console_gets_blocks() -> None:
    """Its raster fonts stop well short of U+2800."""
    assert render.default_charset("auto", "utf-8", "xterm", True) == render.CHARSET_BLOCK
