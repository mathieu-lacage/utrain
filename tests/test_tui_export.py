"""Tests for writing a plot out to a file.

Pure functions and a `tmp_path`, no terminal and no podman, so -- like
`test_tui_render.py` -- these run in CI, where every cram test skips.

The figure formats are asserted on by their magic bytes rather than by their
contents. What matters here is that the right renderer ran and the file is what
its extension claims; whether matplotlib drew the curve one pixel to the left
is matplotlib's business, and a golden image would fail on every version bump.
"""

import pathlib

import pytest

import utrain.exceptions
import utrain.tui.export as export
import utrain.tui.render as render

# The first bytes each format is recognised by. SVG is XML, and matplotlib
# writes the declaration first.
_MAGIC = {
    "png": b"\x89PNG",
    "svg": b"<?xml",
    "pdf": b"%PDF",
}


def _plot(points: int = 3, title: str = "loss", x_label: str = "step") -> render.Plot:
    return render.Plot(
        title=title,
        x_label=x_label,
        xs=[float(i) for i in range(points)],
        ys=[1.0 + i for i in range(points)],
    )


# -- names ----------------------------------------------------------------


def test_default_path_names_the_run_the_phase_and_the_metric() -> None:
    assert export.default_path("shake", "pretrain", "loss", "csv") == pathlib.Path(
        "shake-pretrain-loss.csv"
    )


def test_default_path_slugifies_a_metric_with_a_slash_in_it() -> None:
    # `train/loss` left alone would name a directory that does not exist.
    assert export.default_path("Shake Run", "pretrain", "train/loss", "png") == pathlib.Path(
        "shake-run-pretrain-train-loss.png"
    )


def test_default_path_of_names_with_nothing_usable_in_them() -> None:
    assert export.default_path("!!", "??", "**", "svg") == pathlib.Path("plot.svg")


def test_default_path_is_relative() -> None:
    assert not export.default_path("shake", "pretrain", "loss", "csv").is_absolute()


def test_replace_suffix_swaps_the_extension() -> None:
    assert export.replace_suffix(pathlib.Path("fig1.png"), "pdf") == pathlib.Path("fig1.pdf")


def test_replace_suffix_adds_one_when_there_is_none() -> None:
    assert export.replace_suffix(pathlib.Path("fig1"), "csv") == pathlib.Path("fig1.csv")


# -- csv ------------------------------------------------------------------


def test_csv_is_headed_by_the_axes(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "loss.csv"
    export.write(_plot(), path, "csv")
    assert path.read_text() == "step,loss\n0.0,1.0\n1.0,2.0\n2.0,3.0\n"


def test_csv_carries_whatever_x_axis_the_dashboard_was_on(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "loss.csv"
    export.write(_plot(points=1, x_label="elapsed (s)"), path, "csv")
    assert path.read_text().splitlines()[0] == "elapsed (s),loss"


def test_csv_is_not_thinned(tmp_path: pathlib.Path) -> None:
    """A csv is the data itself; a figure is a picture of it."""
    points = export.MAX_EXPORT_POINTS + 500
    path = tmp_path / "loss.csv"
    export.write(_plot(points=points), path, "csv")
    # One header row plus every point.
    assert len(path.read_text().splitlines()) == points + 1


# -- figures --------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["png", "svg", "pdf"])
def test_a_figure_is_written_in_the_format_that_was_asked_for(
    fmt: str, tmp_path: pathlib.Path
) -> None:
    path = tmp_path / f"loss.{fmt}"
    export.write(_plot(), path, fmt, caption="shake -- pretrain")
    assert path.read_bytes().startswith(_MAGIC[fmt])


def test_a_figure_carries_its_caption(tmp_path: pathlib.Path) -> None:
    """SVG is text, so the run and phase can be read back out of it.

    Matplotlib draws text as paths and repeats the string in an XML comment,
    which is what this reads. The caption here deliberately has no `--` in it,
    the separator the dashboard actually uses: XML forbids a double dash inside
    a comment, so matplotlib spaces it out and the round trip would be a test
    of that escaping rather than of the caption arriving.
    """
    path = tmp_path / "loss.svg"
    export.write(_plot(), path, "svg", caption="shake / pretrain")
    assert "shake / pretrain" in path.read_text()


def test_a_figure_is_titled_by_its_axes(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "loss.svg"
    export.write(_plot(), path, "svg")
    # Rendered as paths unless the text is kept as text, so assert on the
    # accessible title matplotlib writes into the SVG rather than on glyphs.
    assert "loss vs step" in path.read_text()


def test_a_figure_is_thinned(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[int] = []
    real = render.downsample

    def spy(
        xs: list[float], ys: list[float], limit: int = render.MAX_PLOT_POINTS
    ) -> tuple[list[float], list[float]]:
        out = real(xs, ys, limit)
        seen.append(len(out[0]))
        return out

    monkeypatch.setattr(export.render, "downsample", spy)
    export.write(_plot(points=export.MAX_EXPORT_POINTS * 2), tmp_path / "loss.png", "png")
    assert seen and seen[0] <= export.MAX_EXPORT_POINTS + 1


def test_log_y_is_dropped_when_a_point_is_not_positive(tmp_path: pathlib.Path) -> None:
    """The guard `MetricPlot._redraw` applies, so the file matches the screen.

    A log axis silently drops every non-positive point, which on a metric that
    reaches zero would export a curve with holes the dashboard did not have.
    """
    plot = render.Plot(title="delta", x_label="step", xs=[0.0, 1.0], ys=[1.0, 0.0])
    path = tmp_path / "delta.svg"
    export.write(plot, path, "svg", log_y=True)
    assert path.read_bytes().startswith(_MAGIC["svg"])


# -- refusals -------------------------------------------------------------


def test_an_unknown_format_is_refused(tmp_path: pathlib.Path) -> None:
    with pytest.raises(utrain.exceptions.UI, match="cannot export as jpeg"):
        export.write(_plot(), tmp_path / "loss.jpeg", "jpeg")


def test_a_path_in_no_directory_is_refused(tmp_path: pathlib.Path) -> None:
    with pytest.raises(utrain.exceptions.UI, match="no such directory"):
        export.write(_plot(), tmp_path / "nowhere" / "loss.csv", "csv")


def test_a_path_that_cannot_be_written_is_a_ui_error(tmp_path: pathlib.Path) -> None:
    # A directory where the file should be: the open fails with an OSError that
    # a viewer should read as a message, not as a traceback.
    (tmp_path / "loss.csv").mkdir()
    with pytest.raises(utrain.exceptions.UI, match="could not write"):
        export.write(_plot(), tmp_path / "loss.csv", "csv")
