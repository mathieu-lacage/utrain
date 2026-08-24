# pyright: reportUnknownMemberType=false
#
# Confined to this file rather than turned off in `pyproject.toml`, and narrower
# than the `reportUnknownArgumentType`/`reportUnknownVariableType` that already
# are: matplotlib's Axes and Figure take `**kwargs: Unknown` on nearly every
# method, so under strict mode every `axes.plot` and `figure.savefig` below is
# an error about the library's own signatures rather than about anything here.
# The alternative is a `type: ignore` on each of eight lines.
"""Writing one of the dashboard's plots out to a file.

The dashboard draws with `uniplot`, which rasterises to a character grid and
has nowhere to put the result but a terminal. Keeping a curve means leaving
that grid behind, so this module takes the same `render.Plot` the widget draws
and writes it as data (`csv`) or as a figure (`png`, `svg`, `pdf`).

Everything here is a pure function of its arguments and a path, which is what
makes it testable without a terminal -- the contract `tui/render.py` states for
the same reason.

matplotlib is imported lazily, the way `naw.cli._plot_matplotlib` does it, even
though it ships with the `tui` extra: an environment upgraded from an older
extra has textual and uniplot but not matplotlib, and a dark export key with a
message saying what to install beats an `ImportError` traceback over the
terminal.
"""

import csv
import pathlib
import re

from .. import exceptions
from . import render

# `csv` is data and the other three are pictures of it. Offered in this order
# because it is the one that never needs matplotlib.
FORMATS = ("csv", "png", "svg", "pdf")

# Images are thinned to this; csv is not. Past this many points a figure gains
# nothing a reader can see, and an svg or a pdf carrying millions of them takes
# longer to open than the run took to produce them. A csv is the data itself,
# and thinning that would be a lie about what the phase logged.
MAX_EXPORT_POINTS = 20_000

# What a run name, a phase and a metric become in a filename. Metric names are
# routinely `train/loss`, so a slash is not merely cosmetic here -- left alone
# it would send the file to a directory that does not exist.
_UNSAFE = re.compile(r"[^a-z0-9]+")


def _slug(text: str) -> str:
    return _UNSAFE.sub("-", text.lower()).strip("-")


def default_path(run_name: str, phase: str, metric: str, fmt: str) -> pathlib.Path:
    """The filename the export dialog opens with.

    Relative, so it lands in the directory `utrain tui` was started from rather
    than somewhere under the runs directory the viewer would then have to go
    find. All three of the run, the phase and the metric are in the name
    because a session comparing two runs' losses would otherwise write the same
    file twice.
    """
    stem = "-".join(part for part in (_slug(run_name), _slug(phase), _slug(metric)) if part)
    return pathlib.Path(f"{stem or 'plot'}.{fmt}")


def replace_suffix(path: pathlib.Path, fmt: str) -> pathlib.Path:
    """``path`` with its extension swapped for ``fmt``'s."""
    return path.with_suffix(f".{fmt}")


def write(
    plot: render.Plot,
    path: pathlib.Path,
    fmt: str,
    *,
    log_y: bool = False,
    caption: str = "",
) -> None:
    """Write ``plot`` to ``path`` in ``fmt``.

    ``log_y`` and ``caption`` say what the dashboard was showing, so that an
    exported figure is a picture of what the viewer was looking at rather than
    of the same numbers drawn differently. Neither means anything to a csv.
    """
    if fmt not in FORMATS:
        raise exceptions.UI(f"cannot export as {fmt}; try one of {', '.join(FORMATS)}")
    parent = path.parent
    if not parent.is_dir():
        raise exceptions.UI(f"no such directory: {parent}")
    try:
        if fmt == "csv":
            _write_csv(plot, path)
        else:
            _write_figure(plot, path, fmt, log_y=log_y, caption=caption)
    except OSError as e:
        raise exceptions.UI(f"could not write {path}: {e.strerror or e}") from e


def _write_csv(plot: render.Plot, path: pathlib.Path) -> None:
    """Two columns, headed by the axes -- the shape `naw plot --output csv` prints."""
    with path.open("w", newline="") as handle:
        # `\n` rather than the excel dialect's `\r\n`: this lands next to the
        # rest of a session's files on a Linux host, and a spreadsheet reads
        # either while `grep` and `diff` do not.
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([plot.x_label, plot.title])
        writer.writerows(zip(plot.xs, plot.ys))


def _write_figure(
    plot: render.Plot,
    path: pathlib.Path,
    fmt: str,
    *,
    log_y: bool,
    caption: str,
) -> None:
    """Draw ``plot`` with matplotlib and save it.

    The import is here rather than at module scope so that `csv` -- the one
    format that needs nothing installed -- keeps working in an environment
    upgraded from an older `tui` extra, and so that the missing package is a
    message rather than a traceback. `naw.cli._plot_matplotlib` does the same.

    Agg is selected before `pyplot` is imported, which is when the backend is
    fixed: there is no display to open a window on, and every format written
    here is a file.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot
    except ImportError:
        raise exceptions.UI("matplotlib is not installed. Run: pip install 'utrain[tui]'") from None

    xs, ys = render.downsample(plot.xs, plot.ys, MAX_EXPORT_POINTS)

    figure, axes = matplotlib.pyplot.subplots(figsize=(8.0, 4.5))
    try:
        axes.plot(xs, ys, linewidth=1.0)
        axes.set_xlabel(plot.x_label)
        axes.set_ylabel(plot.title)
        # The widget's title, so that the file and the pane name the curve the
        # same way.
        axes.set_title(f"{plot.title} vs {plot.x_label}")
        # The same guard `MetricPlot._redraw` applies: a log axis silently drops
        # every non-positive point, so a metric that legitimately reaches zero
        # would export a curve with holes the dashboard did not have.
        if log_y and all(y > 0 for y in ys):
            axes.set_yscale("log")
        axes.grid(True, linewidth=0.3, alpha=0.5)
        if caption:
            # Which run and which phase. On screen that is the pane's border
            # title, which the figure leaves behind; without it an exported png
            # is a curve of something, from somewhere.
            figure.text(0.99, 0.01, caption, ha="right", va="bottom", fontsize=7, color="gray")
        figure.savefig(path, format=fmt, dpi=150, bbox_inches="tight")
    finally:
        matplotlib.pyplot.close(figure)
