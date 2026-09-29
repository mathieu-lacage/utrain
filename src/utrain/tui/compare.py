"""What the Compare workspace computes: pure functions of a comparison's runs.

A comparison is a set of runs -- the marked ones, a sweep's, or a set saved
under a name -- seen through one phase, one metric and one way of reducing a
curve to a number (its minimum, maximum or last value). Everything the lenses
show is derived here from what `data.Data.compare` read, so it is testable
without a terminal:

- the table: a row per run, with the axes that tell the runs apart, the reduced
  metric and the best of them starred;
- the curves: every run's metric against step, coloured by one axis;
- the response: the reduced metric against a numeric axis, a line per value of
  the colour axis, replicates averaged;
- the diff: the config fields whose values are not the same on every run.

The state is kept in `tuistate` as JSON, so that the comparison on screen when
the app is closed is the one it opens on.
"""

import dataclasses
import math
import typing

import rich.text

from .. import metrics, types
from . import render

# The lenses, in the order `[` and `]` step through them.
LENSES = ("curves", "response", "heatmap", "diff")

REDUCERS = ("min", "max", "last")

# What a comparison's runs come from.
Source = typing.Literal["tray", "sweep", "runs"]

# The colours the curves are drawn in, one per value of the colour axis. The
# eight ANSI names, so that they survive any terminal and any theme -- without
# black, which vanishes on a dark one, and white, which is the highlight.
PALETTE = ("blue", "magenta", "green", "yellow", "cyan", "red")
HIGHLIGHT = "white"

# A tray's runs have no axes of their own, so the config fields that differ
# stand in for them -- up to this many, which is what a table has room for.
_MAX_DIFF_COLUMNS = 4

BEST = "★"


@dataclasses.dataclass(frozen=True)
class State:
    """Which runs are compared, and how. What is saved, by name or as current."""

    source: Source = "tray"
    sweep: str | None = None
    # The runs of a set saved from the tray: a tray changes, a saved
    # comparison does not.
    runs: tuple[str, ...] = ()
    # None for "pick one": `resolve` fills them in from what the runs logged.
    phase: str | None = None
    metric: str | None = None
    reducer: str | None = None
    colour: str | None = None
    lens: str = "curves"
    # The table's sort column, by header; None is the metric's.
    sort: str | None = None
    # The name it was saved or opened under, for the title.
    name: str | None = None

    def to_json(self) -> dict[str, object]:
        out = dataclasses.asdict(self)
        out["runs"] = list(self.runs)
        return out

    @classmethod
    def from_json(cls, value: object) -> "State":
        if not isinstance(value, dict):
            return cls()
        raw = {str(k): v for k, v in typing.cast(dict[object, object], value).items()}

        def text(key: str) -> str | None:
            item = raw.get(key)
            return str(item) if item is not None else None

        source = text("source")
        runs = raw.get("runs")
        return cls(
            source=source if source in ("tray", "sweep", "runs") else "tray",
            sweep=text("sweep"),
            runs=tuple(str(r) for r in typing.cast(list[object], runs))
            if isinstance(runs, list)
            else (),
            phase=text("phase"),
            metric=text("metric"),
            reducer=text("reducer") if text("reducer") in REDUCERS else None,
            colour=text("colour"),
            lens=str(raw["lens"]) if raw.get("lens") in LENSES else "curves",
            sort=text("sort"),
            name=text("name"),
        )


def auto_reducer(metric: str | None) -> str:
    """A loss is judged by its lowest point; anything else by where it ended."""
    if metric is not None and any(w in metric.lower() for w in ("loss", "perplexity", "error")):
        return "min"
    return "last"


def default_metric(names: list[str]) -> str | None:
    """The metric a comparison opens on: a validation loss if there is one."""
    if not names:
        return None
    lowered = [(n, n.lower()) for n in names]
    for want in (("val", "loss"), ("eval", "loss"), ("loss",)):
        for name, low in lowered:
            if all(w in low for w in want):
                return name
    return sorted(names)[0]


def reduce(points: list[metrics.MetricPoint], reducer: str) -> float | None:
    values = [p.value for p in points if math.isfinite(p.value)]
    if not values:
        return None
    if reducer == "min":
        return min(values)
    if reducer == "max":
        return max(values)
    return values[-1]


def best_of(values: dict[str, float], reducer: str) -> str | None:
    """The run whose reduced value is best: lowest for `min`, highest otherwise."""
    if not values:
        return None
    pick = min if reducer == "min" else max
    return pick(values, key=lambda run_id: values[run_id])


def flatten(config: dict[str, object], prefix: str = "") -> dict[str, object]:
    """A nested config as dotted paths, the form a sweep's axes are named in."""
    out: dict[str, object] = {}
    for key, value in config.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            nested = {str(k): v for k, v in typing.cast(dict[object, object], value).items()}
            out.update(flatten(nested, f"{path}."))
        elif path != "run_id":
            out[path] = value
    return out


def differing(configs: dict[str, dict[str, object]], run_ids: list[str]) -> list[str]:
    """The config paths whose value is not the same on every run, in config order."""
    paths: list[str] = []
    for run_id in run_ids:
        for path in configs.get(run_id, {}):
            if path not in paths:
                paths.append(path)
    out: list[str] = []
    for path in paths:
        seen = {repr(configs.get(r, {}).get(path)) for r in run_ids}
        if len(seen) > 1:
            out.append(path)
    return out


def short(path: str) -> str:
    return path.rsplit(".", 1)[-1]


def shown(value: object) -> str:
    if value is None:
        return "–"
    return render.point_text({"_": value}).split("=", 1)[1]


def axes_of(
    sweep: types.SweepRow | None,
    run_ids: list[str],
    configs: dict[str, dict[str, object]],
) -> list[str]:
    """What tells the runs apart: a sweep's axes, or the fields that differ."""
    if sweep is not None:
        return list(sweep.axes)
    paths = [p for p in differing(configs, run_ids) if not p.startswith("compute")]
    return paths[:_MAX_DIFF_COLUMNS]


def value_at(run: types.RunRow, path: str, configs: dict[str, dict[str, object]]) -> object:
    if path in run.point:
        return run.point[path]
    return configs.get(run.id, {}).get(path)


def colours(
    runs: list[types.RunRow], path: str | None, configs: dict[str, dict[str, object]]
) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """Each run's colour, and the legend: one colour per value of `path`.

    Without a colour axis each run gets its own, cycling through the palette.
    """
    by_run: dict[str, str] = {}
    legend: list[tuple[str, str]] = []
    if path is None:
        for index, run in enumerate(runs):
            colour = PALETTE[index % len(PALETTE)]
            by_run[run.id] = colour
            legend.append((run.name, colour))
        return by_run, legend
    keys: dict[str, str] = {}
    for run in runs:
        value = value_at(run, path, configs)
        key = repr(value)
        if key not in keys:
            keys[key] = PALETTE[len(keys) % len(PALETTE)]
            legend.append((f"{short(path)}={shown(value)}", keys[key]))
        by_run[run.id] = keys[key]
    return by_run, legend


def legend_text(legend: list[tuple[str, str]], highlighted: str | None) -> rich.text.Text:
    text = rich.text.Text()
    for label, colour in legend:
        text.append("── ", style=colour)
        text.append(f"{label}  ")
    if highlighted:
        text.append("── ", style=f"bold {HIGHLIGHT}")
        text.append(f"▶ {highlighted}", style="bold")
    return text


# -- the table ------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Table:
    columns: list[str]
    rows: list[list[render.Cell]]
    # The run each row is, in row order.
    run_ids: list[str]
    best: str | None


def metric_column(metric: str | None, reducer: str) -> str:
    return f"{metric or 'metric'} ({reducer})"


def table(
    runs: list[types.RunRow],
    axes: list[str],
    configs: dict[str, dict[str, object]],
    reduced: dict[str, float],
    steps: dict[str, int],
    durations: dict[str, str],
    metric: str | None,
    reducer: str,
    sort: str | None,
    marked: set[str],
) -> Table:
    """A row per run, sorted, with the best run's value starred."""
    value_column = metric_column(metric, reducer)
    columns = ["RUN", *(short(a) for a in axes), "STATUS", value_column, "STEP", "DURATION"]
    best = best_of(reduced, reducer)

    def sort_key(run: types.RunRow) -> tuple[int, object]:
        column = sort if sort in columns else value_column
        if column == value_column:
            value = reduced.get(run.id)
            if value is None:
                return (1, 0.0)
            return (0, value if reducer == "min" else -value)
        if column == "RUN":
            return (0, run.name)
        if column == "STATUS":
            return (0, run.status)
        if column == "STEP":
            return (0, -steps.get(run.id, -1))
        if column == "DURATION":
            return (0, durations.get(run.id, ""))
        path = axes[[short(a) for a in axes].index(column)]
        value = value_at(run, path, configs)
        return (0, value) if isinstance(value, (int, float)) else (0, repr(value))

    ordered = sorted(runs, key=sort_key)
    rows: list[list[render.Cell]] = []
    for run in ordered:
        value = reduced.get(run.id)
        cell = rich.text.Text("–" if value is None else render.format_value(value))
        if run.id == best:
            cell.append(f" {BEST}", style="bold yellow")
        step = steps.get(run.id)
        rows.append(
            [
                f"{render.MARK if run.id in marked else ' '}{run.name}",
                *(shown(value_at(run, a, configs)) for a in axes),
                render.status_cell(run.status, run.status),
                cell,
                "–" if step is None else str(step),
                durations.get(run.id, "–"),
            ]
        )
    return Table(columns, rows, [r.id for r in ordered], best)


def next_sort(columns: list[str], current: str | None, delta: int, default: str) -> str:
    """The column `<` or `>` moves the sort to."""
    here = current if current in columns else default
    return columns[(columns.index(here) + delta) % len(columns)]


# -- the response ---------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Series:
    label: str
    xs: list[float]
    ys: list[float]
    colour: str


def numeric_axes(
    runs: list[types.RunRow], axes: list[str], configs: dict[str, dict[str, object]]
) -> list[str]:
    out: list[str] = []
    for path in axes:
        values = [value_at(r, path, configs) for r in runs]
        if values and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            out.append(path)
    return out


def response(
    runs: list[types.RunRow],
    axes: list[str],
    configs: dict[str, dict[str, object]],
    reduced: dict[str, float],
    colour: str | None,
) -> tuple[str | None, list[Series]]:
    """The reduced metric against the first numeric axis, a line per colour.

    Runs that land on the same point -- replicates, or axes that are neither
    the x nor the colour -- are averaged. Returns the x axis, or None when
    there is no numeric axis to draw against.
    """
    numeric = numeric_axes(runs, axes, configs)
    if not numeric:
        return None, []
    x_axis = next((a for a in numeric if a != colour), numeric[0])
    group_axis = colour if colour != x_axis else None
    by_run, _ = colours(runs, group_axis, configs)
    groups: dict[str, dict[float, list[float]]] = {}
    labels: dict[str, str] = {}
    for run in runs:
        value = reduced.get(run.id)
        if value is None:
            continue
        key = repr(value_at(run, group_axis, configs)) if group_axis else "all"
        labels[key] = (
            f"{short(group_axis)}={shown(value_at(run, group_axis, configs))}"
            if group_axis
            else "all runs"
        )
        x = float(typing.cast(float, value_at(run, x_axis, configs)))
        groups.setdefault(key, {}).setdefault(x, []).append(value)
    series: list[Series] = []
    for index, (key, points) in enumerate(groups.items()):
        xs = sorted(points)
        colour_name = PALETTE[index % len(PALETTE)]
        if group_axis:
            for run in runs:
                if repr(value_at(run, group_axis, configs)) == key:
                    colour_name = by_run[run.id]
                    break
        series.append(
            Series(
                labels[key],
                xs,
                [sum(points[x]) / len(points[x]) for x in xs],
                colour_name,
            )
        )
    return x_axis, series


def log_scale(xs: list[float]) -> bool:
    """Whether an axis reads better on a log scale: positive, over two decades."""
    positive = [x for x in xs if x > 0]
    return len(positive) == len(xs) > 1 and max(positive) / min(positive) >= 100


# -- the diff -------------------------------------------------------------


def diff_table(
    runs: list[types.RunRow], configs: dict[str, dict[str, object]]
) -> tuple[list[str], list[list[render.Cell]]]:
    """A row per differing field, a column per run."""
    ids = [r.id for r in runs]
    columns = ["FIELD", *(r.name for r in runs)]
    rows: list[list[render.Cell]] = [
        [path, *(shown(configs.get(i, {}).get(path)) for i in ids)]
        for path in differing(configs, ids)
    ]
    return columns, rows
