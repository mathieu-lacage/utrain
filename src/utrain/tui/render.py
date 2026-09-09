"""Turns query-layer results into what the TUI's widgets take.

The same job `cli/render.py` does, for a different client. It returns row cells
rather than padded strings -- `cli/output.format_table` sizes its own columns,
which a `DataTable` also does and does better -- and it owns the two
transformations the dashboard needs: pairing a metric against a chosen x axis,
and thinning a series down to what a plot can usefully draw.

Everything here is a pure function of its arguments, which is what makes the
dashboard testable without a terminal.
"""

import dataclasses
import datetime
import typing

import rich.text

from .. import compute, metrics, types
from .. import container as containermod

# The sidebar is narrow, so these carry only what identifies a row; whatever a
# run's image, compute or attempt is goes in the content pane's title, where
# there is room for it.
RUN_COLUMNS = ("ID", "NAME", "STATUS")
PHASE_COLUMNS = ("PHASE", "STATUS", "STARTED", "DURATION")
IMAGE_COLUMNS = ("NAME", "SIZE", "RUNS")
COMPUTE_COLUMNS = (
    "ID",
    "KIND",
    "NAME",
    "CORES",
    "POWER (W)",
    "COMPUTE",
    "MEM_USED (GB)",
    "MEM_TOTAL (GB)",
    "MEM_USED (%)",
)

# Above this many points a plot gains nothing from the extra resolution -- the
# terminal has a few hundred columns at most -- and uniplot pays for every one.
# `naw plot` thins at the same threshold.
MAX_PLOT_POINTS = 1000

# The x axis every phase can offer, whatever it logged.
X_STEP = "step"
X_ELAPSED = "elapsed"

# uniplot's names for how a point is drawn. Half-blocks pack 2x2 points into a
# cell and braille dots 2x4, so braille doubles the vertical resolution -- where
# the terminal font has the glyphs.
CHARSET_BLOCK = "block"
CHARSET_BRAILLE = "braille"

# Terminals that cannot draw braille whatever font is configured, because the
# font is not configurable: the Linux virtual console renders from a framebuffer
# font of a few hundred glyphs, and the rest of these predate the block.
_NO_BRAILLE_TERMS = frozenset(
    {"linux", "dumb", "vt52", "vt100", "vt102", "vt220", "ansi", "cons25", "unknown"}
)


def default_charset(
    setting: str,
    encoding: str | None,
    term: str,
    legacy_windows: bool = False,
) -> str:
    """Which character set to draw curves with, before the viewer says.

    Whether the font in front of the viewer has U+2800-U+28FF is the one thing
    a terminal program cannot find out. The encoding can be asked, and so can
    the terminal's name, and neither answers the question -- a UTF-8 terminal
    accepts the bytes and renders tofu just the same. The cursor-position
    report, which is the only real probe there is, measures the width a glyph
    took: a substituted box is one cell wide, exactly like the glyph it stands
    in for, so it cannot see the failure that matters.

    What is left is a guess from what *is* knowable, and it is a good one:
    braille is in every monospace font in common use, and the terminals that
    cannot draw it are the ones with no font to configure. So braille unless
    something visible says otherwise -- and `b` and `tui_charset` are there for
    the viewer the guess is wrong about, who can see that it is wrong the
    moment the plot is drawn.
    """
    if setting != "auto":
        return CHARSET_BRAILLE if setting == CHARSET_BRAILLE else CHARSET_BLOCK
    if legacy_windows:
        # The old Windows console, whose raster fonts stop well short of U+2800.
        return CHARSET_BLOCK
    if encoding is None or "utf" not in encoding.replace("-", "").lower():
        # Braille has no representation outside Unicode, so this is not a
        # question about the font at all.
        return CHARSET_BLOCK
    if term.split("-")[0].lower() in _NO_BRAILLE_TERMS:
        return CHARSET_BLOCK
    return CHARSET_BRAILLE


# What a plot is: the metric on the y axis, and what it is drawn against. A pair
# rather than the metric alone, because a phase can name two default plots of the
# same metric against different x axes, and the two are different plots.
PlotKey = tuple[str, str]


@dataclasses.dataclass(frozen=True)
class Plot:
    """One metric, resolved against an x axis and ready to hand to uniplot."""

    title: str
    x_label: str
    xs: list[float]
    ys: list[float]
    # The metric's own latest value, already formatted. Carried on the plot so
    # that the curve can be labelled with it: a live readout costs no columns
    # there, and it sits on the curve it describes. Read off the raw series
    # rather than off `ys`, which against a metric x axis holds only the points
    # the join could pair, and which downsampling may have thinned.
    latest: str = ""


# A table cell is plain text unless it is coloured, and only status cells are.
Cell = str | rich.text.Text

# One mapping, used by both sidebar lists. Run statuses are configuring /
# running / done / failed / stopped, phase statuses pending / running / done /
# failed / stopped, and the two overlap enough that splitting them would only
# invite them to drift apart.
_STATUS_STYLES = {
    "configuring": "cyan",
    "pending": "dim",
    "running": "bold yellow",
    "done": "green",
    "failed": "bold red",
    "stopped": "red",
}


def status_style(status: str | None) -> str:
    """The Rich style a status is drawn in. Empty for one we do not know."""
    if status is None:
        return "dim"
    return _STATUS_STYLES.get(status, "")


def status_cell(text: str, status: str | None) -> rich.text.Text:
    """A status as a coloured cell.

    `rich.text.Text` rather than markup: a status is data, and a style applied
    to the object cannot be broken by a square bracket appearing in it.
    """
    return rich.text.Text(text, style=status_style(status))


def format_time(ts: float | None) -> str:
    """A timestamp as a table cell.

    Its own rather than `cli/output.format_time`: `cli/` is a sibling client of
    the query layer, not a library underneath this one, and the two genuinely
    want different things -- see the note in `types.py` about a table wanting
    "2026-08-22 14:03" where a TUI wants "3m ago". The absolute form is used in
    columns, so a listing here reads the same as the CLI's; `format_age` is for
    the places a TUI is better off with the relative one.
    """
    if ts is None:
        return "--"
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def format_age(ts: float | None, now: float) -> str:
    """How long ago, for a view that is refreshing in front of the viewer."""
    if ts is None:
        return "--"
    seconds = int(max(0.0, now - ts))
    if seconds < 60:
        return f"{seconds}s ago"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"


def format_duration(started_at: float | None, ended_at: float | None, now: float) -> str:
    """How long a phase took, or has been taking.

    A running phase has no `ended_at`, and the column it goes in is the one the
    viewer watches to see that it is still moving, so it counts up from `now`.
    """
    if started_at is None:
        return "--"
    end = ended_at if ended_at is not None else now
    seconds = int(max(0.0, end - started_at))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


def run_cells(run: types.RunRow, prefix_len: int) -> list[Cell]:
    return [
        run.id[:prefix_len],
        run.name,
        status_cell(run.status, run.status),
    ]


def phase_cells(entry: types.PhaseListEntry, now: float) -> list[Cell]:
    # A phase the current attempt skipped keeps the status it finished with:
    # the run is standing on that phase's output, so what came of it is what
    # the list should say. Which attempt it came from is in the address, and
    # does not belong in the status column.
    status = entry.status if entry.status is not None else "--"
    return [
        entry.phase,
        status_cell(status, entry.status),
        format_age(entry.started_at, now),
        format_duration(entry.started_at, entry.ended_at, now),
    ]


def run_summary(detail: types.RunDetail) -> list[tuple[str, str]]:
    """A run's own facts, as label/value pairs for the config pane.

    Everything here used to be the run screen's detail block. It sits above the
    config because the two answer the same question -- what is this run -- and
    because neither `run_id` nor `compute` is a config field, so neither appears
    below.
    """
    run = detail.run
    attempts = (
        f"{detail.n_attempts} (latest: {detail.latest_attempt_status})"
        if detail.n_attempts
        else "0"
    )
    return [
        ("id", run.id),
        ("name", run.name),
        ("image", run.image),
        ("compute", run.compute),
        ("status", run.status),
        ("attempts", attempts),
        ("created", format_time(run.created_at)),
    ]


def format_config_value(value: object) -> str:
    """A config value as the read-only pane shows it."""
    if value is None:
        return "--"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def run_title(detail: types.RunDetail) -> str:
    """What the content pane is showing, when it is showing a run."""
    run = detail.run
    return f"{run.name}  --  {run.image} on {run.compute}  --  {run.status}"


def phase_title(phase_label: str, detail: types.RunDetail) -> str:
    """What the content pane is showing, when it is showing a phase."""
    label = phase_label or "no phase"
    return f"{label}  --  {detail.run.name}  --  attempt {_dash(detail.run.attempt)}"


def _dash(value: object | None) -> str:
    return "--" if value is None else str(value)


def x_axis_choices(columns: list[str]) -> list[str]:
    """What the x axis can be cycled through for a phase logging ``columns``.

    Step and elapsed time are always available -- every row carries both. Any
    logged metric can also serve as an x axis, which is how you plot one metric
    against another rather than against time.
    """
    return [X_STEP, X_ELAPSED, *columns]


def downsample(
    xs: list[float], ys: list[float], limit: int = MAX_PLOT_POINTS
) -> tuple[
    list[float],
    list[float],
]:
    """Thin a series to at most ``limit`` points, keeping its shape and its end.

    The last point is always kept: on a live chart it is the one the viewer is
    actually watching, and a stride that happens not to land on it would make
    the curve stop short of the newest value.
    """
    if len(xs) <= limit or limit < 1:
        return xs, ys
    stride = len(xs) // limit + 1
    keep = list(range(0, len(xs), stride))
    if keep[-1] != len(xs) - 1:
        keep.append(len(xs) - 1)
    return [xs[i] for i in keep], [ys[i] for i in keep]


def build_plot(
    name: str,
    points: dict[str, list[metrics.MetricPoint]],
    x_axis: str,
    limit: int = MAX_PLOT_POINTS,
) -> Plot | None:
    """Pair metric ``name`` against ``x_axis``, or None when it cannot be drawn.

    Against step or elapsed time this is a straight read of the metric's own
    points, because each carries both. Against another metric it is a join on
    step: the two series are sparse and independently sampled -- a phase may log
    ``loss`` every step and a validation metric every hundredth -- so only the
    steps they share can be paired, and there may be none.
    """
    series = points.get(name)
    if not series:
        return None

    if x_axis == X_STEP:
        xs = [float(p.step) for p in series]
        ys = [p.value for p in series]
        x_label = "step"
    elif x_axis == X_ELAPSED:
        start = series[0].timestamp
        xs = [p.timestamp - start for p in series]
        ys = [p.value for p in series]
        x_label = "elapsed (s)"
    else:
        other = points.get(x_axis)
        if not other:
            return None
        by_step = {p.step: p.value for p in other}
        pairs = [(by_step[p.step], p.value) for p in series if p.step in by_step]
        if not pairs:
            return None
        xs = [x for x, _ in pairs]
        ys = [y for _, y in pairs]
        x_label = x_axis

    xs, ys = downsample(xs, ys, limit)
    return Plot(title=name, x_label=x_label, xs=xs, ys=ys, latest=format_value(series[-1].value))


def compute_options(info: compute.ComputeInfo) -> list[tuple[str, str]]:
    """(label, value) pairs for the new-run dialog's compute picker.

    The values are exactly what `runs._resolve_compute` accepts -- `cpu` and
    `gpu<index>` -- because it is the one that decides, and a dialog offering
    anything else would only be rejected on create.
    """
    options = [(f"cpu ({info.cpu.name}, {info.cpu.cores} cores)", "cpu")]
    for gpu in info.gpus:
        options.append((f"gpu{gpu.index} ({gpu.name})", f"gpu{gpu.index}"))
    return options


def compute_rows(info: compute.ComputeInfo) -> list[list[str]]:
    """Host CPU and GPUs, one row each, as `compute list` shows them."""
    cpu = info.cpu
    mem_used = cpu.mem_total_gb - cpu.mem_available_gb
    rows = [
        [
            "cpu",
            "cpu",
            cpu.name,
            str(cpu.cores),
            "--",
            "--",
            f"{mem_used:.1f}",
            f"{cpu.mem_total_gb:.1f}",
            f"{100 * mem_used / cpu.mem_total_gb:.1f}%" if cpu.mem_total_gb else "--",
        ]
    ]
    for gpu in info.gpus:
        used_gb = gpu.mem_used_mb / 1024
        total_gb = gpu.mem_total_mb / 1024
        rows.append(
            [
                f"gpu{gpu.index}",
                "gpu",
                gpu.name,
                "--",
                f"{gpu.power_draw:.1f}/{gpu.power_limit:.1f}",
                f"{gpu.util}%",
                f"{used_gb:.1f}",
                f"{total_gb:.1f}",
                f"{100 * used_gb / total_gb:.1f}%" if total_gb else "--",
            ]
        )
    return rows


# -- the destinations -----------------------------------------------------

# The top-level views, in order, as (key, name, mode). Runs has no key of its
# own because escape is how every other destination is left, and that is the
# one it goes to.
DESTINATIONS = (
    ("", "Runs", "runs"),
    ("i", "Images", "images"),
    ("c", "Compute", "compute"),
)


def destinations(active: str) -> rich.text.Text:
    """The strip under the header, with `active` picked out.

    Runs, images and compute are siblings -- none of them is inside another --
    so they are shown side by side rather than reached by pushing one over the
    top of the next.
    """
    line = rich.text.Text("  ")
    for key, name, mode in DESTINATIONS:
        if line.plain != "  ":
            line.append("   ")
        label = f" {key} {name} " if key else f" {name} "
        line.append(label, style="bold reverse" if mode == active else "dim")
    return line


# -- the content column's tabs --------------------------------------------

# The tabs, in order, as (key, name, pane id). The key is the number that
# selects the tab and the pane id is the widget it shows, so this is the one
# place the three are tied together.
CONTENT_TABS = (
    ("2", "Config", "config"),
    ("3", "Plots", "plots"),
    ("4", "Logs", "log"),
)


def content_panes() -> tuple[str, ...]:
    """The widget ids the tabs switch between."""
    return tuple(pane for _, _, pane in CONTENT_TABS)


def content_tabs(active: str) -> rich.text.Text:
    """The tab strip above the content column, with `active` picked out.

    A line of text rather than a `TabbedContent`: the tabs here are switched by
    number keys and never by clicking through a bar, so what is wanted is the
    label and which one is current, not a widget with its own focus and its own
    key handling to keep out of the way of the screen's.
    """
    line = rich.text.Text("  ")
    for key, name, pane in CONTENT_TABS:
        if line.plain != "  ":
            line.append("   ")
        style = "bold reverse" if pane == active else "dim"
        line.append(f" {key} {name} ", style=style)
    return line


# -- the help screen ------------------------------------------------------

# Wide enough for the longest key spelling in `screens._HELP`, so the
# descriptions line up in one column.
_HELP_KEY_WIDTH = 16


def help_heading(heading: str) -> rich.text.Text:
    return rich.text.Text(f"\n{heading}", style="bold")


def help_line(keys: str, what: str) -> rich.text.Text:
    """One key and what it does, as two aligned columns."""
    line = rich.text.Text("  ")
    # A plain Rich colour, as the status styles are: `$accent` is Textual CSS
    # and Rich is what renders this.
    line.append(keys.ljust(_HELP_KEY_WIDTH), style="bold cyan")
    line.append(what)
    return line


# -- the metrics pane -----------------------------------------------------

# What the gutter of the metrics pane says about a metric. `x` and `y` name the
# axes; `*` marks a metric drawn in the stacked view.
MARK_X = "x"
MARK_Y = "y"
MARK_PLOTTED = "*"


@dataclasses.dataclass(frozen=True)
class MetricRow:
    """One line of the metrics pane: what it is doing, its name, its value."""

    mark: str
    name: str
    value: str


def format_value(value: float | None) -> str:
    """A metric's latest value, narrow enough for the sidebar.

    `%g` rather than a fixed number of decimals: a loss of 1.834, a learning
    rate of 3e-4 and a token count of 4.2e6 all have to fit the same column.
    """
    return "--" if value is None else f"{value:.4g}"


def metric_rows(
    columns: list[str],
    last: dict[str, float],
    plotted: list[PlotKey],
    solo: str | None,
) -> list[MetricRow]:
    """The metrics pane, in the order the phase first logged each metric.

    A metric is at most one thing at a time, so the marks are ranked rather than
    combined: being an x axis is what a viewer most needs to see, then being
    the soloed y axis, then merely being drawn.

    The x mark can land on more than one row: the plots drawn need not share an
    x axis, since a phase naming its own plots gives each one its own.
    """
    x_axes = {x for _, x in plotted}
    y_axes = {y for y, _ in plotted}
    rows: list[MetricRow] = []
    for name in columns:
        if name in x_axes:
            mark = MARK_X
        elif name == solo:
            mark = MARK_Y
        elif solo is None and name in y_axes:
            mark = MARK_PLOTTED
        else:
            mark = " "
        rows.append(MetricRow(mark=mark, name=name, value=format_value(last.get(name))))
    return rows


# -- the config pane ------------------------------------------------------


def _mapping(value: object) -> dict[str, object]:
    """`value` as a mapping, or an empty one.

    Config reaches here as `object`, having been `yaml.safe_load`'s `Any` one
    layer down, and every place that indexes into it wants both the guard and a
    typed result. `runs._mapping` does the same for the write path; the two are
    deliberately layer-local rather than shared, since neither module is a
    library for the other.
    """
    if not isinstance(value, dict):
        return {}
    return typing.cast(dict[str, object], value)


@dataclasses.dataclass(frozen=True)
class ConfigRow:
    """One editable config field, resolved against the value on disk.

    ``path`` is where the value lives in `config.yaml` -- `("globals", group,
    key)` or `("phases", phase, key)` -- which is both what the form writes back
    and what makes a row identifiable without depending on its position.
    """

    section: str
    path: tuple[str, ...]
    field: containermod.schema.FieldSchema
    value: object


def _section_rows(
    section: str,
    prefix: tuple[str, ...],
    groups: list[containermod.schema.FieldGroup],
    values: dict[str, object],
    flat: bool,
) -> list[ConfigRow]:
    rows: list[ConfigRow] = []
    for group in groups:
        source = values if flat else _mapping(values.get(group.name))
        for field in group.fields:
            rows.append(
                ConfigRow(
                    section=section if flat else f"{section}: {group.label}",
                    path=prefix + (() if flat else (group.name,)) + (field.key,),
                    field=field,
                    value=source.get(field.key, field.default),
                )
            )
    return rows


def config_rows(
    schema: containermod.schema.ConfigSchema,
    values: dict[str, object],
    phase_order: list[str],
) -> list[ConfigRow]:
    """Every field the image declares, flattened into rows a form can mount.

    Globals nest one level per group and phases are flat, mirroring what
    `runs._write_config` lays out. Phases come in the image's own order rather
    than the schema dict's, so the form reads down the pipeline.
    """
    rows = _section_rows(
        "globals",
        ("globals",),
        schema.globals.groups,
        _mapping(values.get("globals")),
        flat=False,
    )
    phase_values = _mapping(values.get("phases"))
    ordered = [p for p in phase_order if p in schema.phases]
    ordered += [p for p in schema.phases if p not in ordered]
    for phase in ordered:
        rows.extend(
            _section_rows(
                f"phase: {phase}",
                ("phases", phase),
                schema.phases[phase].groups,
                _mapping(phase_values.get(phase)),
                flat=True,
            )
        )
    return rows


def config_values(edited: list[tuple[tuple[str, ...], object]]) -> dict[str, object]:
    """Nest `(path, value)` pairs back into the shape `config.yaml` has.

    The inverse of `config_rows`, so the form never has to know that globals are
    grouped and phases are not -- it hands back the paths it was given.
    """
    out: dict[str, object] = {}
    for path, value in edited:
        cursor = out
        for key in path[:-1]:
            nested = _mapping(cursor.get(key))
            cursor[key] = nested
            cursor = nested
        cursor[path[-1]] = value
    return out
