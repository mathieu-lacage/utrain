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

# The images and compute tables, as `image list` and `compute list` print them.
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
# queued / running / done / failed / stopped (`queued` is a sweep run waiting
# for its compute), phase statuses pending / running / done /
# failed / stopped, and the two overlap enough that splitting them would only
# invite them to drift apart.
_STATUS_STYLES = {
    "configuring": "cyan",
    "queued": "dim",
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


# -- the runs tree ----------------------------------------------------------

# The Runs workspace's one list: sweeps, the runs in them and on their own, and
# a run's phases once it is expanded. A run's name is led by the mark it
# carries while it is in the tray -- in the name's cell rather than a column of
# its own, whose padding the 42-cell sidebar cannot spare. There is no id
# column: a run is addressed by name on screen, and by the goto line otherwise.
TREE_COLUMNS = ("NAME", "POINT", "STATUS")

MARK = "◆"
EXPANDED = "▾"
COLLAPSED = "▸"

# What a sweep's status grid draws for each run status.
STATUS_GLYPHS = {
    "done": "●",
    "running": "◐",
    "queued": "○",
    "failed": "✗",
    "stopped": "■",
    "configuring": "·",
}


@dataclasses.dataclass(frozen=True)
class TreeNode:
    """One row of the runs tree.

    `key` names the thing rather than the row -- `sweep:<id>`, `run:<id>`,
    `phase:<run id>/<phase>` -- so that the cursor can follow it when rows are
    inserted above it by an expand or by a new run.
    """

    key: str
    kind: typing.Literal["sweep", "run", "phase"]
    depth: int
    # Whether the row is open, or None for one that has nothing under it.
    expanded: bool | None
    run: types.RunRow | None = None
    sweep: types.SweepRow | None = None
    phase: types.PhaseListEntry | None = None

    @property
    def run_id(self) -> str | None:
        return self.run.id if self.run is not None else None


def sweep_key(sweep_id: str) -> str:
    return f"sweep:{sweep_id}"


def run_key(run_id: str) -> str:
    return f"run:{run_id}"


def phase_key(run_id: str, phase: str) -> str:
    return f"phase:{run_id}/{phase}"


def build_tree(
    runs: list[types.RunRow],
    sweeps: list[types.SweepRow],
    expanded: set[str],
    phases: dict[str, list[types.PhaseListEntry]],
) -> list[TreeNode]:
    """The runs tree, flattened into rows in reading order.

    Sweeps first, newest first, each followed by its runs in grid order when it
    is open; then the runs that belong to no sweep, newest first, as the list
    has always had them. A run can be opened on its phases once it has an
    attempt: before that it has none.
    """
    nodes: list[TreeNode] = []

    def add_run(run: types.RunRow, depth: int) -> None:
        key = run_key(run.id)
        has_phases = run.attempt is not None
        is_open = has_phases and key in expanded
        nodes.append(TreeNode(key, "run", depth, is_open if has_phases else None, run=run))
        if is_open:
            for entry in phases.get(run.id, []):
                nodes.append(
                    TreeNode(
                        phase_key(run.id, entry.phase),
                        "phase",
                        depth + 1,
                        None,
                        run=run,
                        phase=entry,
                    )
                )

    by_sweep: dict[str, list[types.RunRow]] = {}
    for run in runs:
        if run.sweep_id is not None:
            by_sweep.setdefault(run.sweep_id, []).append(run)
    for sweep in sweeps:
        key = sweep_key(sweep.id)
        members = sorted(by_sweep.get(sweep.id, []), key=lambda r: r.created_at)
        is_open = key in expanded
        nodes.append(TreeNode(key, "sweep", 0, is_open if members else None, sweep=sweep))
        if is_open:
            for run in members:
                add_run(run, 1)
    known = {s.id for s in sweeps}
    for run in runs:
        if run.sweep_id is None or run.sweep_id not in known:
            add_run(run, 0)
    return nodes


def _expander(node: TreeNode) -> str:
    if node.expanded is None:
        return "  "
    return f"{EXPANDED if node.expanded else COLLAPSED} "


def point_text(point: dict[str, object]) -> str:
    """A sweep run's coordinates, short: each axis by its last name."""
    parts: list[str] = []
    for path, value in point.items():
        if isinstance(value, bool):
            shown = "true" if value else "false"
        elif isinstance(value, float):
            shown = f"{value:g}"
        else:
            shown = str(value)
        parts.append(f"{path.rsplit('.', 1)[-1]}={shown}")
    return " ".join(parts)


def sweep_progress(sweep: types.SweepRow) -> str:
    """How far a sweep has got: finished runs over all of them."""
    return f"{sweep.counts.finished}/{sweep.counts.total}"


def tree_cells(node: TreeNode, marked: set[str], now: float) -> list[Cell]:
    indent = "  " * node.depth
    if node.kind == "sweep":
        sweep = node.sweep
        assert sweep is not None
        return [
            f" {indent}{_expander(node)}{sweep.name}",
            sweep_progress(sweep),
            status_cell(sweep.status, _SWEEP_STATUS_STYLE.get(sweep.status)),
        ]
    if node.kind == "phase":
        entry = node.phase
        assert entry is not None
        status = entry.status if entry.status is not None else "--"
        return [
            f" {indent}  {entry.phase}",
            format_duration(entry.started_at, entry.ended_at, now),
            status_cell(status, entry.status),
        ]
    run = node.run
    assert run is not None
    return [
        f"{MARK if run.id in marked else ' '}{indent}{_expander(node)}{run.name}",
        point_text(run.point),
        status_cell(run.status, run.status),
    ]


# A sweep's own statuses, drawn in the run colours they correspond to.
_SWEEP_STATUS_STYLE = {
    "draft": "configuring",
    "running": "running",
    "paused": "queued",
    "done": "done",
    "cancelled": "stopped",
}


def sweep_grid(
    sweep: types.SweepRow,
    runs: list[types.RunRow],
    values: dict[str, str] | None = None,
) -> rich.text.Text:
    """A sweep's runs as a grid: the first axis down, the second across.

    Each cell is a run's status glyph and, once there is one, the value
    `values` gives for it -- a reduced metric, which is what turns the monitor
    into the result as the sweep finishes. A sweep of one axis is one column;
    one of more than two draws every run of a cell's first two coordinates in
    it, a glyph each.
    """
    values = values or {}
    axes = list(sweep.axes)
    text = rich.text.Text()
    if not axes:
        return text
    rows_axis = axes[0]
    cols_axis = axes[1] if len(axes) > 1 else None
    row_values = sweep.axes[rows_axis]
    col_values = sweep.axes[cols_axis] if cols_axis is not None else [None]

    def label(path: str | None, value: object) -> str:
        if path is None:
            return ""
        return point_text({path: value})

    cells: dict[tuple[str, str], list[types.RunRow]] = {}
    for run in runs:
        key = (
            repr(run.point.get(rows_axis)),
            repr(run.point.get(cols_axis)) if cols_axis is not None else "None",
        )
        cells.setdefault(key, []).append(run)

    row_width = max(len(label(rows_axis, v)) for v in row_values) + 2
    col_width = max(
        [14]
        + [len(label(cols_axis, v)) + 2 for v in col_values]
        + [len(values.get(r.id, "")) + 4 for r in runs]
    )
    if cols_axis is not None:
        text.append(" " * row_width)
        for value in col_values:
            text.append(label(cols_axis, value).ljust(col_width), style="bold")
        text.append("\n")
    for row_value in row_values:
        text.append(label(rows_axis, row_value).ljust(row_width), style="bold")
        for col_value in col_values:
            members = cells.get((repr(row_value), repr(col_value)), [])
            cell = rich.text.Text()
            for run in members:
                cell.append(STATUS_GLYPHS.get(run.status, "?"), style=status_style(run.status))
            if len(members) == 1:
                shown = values.get(members[0].id) or members[0].status
                cell.append(f" {shown}")
            cell.pad_right(max(0, col_width - len(cell.plain)))
            text.append_text(cell)
        text.append("\n")
    text.append("\n")
    for status, glyph in STATUS_GLYPHS.items():
        text.append(f"{glyph} {status}  ", style=status_style(status))
    return text


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

    Step and elapsed time are always available -- every row carries both, and
    "step" means the phase's own ``step`` column when it logs one (see
    `build_plot`). Any logged metric can also serve as an x axis, which is how
    you plot one metric against another rather than against time.
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

    Against elapsed time this is a straight read of the metric's own points.
    Anything else is a join on ``_step``: the two series are sparse and
    independently sampled -- a phase may log ``loss`` every step and a
    validation metric every hundredth -- so only the steps they share can be
    paired, and there may be none. The step axis joins against the phase's own
    ``step`` column when it logs one -- wandb's ``_step`` counts log calls, and
    a phase logging its training step (nanochat does) means a viewer asking for
    ``step`` means that column, whose values are a hundred apart where ``_step``
    creeps up by one. A join against step that pairs nothing falls back to
    ``_step``, so the curve survives a phase whose ``step`` column does not sit
    on the rows its metrics are on; an explicit metric axis with nothing to pair
    draws nothing instead, because a curve against a metric it shares no step
    with would be a lie under that label.
    """
    series = points.get(name)
    if not series:
        return None

    if x_axis == X_ELAPSED:
        start = series[0].timestamp
        xs = [p.timestamp - start for p in series]
        ys = [p.value for p in series]
        x_label = "elapsed (s)"
    else:
        other = points.get(x_axis)
        pairs: list[tuple[float, float]] = []
        if other:
            by_step = {p.step: p.value for p in other}
            pairs = [(by_step[p.step], p.value) for p in series if p.step in by_step]
        if pairs:
            xs = [x for x, _ in pairs]
            ys = [y for _, y in pairs]
        elif x_axis == X_STEP:
            # No `step` column to pair against -- or none on the rows this
            # metric is logged on -- leaves wandb's own counter to plot
            # against. It is the only step a phase that logs none has.
            xs = [float(p.step) for p in series]
            ys = [p.value for p in series]
        else:
            return None
        x_label = "step" if x_axis == X_STEP else x_axis

    xs, ys = downsample(xs, ys, limit)
    return Plot(title=name, x_label=x_label, xs=xs, ys=ys, latest=format_value(series[-1].value))


def compute_options(info: compute.ComputeInfo) -> list[tuple[str, str]]:
    """(label, value) pairs for the new-run dialog's compute picker.

    The values are exactly what `runs.resolve_compute` accepts -- `cpu` and
    `gpu<index>` -- because it is the one that decides, and a dialog offering
    anything else would only be rejected on create.
    """
    options = [(f"cpu ({info.cpu.name}, {info.cpu.cores} cores)", "cpu")]
    for gpu in info.gpus:
        options.append((f"gpu{gpu.index} ({gpu.name})", f"gpu{gpu.index}"))
    return options


def default_compute(info: compute.ComputeInfo) -> str:
    """The compute a new run dialog opens on: the first gpu, or the cpu.

    A machine built for training usually has a gpu, and a run that should
    have had one and silently went to the cpu wastes a queue slot before
    anyone notices, so the picker starts on the gpu and moving to the cpu
    is the deliberate act. `compute_options` still lists the cpu first:
    this decides the selection, not the order. The first gpu rather than
    the least busy one because a second gpu on the host is usually busy
    with someone else's run, and picking among them would be a policy the
    dialog is not asked to have.
    """
    if info.gpus:
        return f"gpu{info.gpus[0].index}"
    return "cpu"


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


# The System workspace's compute table: `compute list`'s columns, then what is
# on each device now and how many sweep runs wait for it.
SYSTEM_COMPUTE_COLUMNS = (*COMPUTE_COLUMNS, "RUNNING", "QUEUED")


def system_compute_rows(info: compute.ComputeInfo, runs: list[types.RunRow]) -> list[list[Cell]]:
    """`compute_rows`, with the run on each device and its queue beside it."""
    running: dict[str, list[str]] = {}
    queued: dict[str, int] = {}
    for run in runs:
        if run.status == "running":
            running.setdefault(run.compute, []).append(run.name)
        elif run.status == "queued":
            queued[run.compute] = queued.get(run.compute, 0) + 1
    rows: list[list[Cell]] = []
    for row in compute_rows(info):
        device = row[0]
        names = running.get(device, [])
        rows.append(
            [
                *row,
                ", ".join(names) if names else "--",
                str(queued[device]) if device in queued else "--",
            ]
        )
    return rows


def store_line(summary: types.StoreSummary) -> str:
    """The data store in one line: its size, and what a cleanup would reclaim."""
    line = f"{summary.files} file(s), {human_size(summary.bytes)}"
    if summary.orphaned:
        line += (
            f"; {summary.orphaned} orphaned ({human_size(summary.orphaned_bytes)}), reclaimed by G"
        )
    return line


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


# -- the help screen ------------------------------------------------------

# Wide enough for the longest key spelling in `screens.help_sections`, so the
# descriptions line up in one column.
_HELP_KEY_WIDTH = 18


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
