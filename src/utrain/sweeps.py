"""Sweeps: a grid of runs over some config axes, started in turn.

A sweep is not a new kind of run. It is a name, an image frozen once for all of
its points, and a spec -- the axes and the values each takes -- and creating it
creates one ordinary run per point of the grid, with status `queued`. Every
existing command and screen works on those runs unchanged; what a sweep adds is
knowing which runs belong together and starting them in turn.

Compute is fixed when the sweep is created, as it is for any run: a run's
parameters can depend on its GPU (a batch size that fits its memory), so a run
must not be moved to another one. The runs are spread across the sweep's
computes round-robin, and each waits for its own.

Starting them is the dispatcher's job, as it is for every queued run (see
`dispatcher`): while the sweep is `running` it starts the next of its runs on
each of the sweep's computes that is free -- where free means no run at all is
running on it, so runs started by hand, and other sweeps' runs, are respected.
Pausing or cancelling the sweep holds its queued runs back.
"""

import copy
import dataclasses
import itertools
import json
import math
import pathlib
import re
import time
import typing
import uuid

import sqlalchemy
import sqlalchemy.orm
import yaml

from . import config, container, dispatcher, exceptions, orchestrator, runs, types
from . import db as dbmod

# What the user last asked of a sweep. `done` is not one of them: it is what a
# started sweep becomes once none of its runs is queued or running.
DRAFT = "draft"
RUNNING = "running"
PAUSED = "paused"
CANCELLED = "cancelled"
DONE = "done"

# A sweep's name is how it is addressed (`@lr-depth`) and the stem of its runs'
# names, so it is held to what reads well in both.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def name_problem(name: str) -> str | None:
    """Why `name` cannot name a sweep, or None if it can."""
    if not _NAME_RE.match(name):
        return (
            f"sweep name '{name}' must be letters, digits, '.', '_' or '-', "
            "starting with a letter or digit"
        )
    return None


@dataclasses.dataclass(frozen=True)
class Spec:
    """What a sweep varies, and where its runs go.

    `axes` holds concrete values, already checked against the image's schema:
    the ranges a spec file or the CLI may give (`{log: [1e-4, 1e-2], num: 5}`)
    are expanded by `expand_axis` before a `Spec` is made. Insertion order is
    grid order, the first axis varying slowest.
    """

    axes: dict[str, list[object]]
    replicate: list[str]
    compute: list[str]
    base: str | None

    def to_json(self) -> str:
        return json.dumps(
            {
                "axes": self.axes,
                "replicate": self.replicate,
                "compute": self.compute,
                "base": self.base,
            }
        )

    @classmethod
    def from_json(cls, text: str) -> "Spec":
        data = _mapping(json.loads(text))
        axes = {str(k): list(_sequence(v)) for k, v in _mapping(data.get("axes")).items()}
        return cls(
            axes=axes,
            replicate=[str(r) for r in _sequence(data.get("replicate"))],
            compute=[str(c) for c in _sequence(data.get("compute"))],
            base=str(data["base"]) if data.get("base") is not None else None,
        )

    def size(self) -> int:
        return math.prod(len(v) for v in self.axes.values()) if self.axes else 0


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return {str(k): v for k, v in typing.cast(dict[object, object], value).items()}


def _sequence(value: object) -> list[object]:
    if isinstance(value, (list, tuple)):
        return list(typing.cast(list[object], value))
    return []


# -- axes -------------------------------------------------------------------


def field_for(path: str, schema: container.schema.ConfigSchema) -> container.schema.FieldSchema:
    """The config field an axis path names, or a UI error saying why none does.

    Paths follow the config file's own shape: `globals.<group>.<key>` for a
    field that applies to every phase, `phases.<phase>.<key>` for one phase's.
    """
    parts = path.split(".")
    if len(parts) != 3 or parts[0] not in ("globals", "phases"):
        raise exceptions.UI(f"axis '{path}' must be globals.<group>.<key> or phases.<phase>.<key>")
    section, middle, key = parts
    if section == "globals":
        groups = [g for g in schema.globals.groups if g.name == middle]
        if not groups:
            raise exceptions.UI(f"axis '{path}': the image has no global group '{middle}'")
    else:
        phase = schema.phases.get(middle)
        if phase is None:
            raise exceptions.UI(f"axis '{path}': the image has no configurable phase '{middle}'")
        groups = phase.groups
    for group in groups:
        for field in group.fields:
            if field.key == key:
                return field
    raise exceptions.UI(f"axis '{path}': no field '{key}' there")


def _range(lo: float, hi: float, num: int, log: bool) -> list[float]:
    if num < 1:
        raise exceptions.UI("a range needs at least one value")
    if num == 1:
        return [lo]
    if log:
        if lo <= 0 or hi <= 0:
            raise exceptions.UI("a log range needs positive bounds")
        a, b = math.log10(lo), math.log10(hi)
        return [10 ** (a + i * (b - a) / (num - 1)) for i in range(num)]
    return [lo + i * (hi - lo) / (num - 1) for i in range(num)]


def _tidy(value: float) -> float:
    """A computed range value, rounded to what a person would have typed.

    `10 ** -3.5` is 0.00031622776601683794; six significant digits keep the
    grid exact enough to train on and short enough to read in a column.
    """
    return float(f"{value:.6g}")


def expand_axis(raw: object, field: container.schema.FieldSchema) -> list[object]:
    """The concrete values one axis takes, checked against its field.

    `raw` is what a spec file or the CLI gave for the axis:

    - a list of values;
    - `{values: [...]}`, the same with room for other keys (`replicate`);
    - `{lin: [lo, hi], num: n}` or `{log: [lo, hi], num: n}` for a numeric
      field; an int field's values are rounded and repeats dropped;
    - nothing at all (None, or an empty list), for a bool or enum field: every
      value it can take.
    """
    spec = _mapping(raw) if isinstance(raw, dict) else {}
    if isinstance(raw, dict) and "values" in spec:
        raw = spec["values"]

    values: list[object]
    if isinstance(raw, dict) and ("lin" in spec or "log" in spec):
        if field.type not in ("int", "float"):
            raise exceptions.UI(f"'{field.key}' is {field.type}; a range needs a number field")
        is_log = "log" in spec
        bounds = _sequence(spec["log" if is_log else "lin"])
        if len(bounds) != 2:
            raise exceptions.UI(f"'{field.key}': a range is [low, high]")
        try:
            lo, hi = float(str(bounds[0])), float(str(bounds[1]))
            num = int(str(spec.get("num", 5)))
        except ValueError:
            raise exceptions.UI(f"'{field.key}': a range's bounds and num must be numbers")
        values = [_tidy(v) for v in _range(lo, hi, num, is_log)]
        if field.type == "int":
            values = [int(round(float(str(v)))) for v in values]
    elif raw is None or raw == []:
        if field.type == "bool":
            values = [False, True]
        elif field.type == "enum" and field.options:
            values = list(field.options)
        else:
            raise exceptions.UI(f"'{field.key}': give the values to sweep")
    elif isinstance(raw, (list, tuple)):
        values = _sequence(raw)
    else:
        values = [raw]

    out: list[object] = []
    for value in values:
        if field.type == "bool" and isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("true", "false"):
                value = lowered == "true"
        coerced = runs.coerce_value(field, value)
        if coerced not in out:
            out.append(coerced)
    if not out:
        raise exceptions.UI(f"'{field.key}': an axis needs at least one value")
    return out


def parse_axis_arg(arg: str) -> tuple[str, object]:
    """One `--axis PATH=VALUES` from the command line, as a raw axis.

    VALUES is a comma-separated list (`1e-4,3e-4`), a range
    (`log:1e-4:1e-2:5`, `lin:0:1:3`), or `*` for every value of a bool or enum
    field. The values stay strings: `expand_axis` coerces them against the
    field, which is the only thing that knows their type.
    """
    if "=" not in arg:
        raise exceptions.UI(f"expected PATH=VALUES, got '{arg}'")
    path, _, text = arg.partition("=")
    path, text = path.strip(), text.strip()
    if text == "*":
        return path, None
    for kind in ("lin", "log"):
        if text.startswith(f"{kind}:"):
            parts = text.split(":")[1:]
            if len(parts) not in (2, 3):
                raise exceptions.UI(f"'{arg}': a range is {kind}:LOW:HIGH[:NUM]")
            raw: dict[str, object] = {kind: parts[:2]}
            if len(parts) == 3:
                raw["num"] = parts[2]
            return path, raw
    return path, [v.strip() for v in text.split(",") if v.strip()]


def grid(axes: dict[str, list[object]]) -> list[dict[str, object]]:
    """Every point of the grid, the first axis varying slowest."""
    paths = list(axes)
    return [dict(zip(paths, combo)) for combo in itertools.product(*axes.values())]


def _point_key(point: dict[str, object]) -> str:
    """A point, as a string that is equal exactly when the points are.

    What `extend` matches existing runs on. `sweep_point` itself keeps the
    sweep's axis order instead, which is the order a point is read in.
    """
    return json.dumps(point, sort_keys=True)


def overlay(values: dict[str, object], point: dict[str, object]) -> dict[str, object]:
    """`values` (a nested config) with each of `point`'s paths set."""
    out = copy.deepcopy(values)
    for path, value in point.items():
        section, middle, key = path.split(".")
        node = out.setdefault(section, {})
        assert isinstance(node, dict)
        inner = typing.cast(dict[str, object], node).setdefault(middle, {})
        assert isinstance(inner, dict)
        typing.cast(dict[str, object], inner)[key] = value
    return out


# -- creation ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _Image:
    name: str
    image_id: str
    described: container.schema.DescribeOutput


def _image(
    image: str | None, base: str | None, session: sqlalchemy.orm.Session
) -> tuple[_Image, dict[str, object], str | None]:
    """The image a new sweep runs, the config its points start from, and its base.

    From a base run, all three come from the run: its image name, the id it
    froze, and its config. Otherwise the image is resolved from its name, as
    `run create` does, and the points start from the image's defaults.
    """
    if base is not None:
        base_id = dbmod.resolve_run_id(base, session)
        row = dbmod.get_run(base_id, session)
        if image is not None and image != str(row["image"]):
            raise exceptions.UI(
                f"run '{base}' uses image '{row['image']}', not '{image}'; give one or the other"
            )
        described = container.podman.describe(dbmod.run_image_ref(row))
        values = runs.read_config(base_id, session)
        return _Image(str(row["image"]), str(row["image_id"]), described), values, base_id

    if image is None:
        raise exceptions.UI("a sweep needs an image, or a run to start from")
    presets = container.podman.list_presets()
    if image not in presets:
        raise exceptions.UI(f"image '{image}' not found")
    image_id = container.podman.image_id(presets[image])
    described = container.podman.describe(image_id)
    return _Image(image, image_id, described), {}, None


def create_sweep(
    name: str,
    axes: dict[str, object],
    compute: list[str],
    settings: config.Settings,
    session: sqlalchemy.orm.Session,
    image: str | None = None,
    base: str | None = None,
    replicate: list[str] | None = None,
) -> str:
    """Create a sweep and one queued run per point of its grid.

    `axes` maps each axis path to its raw values (see `expand_axis`). Nothing is
    started: `start_sweep` hands the sweep to a dispatcher.
    """
    problem = name_problem(name)
    if problem is not None:
        raise exceptions.UI(problem)
    taken = session.execute(
        sqlalchemy.select(dbmod.sweeps.c.id).where(dbmod.sweeps.c.name == name)
    ).first()
    if taken is not None:
        raise exceptions.UI(f"a sweep named '{name}' already exists")
    if not axes:
        raise exceptions.UI("a sweep needs at least one axis")
    if not compute:
        raise exceptions.UI("a sweep needs at least one compute")

    img, base_values, base_id = _image(image, base, session)
    if not img.described.phase_order:
        raise exceptions.UI(f"image '{img.name}' has no phases")
    schema = img.described.config_schema

    expanded = {path: expand_axis(raw, field_for(path, schema)) for path, raw in axes.items()}
    replicate = list(replicate or [])
    for path, raw in axes.items():
        if isinstance(raw, dict) and _mapping(raw).get("replicate") and path not in replicate:
            replicate.append(path)
    for path in replicate:
        if path not in expanded:
            raise exceptions.UI(f"replicate axis '{path}' is not one of the sweep's axes")

    # Checked once, here: a run's compute is fixed at creation, and so is each
    # of these runs'.
    computes = [runs.resolve_compute(c) for c in compute]
    spec = Spec(axes=expanded, replicate=replicate, compute=computes, base=base_id)

    sweep_id = uuid.uuid4().hex
    session.execute(
        sqlalchemy.insert(dbmod.sweeps).values(
            id=sweep_id,
            name=name,
            image=img.name,
            image_id=img.image_id,
            spec=spec.to_json(),
            state=DRAFT,
            created_at=time.time(),
        )
    )
    _create_points(sweep_id, name, img, spec, base_values, grid(spec.axes), 0, settings, session)
    return sweep_id


def _create_points(
    sweep_id: str,
    name: str,
    img: _Image,
    spec: Spec,
    base_values: dict[str, object],
    points: list[dict[str, object]],
    first_index: int,
    settings: config.Settings,
    session: sqlalchemy.orm.Session,
) -> None:
    """One queued run per point, numbered and assigned compute from `first_index`.

    Runs are named `<sweep>-<nn>` and spread across the sweep's computes by
    their index, so a sweep that is extended keeps numbering and spreading
    where it left off.
    """
    width = max(2, len(str(first_index + len(points))))
    now = time.time()
    for offset, point in enumerate(points):
        index = first_index + offset
        compute = spec.compute[index % len(spec.compute)]
        run_id = uuid.uuid4().hex
        run_dir = settings.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg = runs.validated_config(
            run_id, compute, overlay(base_values, point), img.described.config_schema
        )
        (run_dir / "config.yaml").write_text(
            yaml.dump(cfg, default_flow_style=False, sort_keys=False)
        )
        session.execute(
            sqlalchemy.insert(dbmod.runs).values(
                id=run_id,
                name=f"{name}-{index + 1:0{width}d}",
                image=img.name,
                image_id=img.image_id,
                compute=compute,
                status="queued",
                config_hash=None,
                # Strictly increasing, so that creation order -- grid order --
                # is also what `created_at` sorts by.
                created_at=now + index * 1e-6,
                queued_at=now + index * 1e-6,
                sweep_id=sweep_id,
                sweep_point=json.dumps(point),
            )
        )


# -- reading ----------------------------------------------------------------


def resolve_sweep_id(ref: str, session: sqlalchemy.orm.Session) -> str:
    """A sweep's id, from its name or a unique prefix of its id.

    A leading `@` is accepted and ignored, so the address form the TUI's goto
    line uses works here too. The name wins: a name is what a person types.
    """
    ref = ref.removeprefix("@")
    row = session.execute(
        sqlalchemy.select(dbmod.sweeps.c.id).where(dbmod.sweeps.c.name == ref)
    ).first()
    if row is not None:
        return str(row[0])
    ids = (
        session.execute(
            sqlalchemy.select(dbmod.sweeps.c.id).where(dbmod.sweeps.c.id.like(f"{ref}%"))
        )
        .scalars()
        .fetchall()
    )
    if not ids:
        raise exceptions.UI(f"sweep '{ref}' not found")
    if len(ids) > 1:
        raise exceptions.UI(f"sweep id prefix '{ref}' is ambiguous")
    return str(ids[0])


def _row(sweep_id: str, session: sqlalchemy.orm.Session) -> sqlalchemy.engine.RowMapping:
    row = (
        session.execute(sqlalchemy.select(dbmod.sweeps).where(dbmod.sweeps.c.id == sweep_id))
        .mappings()
        .fetchone()
    )
    if row is None:
        raise exceptions.UI(f"sweep '{sweep_id}' not found")
    return row


def _run_statuses(sweep_id: str, session: sqlalchemy.orm.Session) -> list[str]:
    return [
        str(s)
        for s in session.execute(
            sqlalchemy.select(dbmod.runs.c.status).where(dbmod.runs.c.sweep_id == sweep_id)
        )
        .scalars()
        .fetchall()
    ]


def _counts(statuses: list[str]) -> types.SweepCounts:
    return types.SweepCounts(
        total=len(statuses),
        queued=statuses.count("queued"),
        running=statuses.count("running"),
        done=statuses.count("done"),
        failed=statuses.count("failed"),
        stopped=statuses.count("stopped"),
    )


def status_of(state: str, counts: types.SweepCounts) -> str:
    """What a sweep is doing: its state, or `done` once there is nothing left."""
    if state in (RUNNING, PAUSED) and counts.queued == 0 and counts.running == 0:
        return DONE
    return state


def _sweep_row(
    row: sqlalchemy.engine.RowMapping, session: sqlalchemy.orm.Session
) -> types.SweepRow:
    spec = Spec.from_json(str(row["spec"]))
    counts = _counts(_run_statuses(str(row["id"]), session))
    return types.SweepRow(
        id=str(row["id"]),
        name=str(row["name"]),
        image=str(row["image"]),
        image_id=str(row["image_id"]),
        status=status_of(str(row["state"]), counts),
        created_at=float(row["created_at"]),
        axes=spec.axes,
        replicate=spec.replicate,
        compute=spec.compute,
        base=spec.base,
        counts=counts,
    )


def list_sweeps(session: sqlalchemy.orm.Session) -> list[types.SweepRow]:
    """Every sweep, newest first."""
    dispatcher.ensure(session)
    rows = (
        session.execute(sqlalchemy.select(dbmod.sweeps).order_by(dbmod.sweeps.c.created_at.desc()))
        .mappings()
        .fetchall()
    )
    return [_sweep_row(r, session) for r in rows]


def list_sweep_ids(session: sqlalchemy.orm.Session) -> list[str]:
    return [
        str(i)
        for i in session.execute(
            sqlalchemy.select(dbmod.sweeps.c.id).order_by(dbmod.sweeps.c.created_at.desc())
        )
        .scalars()
        .fetchall()
    ]


def get_sweep(ref: str, session: sqlalchemy.orm.Session) -> types.SweepDetail:
    sweep_id = resolve_sweep_id(ref, session)
    dispatcher.ensure(session)
    row = _row(sweep_id, session)
    run_rows = (
        session.execute(
            sqlalchemy.select(dbmod.runs)
            .where(dbmod.runs.c.sweep_id == sweep_id)
            .order_by(dbmod.runs.c.created_at)
        )
        .mappings()
        .fetchall()
    )
    return types.SweepDetail(
        sweep=_sweep_row(row, session),
        runs=[runs.get_run(str(r["id"]), session) for r in run_rows],
    )


# -- lifecycle --------------------------------------------------------------


def _set_state(sweep_id: str, state: str, session: sqlalchemy.orm.Session) -> None:
    session.execute(
        sqlalchemy.update(dbmod.sweeps).where(dbmod.sweeps.c.id == sweep_id).values(state=state)
    )


def _to_back_of_queue(sweep_id: str, session: sqlalchemy.orm.Session) -> None:
    """Queue the sweep's queued runs behind every run already waiting.

    A sweep's runs join the queue when it starts, not when they were created,
    and keep grid order among themselves.
    """
    now = time.time()
    queued = (
        session.execute(
            sqlalchemy.select(dbmod.runs.c.id)
            .where((dbmod.runs.c.sweep_id == sweep_id) & (dbmod.runs.c.status == "queued"))
            .order_by(dbmod.runs.c.created_at)
        )
        .scalars()
        .fetchall()
    )
    for i, run_id in enumerate(queued):
        session.execute(
            sqlalchemy.update(dbmod.runs)
            .where(dbmod.runs.c.id == run_id)
            .values(queued_at=now + i * 1e-6)
        )


def start_sweep(ref: str, session: sqlalchemy.orm.Session) -> str:
    """Hand a draft or paused sweep to a dispatcher. Also how a pause is resumed."""
    sweep_id = resolve_sweep_id(ref, session)
    state = str(_row(sweep_id, session)["state"])
    if state == CANCELLED:
        raise exceptions.UI("sweep was cancelled; 'sweep retry' requeues its runs")
    _preflight(sweep_id, session)
    _set_state(sweep_id, RUNNING, session)
    _to_back_of_queue(sweep_id, session)
    dispatcher.ensure(session)
    return sweep_id


def _preflight(sweep_id: str, session: sqlalchemy.orm.Session) -> None:
    """Refuse, in the foreground, what would fail every run in the background.

    `run start` checks for the GPU toolkit before anything is created so that
    the user sees why (issue #26); a sweep's runs are started by a detached
    dispatcher whose only output is its log, so the same check belongs here.
    """
    for compute in Spec.from_json(str(_row(sweep_id, session)["spec"])).compute:
        orchestrator.ensure_gpu_toolkit(compute)


def pause_sweep(ref: str, session: sqlalchemy.orm.Session) -> str:
    """Stop starting new runs. Runs already going carry on."""
    sweep_id = resolve_sweep_id(ref, session)
    state = str(_row(sweep_id, session)["state"])
    if state not in (RUNNING, PAUSED):
        raise exceptions.UI(f"sweep is {state}; only a running sweep can be paused")
    _set_state(sweep_id, PAUSED, session)
    return sweep_id


def cancel_sweep(ref: str, session: sqlalchemy.orm.Session) -> str:
    """Stop the sweep's running runs and drop its queued ones.

    Dropped runs are kept, as `stopped` runs that never started, rather than
    deleted: the grid stays whole, and `sweep retry` can queue them again. A
    running run reads `stopped` once its orchestrator has exited.
    """
    sweep_id = resolve_sweep_id(ref, session)
    _set_state(sweep_id, CANCELLED, session)
    for run_id in (
        session.execute(
            sqlalchemy.select(dbmod.runs.c.id).where(
                (dbmod.runs.c.sweep_id == sweep_id) & dbmod.runs.c.status.in_(["running", "queued"])
            )
        )
        .scalars()
        .fetchall()
    ):
        runs.stop_run(str(run_id), session)
    return sweep_id


def retry_sweep(ref: str, session: sqlalchemy.orm.Session) -> tuple[str, int]:
    """Queue the sweep's failed and stopped runs again, and restart dispatching.

    Returns the sweep id and how many runs were queued. A requeued run that
    already ran is restarted from its first phase when its turn comes, as a new
    attempt; one that never started is started.
    """
    sweep_id = resolve_sweep_id(ref, session)
    _preflight(sweep_id, session)
    retryable = (dbmod.runs.c.sweep_id == sweep_id) & (
        dbmod.runs.c.status.in_(["failed", "stopped"])
    )
    n = len(session.execute(sqlalchemy.select(dbmod.runs.c.id).where(retryable)).fetchall())
    session.execute(
        sqlalchemy.update(dbmod.runs)
        .where(retryable)
        .values(status="queued", queued_from_phase=None)
    )
    _set_state(sweep_id, RUNNING, session)
    _to_back_of_queue(sweep_id, session)
    dispatcher.ensure(session)
    return sweep_id, n


def extend_sweep(
    ref: str,
    axes: dict[str, object],
    settings: config.Settings,
    session: sqlalchemy.orm.Session,
) -> tuple[str, int]:
    """Add values to existing axes, and a queued run for each new point.

    Existing points keep their runs, matched by `sweep_point`. Returns the
    sweep id and how many runs were added.
    """
    sweep_id = resolve_sweep_id(ref, session)
    row = _row(sweep_id, session)
    spec = Spec.from_json(str(row["spec"]))
    image_ref = str(row["image_id"])
    if not container.podman.image_exists(image_ref):
        raise exceptions.UI(f"image '{row['image']}' of sweep '{row['name']}' is not in the store")
    described = container.podman.describe(image_ref)
    schema = described.config_schema

    new_axes = {path: list(values) for path, values in spec.axes.items()}
    for path, raw in axes.items():
        if path not in new_axes:
            raise exceptions.UI(
                f"'{path}' is not an axis of sweep '{row['name']}'; "
                "extend adds values to existing axes"
            )
        for value in expand_axis(raw, field_for(path, schema)):
            if value not in new_axes[path]:
                new_axes[path].append(value)

    existing = {
        _point_key(_mapping(json.loads(str(p))))
        for p in session.execute(
            sqlalchemy.select(dbmod.runs.c.sweep_point).where(dbmod.runs.c.sweep_id == sweep_id)
        )
        .scalars()
        .fetchall()
    }
    points = [p for p in grid(new_axes) if _point_key(p) not in existing]
    new_spec = dataclasses.replace(spec, axes=new_axes)
    session.execute(
        sqlalchemy.update(dbmod.sweeps)
        .where(dbmod.sweeps.c.id == sweep_id)
        .values(spec=new_spec.to_json())
    )
    base_values = runs.read_config(spec.base, session) if spec.base else {}
    img = _Image(str(row["image"]), image_ref, described)
    _create_points(
        sweep_id,
        str(row["name"]),
        img,
        new_spec,
        base_values,
        points,
        len(existing),
        settings,
        session,
    )
    dispatcher.ensure(session)
    return sweep_id, len(points)


def delete_sweep(ref: str, force: bool, session: sqlalchemy.orm.Session) -> tuple[str, int]:
    """Delete a sweep and every one of its runs. Returns the id and the run count."""
    sweep_id = resolve_sweep_id(ref, session)
    run_ids = [
        (str(i), str(s))
        for i, s in session.execute(
            sqlalchemy.select(dbmod.runs.c.id, dbmod.runs.c.status).where(
                dbmod.runs.c.sweep_id == sweep_id
            )
        ).fetchall()
    ]
    if not force and any(s == "running" for _, s in run_ids):
        raise exceptions.UI("sweep has running runs; use --force or cancel it first")
    # Stop dispatching before deleting anything, so no run starts under us.
    _set_state(sweep_id, CANCELLED, session)
    for run_id, _ in run_ids:
        runs.delete_run(run_id, force=True, session=session)
    session.execute(sqlalchemy.delete(dbmod.sweeps).where(dbmod.sweeps.c.id == sweep_id))
    return sweep_id, len(run_ids)


# -- spec files -------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SpecFile:
    """A sweep as a YAML file gives it, before anything is checked."""

    name: str | None
    image: str | None
    base: str | None
    axes: dict[str, object]
    compute: list[str]
    replicate: list[str]


def read_spec_file(path: pathlib.Path) -> SpecFile:
    """Parse a sweep spec file. Checking it is `create_sweep`'s job."""
    try:
        data = _mapping(yaml.safe_load(path.read_text()))
    except OSError as e:
        raise exceptions.UI(f"cannot read '{path}': {e.strerror}")
    except yaml.YAMLError as e:
        raise exceptions.UI(f"'{path}' is not valid YAML: {e}")
    compute = data.get("compute")
    base = data.get("base")
    return SpecFile(
        name=str(data["name"]) if data.get("name") is not None else None,
        image=str(data["image"]) if data.get("image") is not None else None,
        base=str(base) if base is not None else None,
        axes=_mapping(data.get("axes")),
        compute=[str(c) for c in _sequence(compute)]
        if isinstance(compute, list)
        else ([str(compute)] if compute is not None else []),
        replicate=[str(r) for r in _sequence(data.get("replicate"))],
    )
