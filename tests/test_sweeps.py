"""Unit tests for sweeps: axes, grid expansion, creation and dispatching.

Podman is stubbed and the dispatcher's `start` is replaced by a recorder, so
nothing here starts a container: what is under test is which runs a sweep
creates, with what config, and which of them a tick of the dispatcher picks.
The dispatcher's own rules, whatever the runs, are in `test_dispatcher`.
"""

import json
import pathlib
import typing

import pytest
import sqlalchemy
import sqlalchemy.orm
import yaml

import utrain.cli.main
import utrain.cli.render
import utrain.config
import utrain.container.podman
import utrain.container.schema
import utrain.db
import utrain.dispatcher
import utrain.exceptions
import utrain.lock
import utrain.orchestrator
import utrain.sweeps

FROZEN_ID = "ab" * 32
# Taken before any fixture stubs it, for the one test about the check itself.
_REAL_TOOLKIT_CHECK = utrain.orchestrator.ensure_gpu_toolkit
S = utrain.container.schema


def _describe() -> S.DescribeOutput:
    return S.DescribeOutput(
        name="img",
        phases=[S.PhaseInfo(name="pretrain", label="Train")],
        phase_order=["pretrain"],
        config_schema=S.ConfigSchema(
            globals=S.GlobalConfigSchema(
                groups=[
                    S.FieldGroup(
                        name="model",
                        label="Model",
                        fields=[
                            S.FieldSchema(
                                key="n_layer", label="Layers", type="int", default=4, min=1, max=48
                            ),
                            S.FieldSchema(
                                key="dtype",
                                label="Precision",
                                type="enum",
                                default="fp32",
                                options=["fp32", "bf16"],
                            ),
                        ],
                    )
                ]
            ),
            phases={
                "pretrain": S.PhaseConfigSchema(
                    groups=[
                        S.FieldGroup(
                            name="optim",
                            label="Optimiser",
                            fields=[
                                S.FieldSchema(
                                    key="lr", label="Learning rate", type="float", default=0.001
                                ),
                                S.FieldSchema(
                                    key="resume", label="Resume", type="bool", default=False
                                ),
                            ],
                        )
                    ]
                )
            },
        ),
    )


def _field(path: str) -> S.FieldSchema:
    return utrain.sweeps.field_for(path, _describe().config_schema)


@pytest.fixture()
def podman(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """An image store holding `img`, and a host with a CPU and two GPUs."""
    monkeypatch.setattr(utrain.container.podman, "list_presets", lambda: {"img": "localhost/img"})
    monkeypatch.setattr(utrain.container.podman, "image_id", lambda ref: FROZEN_ID)
    monkeypatch.setattr(utrain.container.podman, "image_exists", lambda ref: True)
    monkeypatch.setattr(utrain.container.podman, "describe", lambda ref: _describe())
    fixture = tmp_path / "compute.json"
    fixture.write_text(
        json.dumps(
            {
                "cpu": {"name": "cpu", "cores": 8, "mem_total_gb": 32, "mem_available_gb": 16},
                "gpus": [
                    {
                        "index": i,
                        "name": "gpu",
                        "power_draw": 1,
                        "power_limit": 2,
                        "util": 0,
                        "mem_used_mb": 1,
                        "mem_total_mb": 2,
                    }
                    for i in (0, 1)
                ],
            }
        )
    )
    monkeypatch.setenv("UTRAIN_COMPUTE_FIXTURE", str(fixture))
    # The GPU toolkit check has its own test below; everything else here is
    # about which runs start, not whether the host could run them.
    monkeypatch.setattr(utrain.orchestrator, "ensure_gpu_toolkit", lambda compute: None)


@pytest.fixture()
def settings(tmp_path: pathlib.Path) -> utrain.config.Settings:
    return utrain.config.Settings(data_dir=tmp_path / "data")


@pytest.fixture()
def session(settings: utrain.config.Settings) -> typing.Iterator[sqlalchemy.orm.Session]:
    with utrain.db.with_db(settings) as s:
        yield s


def _create(
    session: sqlalchemy.orm.Session,
    settings: utrain.config.Settings,
    axes: dict[str, object] | None = None,
    compute: list[str] | None = None,
    **kwargs: typing.Any,
) -> str:
    return utrain.sweeps.create_sweep(
        name=kwargs.pop("name", "lr-depth"),
        axes=axes
        if axes is not None
        else {"phases.pretrain.lr": [1e-4, 3e-4], "globals.model.n_layer": [4, 8]},
        compute=compute if compute is not None else ["gpu0", "gpu1"],
        settings=settings,
        session=session,
        image=kwargs.pop("image", "img"),
        **kwargs,
    )


def _statuses(session: sqlalchemy.orm.Session, sweep_id: str) -> list[str]:
    return [
        str(s)
        for s in session.execute(
            sqlalchemy.select(utrain.db.runs.c.status)
            .where(utrain.db.runs.c.sweep_id == sweep_id)
            .order_by(utrain.db.runs.c.created_at)
        ).scalars()
    ]


# -- axes -----------------------------------------------------------------


def test_an_axis_path_names_a_global_or_a_phase_field() -> None:
    assert _field("globals.model.n_layer").key == "n_layer"
    assert _field("phases.pretrain.lr").key == "lr"


@pytest.mark.parametrize(
    "path",
    ["lr", "phases.pretrain", "globals.nope.n_layer", "phases.nope.lr", "phases.pretrain.nope"],
)
def test_a_path_the_schema_does_not_have_is_refused(path: str) -> None:
    with pytest.raises(utrain.exceptions.UI):
        _field(path)


def test_a_list_axis_is_coerced_to_the_field_type() -> None:
    assert utrain.sweeps.expand_axis(["4", "8"], _field("globals.model.n_layer")) == [4, 8]


def test_a_log_range_is_spaced_evenly_in_log_space() -> None:
    values = utrain.sweeps.expand_axis(
        {"log": [1e-4, 1e-2], "num": 3}, _field("phases.pretrain.lr")
    )
    assert values == [1e-4, 1e-3, 1e-2]


def test_a_linear_range_of_an_int_field_rounds_and_drops_repeats() -> None:
    values = utrain.sweeps.expand_axis({"lin": [1, 2], "num": 5}, _field("globals.model.n_layer"))
    assert values == [1, 2]


def test_no_values_means_every_value_of_a_bool_or_an_enum() -> None:
    assert utrain.sweeps.expand_axis(None, _field("phases.pretrain.resume")) == [False, True]
    assert utrain.sweeps.expand_axis(None, _field("globals.model.dtype")) == ["fp32", "bf16"]


def test_no_values_for_a_number_field_is_an_error() -> None:
    with pytest.raises(utrain.exceptions.UI):
        utrain.sweeps.expand_axis(None, _field("phases.pretrain.lr"))


def test_axis_values_are_checked_against_the_field() -> None:
    with pytest.raises(utrain.exceptions.UI):
        utrain.sweeps.expand_axis([0, 4], _field("globals.model.n_layer"))  # min is 1
    with pytest.raises(utrain.exceptions.UI):
        utrain.sweeps.expand_axis(["fp8"], _field("globals.model.dtype"))


def test_the_cli_axis_syntax() -> None:
    parse = utrain.sweeps.parse_axis_arg
    assert parse("globals.model.n_layer=4,6, 8") == ("globals.model.n_layer", ["4", "6", "8"])
    assert parse("phases.pretrain.lr=log:1e-4:1e-2:5") == (
        "phases.pretrain.lr",
        {"log": ["1e-4", "1e-2"], "num": "5"},
    )
    assert parse("phases.pretrain.lr=lin:0:1") == ("phases.pretrain.lr", {"lin": ["0", "1"]})
    assert parse("globals.model.dtype=*") == ("globals.model.dtype", None)
    with pytest.raises(utrain.exceptions.UI):
        parse("phases.pretrain.lr")


def test_the_grid_varies_the_first_axis_slowest() -> None:
    assert utrain.sweeps.grid({"a": [1, 2], "b": ["x", "y"]}) == [
        {"a": 1, "b": "x"},
        {"a": 1, "b": "y"},
        {"a": 2, "b": "x"},
        {"a": 2, "b": "y"},
    ]


def test_a_spec_survives_its_json_round_trip() -> None:
    spec = utrain.sweeps.Spec(
        axes={"phases.pretrain.lr": [0.001, 0.01]},
        replicate=[],
        compute=["gpu0"],
        base=None,
    )
    assert utrain.sweeps.Spec.from_json(spec.to_json()) == spec
    assert spec.size() == 2


# -- creating -------------------------------------------------------------


@pytest.mark.usefixtures("podman")
def test_creating_a_sweep_queues_one_run_per_point(
    session: sqlalchemy.orm.Session, settings: utrain.config.Settings
) -> None:
    sweep_id = _create(session, settings)
    detail = utrain.sweeps.get_sweep("lr-depth", session)

    assert detail.sweep.id == sweep_id
    assert detail.sweep.status == "draft"
    assert detail.sweep.counts.total == 4
    assert [r.name for r in detail.runs] == [
        "lr-depth-01",
        "lr-depth-02",
        "lr-depth-03",
        "lr-depth-04",
    ]
    assert all(r.status == "queued" and r.image_id == FROZEN_ID for r in detail.runs)
    assert [r.point for r in detail.runs] == [
        {"phases.pretrain.lr": 1e-4, "globals.model.n_layer": 4},
        {"phases.pretrain.lr": 1e-4, "globals.model.n_layer": 8},
        {"phases.pretrain.lr": 3e-4, "globals.model.n_layer": 4},
        {"phases.pretrain.lr": 3e-4, "globals.model.n_layer": 8},
    ]


@pytest.mark.usefixtures("podman")
def test_each_run_keeps_the_compute_it_was_given_round_robin(
    session: sqlalchemy.orm.Session, settings: utrain.config.Settings
) -> None:
    _create(session, settings)
    detail = utrain.sweeps.get_sweep("lr-depth", session)
    assert [r.compute for r in detail.runs] == ["gpu0", "gpu1", "gpu0", "gpu1"]


@pytest.mark.usefixtures("podman")
def test_each_run_config_carries_its_point_over_the_defaults(
    session: sqlalchemy.orm.Session, settings: utrain.config.Settings
) -> None:
    _create(session, settings)
    run = utrain.sweeps.get_sweep("lr-depth", session).runs[3]
    cfg = yaml.safe_load((settings.runs_dir / run.id / "config.yaml").read_text())
    assert cfg == {
        "run_id": run.id,
        "compute": "gpu1",
        "globals": {"model": {"n_layer": 8, "dtype": "fp32"}},
        "phases": {"pretrain": {"lr": 3e-4, "resume": False}},
    }


@pytest.mark.usefixtures("podman")
def test_a_sweep_from_a_run_starts_from_that_run_config(
    session: sqlalchemy.orm.Session, settings: utrain.config.Settings
) -> None:
    base = "c" * 32
    (settings.runs_dir / base).mkdir(parents=True)
    (settings.runs_dir / base / "config.yaml").write_text(
        f"run_id: {base}\ncompute: cpu\nglobals:\n  model:\n    n_layer: 6\n    dtype: bf16\n"
        "phases:\n  pretrain:\n    lr: 0.002\n    resume: true\n"
    )
    session.execute(
        sqlalchemy.insert(utrain.db.runs).values(
            id=base,
            name="base",
            image="img",
            image_id=FROZEN_ID,
            compute="cpu",
            status="done",
            config_hash=None,
            created_at=1.0,
        )
    )
    # Stored as `run create` would have, from the stubbed `describe`.
    utrain.db.describe_image(FROZEN_ID, session)
    _create(session, settings, axes={"phases.pretrain.lr": [0.01]}, compute=["cpu"], base="c")
    detail = utrain.sweeps.get_sweep("lr-depth", session)
    cfg = yaml.safe_load((settings.runs_dir / detail.runs[0].id / "config.yaml").read_text())
    assert cfg["globals"] == {"model": {"n_layer": 6, "dtype": "bf16"}}
    assert cfg["phases"] == {"pretrain": {"lr": 0.01, "resume": True}}
    assert detail.sweep.base == base


@pytest.mark.usefixtures("podman")
def test_a_sweep_name_is_unique_and_well_formed(
    session: sqlalchemy.orm.Session, settings: utrain.config.Settings
) -> None:
    _create(session, settings)
    with pytest.raises(utrain.exceptions.UI, match="already exists"):
        _create(session, settings)
    with pytest.raises(utrain.exceptions.UI, match="must be letters"):
        _create(session, settings, name="bad name")


@pytest.mark.usefixtures("podman")
def test_an_unknown_compute_is_refused_at_creation(
    session: sqlalchemy.orm.Session, settings: utrain.config.Settings
) -> None:
    with pytest.raises(utrain.exceptions.UI, match="unknown --compute"):
        _create(session, settings, compute=["gpu7"])


@pytest.mark.usefixtures("podman")
def test_a_replicate_axis_must_be_an_axis(
    session: sqlalchemy.orm.Session, settings: utrain.config.Settings
) -> None:
    with pytest.raises(utrain.exceptions.UI, match="replicate"):
        _create(session, settings, replicate=["phases.pretrain.resume"])


@pytest.mark.usefixtures("podman")
def test_a_sweep_is_found_by_name_by_at_name_or_by_id_prefix(
    session: sqlalchemy.orm.Session, settings: utrain.config.Settings
) -> None:
    sweep_id = _create(session, settings)
    assert utrain.sweeps.resolve_sweep_id("lr-depth", session) == sweep_id
    assert utrain.sweeps.resolve_sweep_id("@lr-depth", session) == sweep_id
    assert utrain.sweeps.resolve_sweep_id(sweep_id[:6], session) == sweep_id
    with pytest.raises(utrain.exceptions.UI, match="not found"):
        utrain.sweeps.resolve_sweep_id("nope", session)


# -- dispatching ----------------------------------------------------------


class _Starts:
    """Records the runs a tick starts, and marks them running as start_run would."""

    def __init__(self, fail: set[str] | None = None) -> None:
        self.started: list[str] = []
        self.fail = fail or set()

    def __call__(self, run_id: str, session: sqlalchemy.orm.Session) -> None:
        if run_id in self.fail:
            raise utrain.exceptions.UI("no toolkit")
        self.started.append(run_id)
        session.execute(
            sqlalchemy.update(utrain.db.runs)
            .where(utrain.db.runs.c.id == run_id)
            .values(status="running")
        )


def _ids(session: sqlalchemy.orm.Session) -> list[str]:
    return [r.id for r in utrain.sweeps.get_sweep("lr-depth", session).runs]


def _tick(session: sqlalchemy.orm.Session, starts: "_Starts", log: list[str] | None = None) -> bool:
    return utrain.dispatcher.tick(
        session, starts, log=log.append if log is not None else lambda m: None
    )


@pytest.mark.usefixtures("podman")
def test_a_draft_sweep_is_not_dispatched(
    session: sqlalchemy.orm.Session, settings: utrain.config.Settings
) -> None:
    _create(session, settings)
    starts = _Starts()
    assert not _tick(session, starts)
    assert starts.started == []


@pytest.mark.usefixtures("podman")
def test_a_tick_starts_one_run_per_free_compute_in_grid_order(
    session: sqlalchemy.orm.Session, settings: utrain.config.Settings
) -> None:
    sweep_id = _create(session, settings)
    utrain.sweeps.start_sweep(sweep_id, session)
    ids = _ids(session)
    starts = _Starts()

    assert _tick(session, starts)
    assert starts.started == [ids[0], ids[1]]

    # Both computes are busy now: the next tick starts nothing.
    assert _tick(session, starts)
    assert starts.started == [ids[0], ids[1]]

    # gpu0's run finishes: its next run starts, and only that one.
    session.execute(
        sqlalchemy.update(utrain.db.runs).where(utrain.db.runs.c.id == ids[0]).values(status="done")
    )
    _tick(session, starts)
    assert starts.started == [ids[0], ids[1], ids[2]]


@pytest.mark.usefixtures("podman")
def test_a_run_waits_for_its_own_compute_even_when_another_is_free(
    session: sqlalchemy.orm.Session, settings: utrain.config.Settings
) -> None:
    sweep_id = _create(session, settings, axes={"phases.pretrain.lr": [1e-4, 2e-4, 3e-4]})
    utrain.sweeps.start_sweep(sweep_id, session)
    ids = _ids(session)  # gpu0, gpu1, gpu0
    starts = _Starts()
    _tick(session, starts)
    # gpu1 finishes, gpu0 is still busy: the third run is gpu0's, so it waits.
    session.execute(
        sqlalchemy.update(utrain.db.runs).where(utrain.db.runs.c.id == ids[1]).values(status="done")
    )
    _tick(session, starts)
    assert starts.started == [ids[0], ids[1]]


@pytest.mark.usefixtures("podman")
def test_a_compute_busy_with_a_run_started_by_hand_is_left_alone(
    session: sqlalchemy.orm.Session, settings: utrain.config.Settings
) -> None:
    sweep_id = _create(session, settings)
    utrain.sweeps.start_sweep(sweep_id, session)
    session.execute(
        sqlalchemy.insert(utrain.db.runs).values(
            id="d" * 32,
            name="mine",
            image="img",
            image_id=FROZEN_ID,
            compute="gpu0",
            status="running",
            config_hash=None,
            created_at=1.0,
        )
    )
    starts = _Starts()
    _tick(session, starts)
    assert starts.started == [_ids(session)[1]]


@pytest.mark.usefixtures("podman")
def test_a_run_that_cannot_start_is_failed_rather_than_retried_forever(
    session: sqlalchemy.orm.Session, settings: utrain.config.Settings
) -> None:
    sweep_id = _create(session, settings)
    utrain.sweeps.start_sweep(sweep_id, session)
    # Committed, as the CLI commits before the dispatcher runs: a failed start
    # rolls back whatever it half wrote, and must not take the sweep with it.
    session.commit()
    ids = _ids(session)
    messages: list[str] = []
    _tick(session, _Starts(fail={ids[0]}), messages)
    assert _statuses(session, sweep_id)[:2] == ["failed", "running"]
    assert any("no toolkit" in m for m in messages)


@pytest.mark.usefixtures("podman")
def test_a_paused_sweep_starts_nothing_and_a_finished_one_is_done(
    session: sqlalchemy.orm.Session, settings: utrain.config.Settings
) -> None:
    sweep_id = _create(session, settings, axes={"phases.pretrain.lr": [1e-4]})
    utrain.sweeps.start_sweep(sweep_id, session)
    utrain.sweeps.pause_sweep(sweep_id, session)
    assert utrain.sweeps.get_sweep(sweep_id, session).sweep.status == "paused"
    assert not _tick(session, _Starts())

    utrain.sweeps.start_sweep(sweep_id, session)
    session.execute(
        sqlalchemy.update(utrain.db.runs)
        .where(utrain.db.runs.c.sweep_id == sweep_id)
        .values(status="done")
    )
    assert utrain.sweeps.get_sweep(sweep_id, session).sweep.status == "done"
    assert not _tick(session, _Starts())


# -- lifecycle ------------------------------------------------------------


@pytest.mark.usefixtures("podman")
def test_starting_a_gpu_sweep_without_the_toolkit_is_refused_up_front(
    monkeypatch: pytest.MonkeyPatch,
    session: sqlalchemy.orm.Session,
    settings: utrain.config.Settings,
) -> None:
    sweep_id = _create(session, settings)
    monkeypatch.setattr(utrain.orchestrator.shutil, "which", lambda name: None)
    monkeypatch.setattr(utrain.orchestrator, "ensure_gpu_toolkit", _REAL_TOOLKIT_CHECK)
    with pytest.raises(utrain.exceptions.UI, match="nvidia-ctk"):
        utrain.sweeps.start_sweep(sweep_id, session)
    assert utrain.sweeps.get_sweep(sweep_id, session).sweep.status == "draft"


@pytest.mark.usefixtures("podman")
def test_cancel_stops_queued_runs_and_retry_queues_them_again(
    session: sqlalchemy.orm.Session, settings: utrain.config.Settings
) -> None:
    sweep_id = _create(session, settings)
    utrain.sweeps.cancel_sweep(sweep_id, session)
    assert _statuses(session, sweep_id) == ["stopped"] * 4
    assert utrain.sweeps.get_sweep(sweep_id, session).sweep.status == "cancelled"
    with pytest.raises(utrain.exceptions.UI, match="cancelled"):
        utrain.sweeps.start_sweep(sweep_id, session)

    _, n = utrain.sweeps.retry_sweep(sweep_id, session)
    assert n == 4
    assert _statuses(session, sweep_id) == ["queued"] * 4
    assert utrain.sweeps.get_sweep(sweep_id, session).sweep.status == "running"


@pytest.mark.usefixtures("podman")
def test_extend_adds_runs_for_new_points_only(
    session: sqlalchemy.orm.Session, settings: utrain.config.Settings
) -> None:
    sweep_id = _create(session, settings)
    _, n = utrain.sweeps.extend_sweep(
        sweep_id, {"globals.model.n_layer": ["8", "12"]}, settings, session
    )
    assert n == 2
    detail = utrain.sweeps.get_sweep(sweep_id, session)
    assert detail.sweep.axes["globals.model.n_layer"] == [4, 8, 12]
    assert [r.name for r in detail.runs[4:]] == ["lr-depth-05", "lr-depth-06"]
    assert [r.point for r in detail.runs[4:]] == [
        {"phases.pretrain.lr": 1e-4, "globals.model.n_layer": 12},
        {"phases.pretrain.lr": 3e-4, "globals.model.n_layer": 12},
    ]
    with pytest.raises(utrain.exceptions.UI, match="not an axis"):
        utrain.sweeps.extend_sweep(sweep_id, {"phases.pretrain.resume": None}, settings, session)


@pytest.mark.usefixtures("podman")
def test_delete_removes_the_sweep_and_its_runs(
    session: sqlalchemy.orm.Session, settings: utrain.config.Settings
) -> None:
    sweep_id = _create(session, settings)
    _, n = utrain.sweeps.delete_sweep(sweep_id, force=False, session=session)
    assert n == 4
    assert utrain.sweeps.list_sweeps(session) == []
    assert session.execute(sqlalchemy.select(utrain.db.runs)).fetchall() == []


# -- rendering ------------------------------------------------------------


@pytest.mark.usefixtures("podman")
def test_sweep_show_lists_each_run_with_its_point(
    session: sqlalchemy.orm.Session, settings: utrain.config.Settings
) -> None:
    _create(session, settings)
    text = utrain.cli.render.sweep_detail(utrain.sweeps.get_sweep("lr-depth", session))
    assert "status:   draft" in text
    assert "  phases.pretrain.lr: 0.0001, 0.0003" in text
    assert "lr=0.0003 n_layer=8" in text


# -- command line ---------------------------------------------------------


def _cli(*argv: str) -> None:
    args = utrain.cli.main.build_parser().parse_args(list(argv))
    args.func(args)


@pytest.mark.usefixtures("podman")
def test_sweep_create_from_flags_and_from_a_spec_file(
    monkeypatch: pytest.MonkeyPatch,
    settings: utrain.config.Settings,
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("UTRAIN_DATA_DIR", str(settings.data_dir))
    _cli(
        "sweep",
        "create",
        "--name",
        "flags",
        "--image",
        "img",
        "--axis",
        "phases.pretrain.lr=log:1e-4:1e-2:3",
        "--axis",
        "globals.model.dtype=*",
        "--compute",
        "gpu0,gpu1",
    )
    out = capsys.readouterr().out
    assert "runs:     6 (" in out
    assert "lr=0.01 dtype=bf16" in out

    spec = tmp_path / "spec.yaml"
    spec.write_text(
        "name: from-file\nimage: img\ncompute: cpu\n"
        "axes:\n  globals.model.n_layer: [2, 4]\n"
        "  phases.pretrain.resume: {values: [false, true], replicate: true}\n"
    )
    # A flag overrides the file: same spec, another name.
    _cli("sweep", "create", str(spec), "--name", "renamed", "--print-id")
    sweep_id = capsys.readouterr().out.strip()
    _cli("sweep", "show", sweep_id)
    out = capsys.readouterr().out
    assert "name:     renamed" in out
    assert "phases.pretrain.resume: false, true  (replicate)" in out

    _cli("sweep", "list")
    out = capsys.readouterr().out
    assert "flags" in out and "renamed" in out
