"""Tests for reading a phase's ``.rtsdb`` metrics.

Fixtures are written with ``naw.wandb``, which is the same producer the
containers use -- the orchestrator mounts it over the image's real ``wandb``
(``orchestrator._wandb_mount_args``). Nothing here needs podman, so unlike the
cram suite these run in CI.
"""

import os
import pathlib

import naw.wandb
import pytest

import utrain.metrics

_PROJECT = "whatever-the-container-called-it"


def _write(attempt_dir: pathlib.Path, phase: str, rows: list[dict[str, float]]) -> pathlib.Path:
    """A phase's metrics file, laid out the way a real phase leaves it.

    `naw.wandb` always inserts a literal ``wandb`` path segment, and under
    utrain that segment *is* the phase's directory, because the orchestrator
    bind-mounts `<attempt_dir>/wandb/<phase>` at `<root>/wandb`. There is no
    mount here, so the rename stands in for one. The project name is
    deliberately not the phase: utrain does not choose it and must not depend
    on it.
    """
    staging = attempt_dir / f".staging-{phase}"
    run = naw.wandb.init(project=_PROJECT, id="rid", dir=str(staging))
    for step, row in enumerate(rows):
        run.log(dict(row), step=step, commit=True)
    run.finish()

    phase_dir = attempt_dir / "wandb" / phase
    phase_dir.parent.mkdir(parents=True, exist_ok=True)
    (staging / "wandb").rename(phase_dir)
    staging.rmdir()
    return phase_dir / _PROJECT / "rid.rtsdb"


def test_find_rtsdb_missing_phase_dir(tmp_path: pathlib.Path) -> None:
    assert utrain.metrics.find_rtsdb(tmp_path, "pretrain") is None


def test_find_rtsdb_empty_phase_dir(tmp_path: pathlib.Path) -> None:
    (tmp_path / "wandb" / "pretrain").mkdir(parents=True)
    assert utrain.metrics.find_rtsdb(tmp_path, "pretrain") is None


def test_find_rtsdb_ignores_a_file_outside_a_project_dir(tmp_path: pathlib.Path) -> None:
    """Only `<phase>/<project>/<id>.rtsdb` counts -- a stray file is not a run."""
    phase_dir = tmp_path / "wandb" / "pretrain"
    phase_dir.mkdir(parents=True)
    (phase_dir / "stray.rtsdb").write_bytes(b"")
    assert utrain.metrics.find_rtsdb(tmp_path, "pretrain") is None


def test_find_rtsdb(tmp_path: pathlib.Path) -> None:
    path = _write(tmp_path, "pretrain", [{"loss": 1.0}])
    assert utrain.metrics.find_rtsdb(tmp_path, "pretrain") == path


def test_find_rtsdb_follows_the_newest(tmp_path: pathlib.Path) -> None:
    """A phase re-run under a new wandb id is followed, not the run before it.

    The ids do not order, so the newest file is the only way to tell which one
    is still being written to.
    """
    phase_dir = tmp_path / "wandb" / "pretrain" / _PROJECT
    phase_dir.mkdir(parents=True)
    for name, mtime in (("b.rtsdb", 200.0), ("a.rtsdb", 300.0), ("c.rtsdb", 100.0)):
        path = phase_dir / name
        path.write_bytes(b"")
        os.utime(path, (mtime, mtime))
    found = utrain.metrics.find_rtsdb(tmp_path, "pretrain")
    assert found is not None and found.name == "a.rtsdb"


def test_find_rtsdb_is_deterministic(tmp_path: pathlib.Path) -> None:
    """Two files of the same age must not resolve differently between calls."""
    phase_dir = tmp_path / "wandb" / "pretrain" / _PROJECT
    phase_dir.mkdir(parents=True)
    for name in ("b.rtsdb", "a.rtsdb", "c.rtsdb"):
        path = phase_dir / name
        path.write_bytes(b"")
        os.utime(path, (100.0, 100.0))
    found = utrain.metrics.find_rtsdb(tmp_path, "pretrain")
    assert found is not None and found.name == "c.rtsdb"


def test_read_columns_and_points(tmp_path: pathlib.Path) -> None:
    path = _write(tmp_path, "pretrain", [{"loss": 3.0}, {"loss": 2.0}, {"loss": 1.0}])

    with utrain.metrics.Tail(path) as tail:
        update = tail.read()

        assert update.columns == ["loss"]
        assert tail.columns == ["loss"]
        assert [p.value for p in update.points["loss"]] == [3.0, 2.0, 1.0]
        assert [p.step for p in update.points["loss"]] == [0, 1, 2]
        assert all(p.timestamp > 0 for p in update.points["loss"])


def test_read_sparse_columns_keep_their_own_x(tmp_path: pathlib.Path) -> None:
    """A metric logged every other step must not borrow the other's steps."""
    path = _write(
        tmp_path,
        "pretrain",
        [{"loss": 3.0, "val": 9.0}, {"loss": 2.0}, {"loss": 1.0, "val": 8.0}],
    )
    points = utrain.metrics.read_all(path)

    assert [p.step for p in points["loss"]] == [0, 1, 2]
    assert [p.step for p in points["val"]] == [0, 2]
    assert [p.value for p in points["val"]] == [9.0, 8.0]


def test_a_second_read_of_a_finished_file_is_empty(tmp_path: pathlib.Path) -> None:
    path = _write(tmp_path, "pretrain", [{"loss": 1.0}, {"loss": 2.0}])

    with utrain.metrics.Tail(path) as tail:
        assert tail.read().points["loss"]
        second = tail.read()
        assert second.points == {}
        assert second.columns == []
        assert second.empty


def test_read_resumes_where_the_last_read_stopped(tmp_path: pathlib.Path) -> None:
    """The whole point of a Tail: each read returns only what is new.

    The file is grown a prefix at a time rather than by driving a live writer,
    because ``naw``'s ``chunk.Writer`` wraps an ordinary buffered file with no
    autoflush -- a real producer's rows reach disk in buffer-sized batches. A
    growing byte prefix is what the reader sees either way, and it is
    deterministic.
    """
    source = _write(tmp_path, "pretrain", [{"loss": float(i)} for i in range(40)])
    whole = utrain.metrics.read_all(source)
    data = source.read_bytes()

    growing = tmp_path / "growing.rtsdb"
    growing.write_bytes(data[: len(data) // 3])

    with utrain.metrics.Tail(growing) as tail:
        first = tail.read()
        assert first.points["loss"]

        # Nothing appended since.
        assert tail.read().empty

        growing.write_bytes(data)
        second = tail.read()

        assert second.points["loss"]
        # The batches join up into the full series, with nothing dropped or
        # repeated at the seam.
        assert first.points["loss"] + second.points["loss"] == whole["loss"]


def test_read_accumulates_to_the_same_series(tmp_path: pathlib.Path) -> None:
    """Following a file byte by byte must reconstruct one whole read exactly."""
    source = _write(tmp_path, "pretrain", [{"loss": float(i), "mfu": i / 2} for i in range(25)])
    whole = utrain.metrics.read_all(source)
    data = source.read_bytes()

    growing = tmp_path / "growing.rtsdb"
    accumulated: dict[str, list[utrain.metrics.MetricPoint]] = {}

    with utrain.metrics.Tail(growing) as tail:
        for size in range(0, len(data) + 32, 32):
            growing.write_bytes(data[:size])
            for name, points in tail.read().points.items():
                accumulated.setdefault(name, []).extend(points)

    assert accumulated == whole
    assert sorted(tail.columns) == sorted(whole)


def test_a_tail_on_a_phase_that_has_not_logged_yet(tmp_path: pathlib.Path) -> None:
    """A Tail may outlive the absence of its file, and start when it appears."""
    path = tmp_path / "wandb" / "pretrain" / _PROJECT / "rid.rtsdb"

    with utrain.metrics.Tail(path) as tail:
        assert tail.read().empty
        assert tail.columns == []

        assert _write(tmp_path, "pretrain", [{"loss": 1.0}]) == path

        update = tail.read()
        assert [p.value for p in update.points["loss"]] == [1.0]
        assert update.columns == ["loss"]


def test_read_of_an_empty_file_is_empty(tmp_path: pathlib.Path) -> None:
    """A file the writer created but has not written a header to yet."""
    path = tmp_path / "empty.rtsdb"
    path.write_bytes(b"")

    with utrain.metrics.Tail(path) as tail:
        update = tail.read()

    assert update.empty
    assert update.columns == []


@pytest.mark.parametrize("phase", ["tokenizer", "pretrain"])
def test_read_ignores_underscore_columns(tmp_path: pathlib.Path, phase: str) -> None:
    path = _write(tmp_path, phase, [{"loss": 1.0}])

    with utrain.metrics.Tail(path) as tail:
        tail.read()
        assert tail.columns == ["loss"]
