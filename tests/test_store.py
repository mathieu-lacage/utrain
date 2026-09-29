"""Unit tests for the content-addressed store.

`store.gc` and `store.check` both work purely off the filesystem plus the
attempts table, so they can be exercised by building a data directory by hand
-- no podman, so these run in CI, like test_query_layer.py.
"""

import hashlib
import os
import pathlib
import subprocess
import typing

import pytest
import sqlalchemy
import sqlalchemy.dialects.sqlite

import utrain.cli.render
import utrain.config
import utrain.db
import utrain.exceptions
import utrain.store
import utrain.types


@pytest.fixture()
def root(
    tmp_path: pathlib.Path,
) -> typing.Iterator[tuple[utrain.config.Settings, sqlalchemy.orm.Session]]:
    settings = utrain.config.Settings(data_dir=tmp_path)
    with utrain.db.with_db(settings) as session:
        yield settings, session


def _insert_run(
    session: sqlalchemy.orm.Session,
    run_id: str,
    status: str = "done",
) -> None:
    session.execute(
        sqlalchemy.insert(utrain.db.runs).values(
            id=run_id,
            name="run",
            image="img",
            image_id="ab" * 32,
            compute="cpu",
            status=status,
            config_hash=None,
            created_at=1.0,
        )
    )


def _insert_attempt(
    session: sqlalchemy.orm.Session,
    run_id: str,
    attempt: int,
    status: str,
) -> None:
    session.execute(
        sqlalchemy.insert(utrain.db.run_attempts).values(
            run_id=run_id,
            attempt=attempt,
            from_phase=None,
            status=status,
            pid=None,
            started_at=1.0,
            ended_at=2.0 if status == "done" else None,
        )
    )


def _insert_done_attempt(session: sqlalchemy.orm.Session, run_id: str, attempt: int = 1) -> None:
    _insert_run(session, run_id)
    _insert_attempt(session, run_id, attempt, "done")


def _add_linked_data(
    session: sqlalchemy.orm.Session,
    settings: utrain.config.Settings,
    run_id: str,
    attempt: int,
    phase: str,
    name: str,
    content: bytes,
) -> None:
    """Leave behind exactly what a phase that succeeded leaves: a 0o444 store
    file named after the content's hash, hardlinked into the phase's data
    dir, and the phase recorded `done`."""
    session.execute(
        sqlalchemy.dialects.sqlite.insert(utrain.db.run_phases)
        .values(run_id=run_id, attempt=attempt, phase=phase, phase_order=0, status="done")
        .on_conflict_do_nothing()
    )
    sha = hashlib.sha256(content).hexdigest()
    store_path = settings.data_dir / "store" / sha
    store_path.parent.mkdir(parents=True, exist_ok=True)
    # As `consolidate` leaves it: one 0o444 file per distinct content, so a
    # second phase sharing it must not rewrite the file.
    if not store_path.exists():
        store_path.write_bytes(content)
        store_path.chmod(0o444)
    data_dir = settings.runs_dir / run_id / "attempt" / str(attempt) / "data" / phase
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / name
    target.write_bytes(content)
    target.unlink()
    os.link(store_path, target)


def _break_hardlink(settings: utrain.config.Settings, path: pathlib.Path) -> None:
    """Replace a hardlink with a private copy: right content, wrong inode."""
    content = path.read_bytes()
    path.unlink()
    path.write_bytes(content)


def test_store_check_accepts_deduplicated_data(
    root: tuple[utrain.config.Settings, sqlalchemy.orm.Session],
) -> None:
    settings, session = root
    _insert_done_attempt(session, "a" * 32)
    # Two phases, three files, two distinct contents: one shared across phases,
    # mirroring the layout the cram suite's fake container produces.
    _add_linked_data(session, settings, "a" * 32, 1, "tokenizer", "tok.txt", b"tok\n")
    _add_linked_data(session, settings, "a" * 32, 1, "tokenizer", "shared.txt", b"shared\n")
    _add_linked_data(session, settings, "a" * 32, 1, "pretrain", "shared.txt", b"shared\n")

    result = utrain.store.check(settings, session)

    assert result == utrain.types.StoreCheckResult(
        attempts=1,
        data_files=3,
        store_files=2,
        orphaned=0,
        problems=[],
    )
    assert utrain.cli.render.store_check(result) == (
        "checked 3 data file(s) in 1 attempt(s) against 2 store file(s): ok"
    )


def test_store_check_flags_a_copied_file(
    root: tuple[utrain.config.Settings, sqlalchemy.orm.Session],
) -> None:
    settings, session = root
    run_id = "b" * 32
    _insert_done_attempt(session, run_id)
    _add_linked_data(session, settings, run_id, 1, "pretrain", "model.bin", b"weights")
    _break_hardlink(
        settings, settings.runs_dir / run_id / "attempt" / "1" / "data" / "pretrain" / "model.bin"
    )

    result = utrain.store.check(settings, session)

    assert result.data_files == 1
    assert result.store_files == 1
    assert len(result.problems) == 1
    problem = result.problems[0]
    assert problem.path == pathlib.Path(f"runs/{run_id}/attempt/1/data/pretrain/model.bin")
    assert problem.problem == "not a hardlink into the store"


def test_store_check_flags_a_misnamed_store_file(
    root: tuple[utrain.config.Settings, sqlalchemy.orm.Session],
) -> None:
    settings, session = root
    run_id = "c" * 32
    _insert_done_attempt(session, run_id)
    _add_linked_data(session, settings, run_id, 1, "pretrain", "model.bin", b"weights")
    # Rename the store entry so its name no longer is its content's hash; the
    # data file keeps sharing its inode, which alone is not enough.
    store_file = settings.data_dir / "store" / hashlib.sha256(b"weights").hexdigest()
    store_file.rename(store_file.with_name("deadbeef"))

    result = utrain.store.check(settings, session)

    assert len(result.problems) == 1
    assert result.problems[0].path == pathlib.Path("store/deadbeef")
    assert result.problems[0].problem == (
        f"content hashes to {hashlib.sha256(b'weights').hexdigest()}, not its name"
    )


def test_store_check_skips_phases_that_are_not_done(
    root: tuple[utrain.config.Settings, sqlalchemy.orm.Session],
) -> None:
    settings, session = root
    run_id = "d" * 32
    _insert_run(session, run_id, status="running")
    _insert_attempt(session, run_id, 1, "running")
    session.execute(
        sqlalchemy.insert(utrain.db.run_phases).values(
            run_id=run_id, attempt=1, phase="pretrain", phase_order=0, status="running"
        )
    )
    # A running phase's data is plain files: it is consolidated once it succeeds.
    data_dir = settings.runs_dir / run_id / "attempt" / "1" / "data" / "pretrain"
    data_dir.mkdir(parents=True)
    (data_dir / "model.bin").write_bytes(b"weights")

    result = utrain.store.check(settings, session)

    assert result.attempts == 0
    assert result.data_files == 0
    assert result.problems == []


def test_store_check_skips_done_attempts_without_a_data_dir(
    root: tuple[utrain.config.Settings, sqlalchemy.orm.Session],
) -> None:
    settings, session = root
    _insert_done_attempt(session, "e" * 32)

    result = utrain.store.check(settings, session)

    assert result.attempts == 0
    assert result.problems == []


def test_store_check_counts_orphaned_store_files(
    root: tuple[utrain.config.Settings, sqlalchemy.orm.Session],
) -> None:
    settings, session = root
    # What the directory looks like after `run delete` but before `store gc`.
    orphan = settings.data_dir / "store" / hashlib.sha256(b"weights").hexdigest()
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(b"weights")

    result = utrain.store.check(settings, session)

    assert result == utrain.types.StoreCheckResult(
        attempts=0,
        data_files=0,
        store_files=1,
        orphaned=1,
        problems=[],
    )
    assert "note: 1 orphaned store file(s)" in utrain.cli.render.store_check(result)


def test_store_check_without_hashing_trusts_the_names(
    root: tuple[utrain.config.Settings, sqlalchemy.orm.Session],
) -> None:
    settings, session = root
    _insert_done_attempt(session, "f" * 32)
    _add_linked_data(session, settings, "f" * 32, 1, "pretrain", "model.bin", b"weights")
    store_file = settings.data_dir / "store" / hashlib.sha256(b"weights").hexdigest()
    store_file.rename(store_file.with_name("deadbeef"))

    result = utrain.store.check(settings, session, hash_contents=False)

    assert result.problems == []
    assert result.data_files == 1
    assert result.store_files == 1


def test_store_check_verbose_lists_a_link_per_file(
    root: tuple[utrain.config.Settings, sqlalchemy.orm.Session],
) -> None:
    settings, session = root
    run_id = "1" * 32
    _insert_done_attempt(session, run_id)
    _add_linked_data(session, settings, run_id, 1, "tokenizer", "tok.txt", b"tok\n")
    # Two pretrain files share one store file: both get their own line.
    _add_linked_data(session, settings, run_id, 1, "pretrain", "shared.txt", b"shared\n")
    _add_linked_data(session, settings, run_id, 1, "pretrain", "shared2.txt", b"shared\n")

    result = utrain.store.check(settings, session, verbose=True)

    data_prefix = pathlib.Path(f"runs/{run_id}/attempt/1/data")
    assert result.links == [
        utrain.types.StoreLink(
            data=data_prefix / "pretrain" / "shared.txt",
            store=pathlib.Path("store") / hashlib.sha256(b"shared\n").hexdigest(),
        ),
        utrain.types.StoreLink(
            data=data_prefix / "pretrain" / "shared2.txt",
            store=pathlib.Path("store") / hashlib.sha256(b"shared\n").hexdigest(),
        ),
        utrain.types.StoreLink(
            data=data_prefix / "tokenizer" / "tok.txt",
            store=pathlib.Path("store") / hashlib.sha256(b"tok\n").hexdigest(),
        ),
    ]
    rendered = utrain.cli.render.store_check(result)
    h = hashlib.sha256(b"shared\n").hexdigest()
    assert rendered.splitlines()[0] == (
        f"runs/{run_id}/attempt/1/data/pretrain/shared.txt  store/{h}"
    )
    # Not asked for, no links.
    assert utrain.store.check(settings, session).links == []


def test_store_check_scope_narrows_to_a_run(
    root: tuple[utrain.config.Settings, sqlalchemy.orm.Session],
) -> None:
    settings, session = root
    _insert_done_attempt(session, "ab" * 16)
    _insert_done_attempt(session, "cd" * 16)
    _add_linked_data(session, settings, "ab" * 16, 1, "pretrain", "model.bin", b"weights")
    _add_linked_data(session, settings, "cd" * 16, 1, "pretrain", "model.bin", b"weights")

    # A prefix resolves the same way it does everywhere else in the CLI.
    result = utrain.store.check(settings, session, scope="ab")

    assert result.attempts == 1
    assert result.data_files == 1
    assert result.problems == []


def test_store_check_scope_narrows_to_an_attempt(
    root: tuple[utrain.config.Settings, sqlalchemy.orm.Session],
) -> None:
    settings, session = root
    run_id = "2" * 32
    _insert_run(session, run_id)
    _insert_attempt(session, run_id, 1, "done")
    _insert_attempt(session, run_id, 2, "done")
    _add_linked_data(session, settings, run_id, 1, "pretrain", "old.bin", b"old")
    _add_linked_data(session, settings, run_id, 2, "pretrain", "new.bin", b"new")

    result = utrain.store.check(settings, session, scope=f"{run_id}/2")

    assert result.attempts == 1
    assert result.data_files == 1
    assert result.problems == []
    assert result.links == []  # not verbose
    # The checked file is attempt 2's, not attempt 1's.
    result = utrain.store.check(settings, session, scope=f"{run_id}/2", verbose=True)
    assert result.links == [
        utrain.types.StoreLink(
            data=pathlib.Path(f"runs/{run_id}/attempt/2/data/pretrain/new.bin"),
            store=pathlib.Path("store") / hashlib.sha256(b"new").hexdigest(),
        )
    ]


def test_store_check_scope_narrows_to_a_phase(
    root: tuple[utrain.config.Settings, sqlalchemy.orm.Session],
) -> None:
    settings, session = root
    run_id = "3" * 32
    _insert_done_attempt(session, run_id)
    session.execute(
        sqlalchemy.insert(utrain.db.run_phases).values(
            run_id=run_id, attempt=1, phase="tokenizer", phase_order=0, status="done"
        )
    )
    session.execute(
        sqlalchemy.insert(utrain.db.run_phases).values(
            run_id=run_id, attempt=1, phase="pretrain", phase_order=1, status="done"
        )
    )
    _add_linked_data(session, settings, run_id, 1, "tokenizer", "tok.txt", b"tok\n")
    _add_linked_data(session, settings, run_id, 1, "pretrain", "model.bin", b"weights")

    result = utrain.store.check(settings, session, scope=f"{run_id}/1/tokenizer", verbose=True)

    assert result.attempts == 1
    assert result.data_files == 1
    assert result.links == [
        utrain.types.StoreLink(
            data=pathlib.Path(f"runs/{run_id}/attempt/1/data/tokenizer/tok.txt"),
            store=pathlib.Path("store") / hashlib.sha256(b"tok\n").hexdigest(),
        )
    ]


def test_store_check_scope_rejects_bad_addresses(
    root: tuple[utrain.config.Settings, sqlalchemy.orm.Session],
) -> None:
    settings, session = root
    run_id = "4" * 32
    _insert_done_attempt(session, run_id)
    session.execute(
        sqlalchemy.insert(utrain.db.run_phases).values(
            run_id=run_id, attempt=1, phase="pretrain", phase_order=0, status="done"
        )
    )

    with pytest.raises(utrain.exceptions.UI, match="run 'zz' not found"):
        utrain.store.check(settings, session, scope="zz")
    # A phase component needs its attempt: RUN_ID/PHASE is not a valid scope.
    with pytest.raises(utrain.exceptions.UI, match="expected attempt number, got 'pretrain'"):
        utrain.store.check(settings, session, scope=f"{run_id}/pretrain")
    with pytest.raises(utrain.exceptions.UI, match=f"attempt 9 of run '{run_id}' not found"):
        utrain.store.check(settings, session, scope=f"{run_id}/9")
    with pytest.raises(
        utrain.exceptions.UI, match=f"phase 'serve' not found in attempt 1 of run '{run_id}'"
    ):
        utrain.store.check(settings, session, scope=f"{run_id}/1/serve")


def test_store_check_scope_names_unfinished_work_instead_of_passing_silently(
    root: tuple[utrain.config.Settings, sqlalchemy.orm.Session],
) -> None:
    settings, session = root
    run_id = "5" * 32
    _insert_run(session, run_id, status="running")
    _insert_attempt(session, run_id, 1, "done")
    _insert_attempt(session, run_id, 2, "running")
    _add_linked_data(session, settings, run_id, 1, "pretrain", "model.bin", b"weights")

    # The done attempt is checked, the running one skipped, as unscoped.
    result = utrain.store.check(settings, session, scope=run_id)
    assert result.attempts == 1
    assert result.data_files == 1

    # Asked about the running attempt directly, silence would read as ok.
    with pytest.raises(
        utrain.exceptions.UI, match=f"attempt 2 of run '{run_id}' has no completed phase"
    ):
        utrain.store.check(settings, session, scope=f"{run_id}/2")

    session.execute(
        sqlalchemy.insert(utrain.db.run_phases).values(
            run_id=run_id, attempt=2, phase="pretrain", phase_order=0, status="running"
        )
    )
    with pytest.raises(
        utrain.exceptions.UI, match=f"phase 'pretrain' of attempt 2 of run '{run_id}' is not done"
    ):
        utrain.store.check(settings, session, scope=f"{run_id}/2/pretrain")

    run_id = "6" * 32
    _insert_run(session, run_id, status="running")
    _insert_attempt(session, run_id, 1, "running")
    with pytest.raises(utrain.exceptions.UI, match=f"run '{run_id}' has no completed phase"):
        utrain.store.check(settings, session, scope=run_id)


def test_store_check_checks_the_done_phases_of_a_failed_attempt(
    root: tuple[utrain.config.Settings, sqlalchemy.orm.Session],
) -> None:
    settings, session = root
    run_id = "7" * 32
    _insert_run(session, run_id, status="failed")
    _insert_attempt(session, run_id, 1, "failed")
    _add_linked_data(session, settings, run_id, 1, "tokenizer", "tok.txt", b"tok\n")
    session.execute(
        sqlalchemy.insert(utrain.db.run_phases).values(
            run_id=run_id, attempt=1, phase="pretrain", phase_order=1, status="failed"
        )
    )
    failed = settings.runs_dir / run_id / "attempt" / "1" / "data" / "pretrain"
    failed.mkdir(parents=True)
    (failed / "partial.bin").write_bytes(b"half a model")

    result = utrain.store.check(settings, session)

    assert (result.attempts, result.data_files, result.problems) == (1, 1, [])


# -- consolidate -----------------------------------------------------------


def _phase_dir(settings: utrain.config.Settings, phase: str) -> pathlib.Path:
    path = settings.runs_dir / ("8" * 32) / "attempt" / "1" / "data" / phase
    path.mkdir(parents=True)
    return path


def _store_file(settings: utrain.config.Settings, content: bytes) -> pathlib.Path:
    return settings.data_dir / "store" / hashlib.sha256(content).hexdigest()


def test_consolidate_moves_each_distinct_content_into_the_store_once(
    root: tuple[utrain.config.Settings, sqlalchemy.orm.Session],
) -> None:
    settings, _ = root
    phase = _phase_dir(settings, "tokenizer")
    (phase / "a.txt").write_bytes(b"same\n")
    (phase / "sub").mkdir()
    (phase / "sub" / "b.txt").write_bytes(b"same\n")
    (phase / "c.txt").write_bytes(b"other\n")

    utrain.store.consolidate(phase, settings)

    same = _store_file(settings, b"same\n")
    for path in (phase / "a.txt", phase / "sub" / "b.txt"):
        assert path.samefile(same)
    assert (phase / "c.txt").samefile(_store_file(settings, b"other\n"))
    assert same.stat().st_mode & 0o777 == 0o444
    assert sorted(p.name for p in (settings.data_dir / "store").iterdir()) == sorted(
        hashlib.sha256(c).hexdigest() for c in (b"same\n", b"other\n")
    )
    # Nothing left behind beside the data files.
    assert sorted(p.name for p in phase.iterdir()) == ["a.txt", "c.txt", "sub"]


def test_consolidate_links_to_content_the_store_already_has(
    root: tuple[utrain.config.Settings, sqlalchemy.orm.Session],
) -> None:
    settings, _ = root
    existing = _store_file(settings, b"weights")
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"weights")
    existing.chmod(0o444)
    phase = _phase_dir(settings, "pretrain")
    (phase / "model.bin").write_bytes(b"weights")

    utrain.store.consolidate(phase, settings)

    assert (phase / "model.bin").samefile(existing)
    assert (phase / "model.bin").read_bytes() == b"weights"


def test_consolidate_hashes_only_what_the_phase_added(
    root: tuple[utrain.config.Settings, sqlalchemy.orm.Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The next phase starts as a hardlink copy of the last, already in the store."""
    settings, _ = root
    first = _phase_dir(settings, "tokenizer")
    for i in range(5):
        (first / f"shard{i}.bin").write_bytes(f"shard {i}".encode())
    utrain.store.consolidate(first, settings)
    second = _phase_dir(settings, "pretrain")
    subprocess.run(["cp", "-rl", f"{first}/.", str(second)], check=True)
    (second / "model.bin").write_bytes(b"weights")

    hashed: list[pathlib.Path] = []
    real = utrain.store._sha256  # pyright: ignore[reportPrivateUsage]

    def counted(path: pathlib.Path) -> str:
        hashed.append(path)
        return real(path)

    monkeypatch.setattr(utrain.store, "_sha256", counted)
    utrain.store.consolidate(second, settings)
    assert hashed == [second / "model.bin"]

    # And a second pass has nothing left to do.
    hashed.clear()
    utrain.store.consolidate(second, settings)
    assert hashed == []


def test_consolidated_data_passes_the_check(
    root: tuple[utrain.config.Settings, sqlalchemy.orm.Session],
) -> None:
    settings, session = root
    run_id = "8" * 32
    _insert_done_attempt(session, run_id)
    session.execute(
        sqlalchemy.insert(utrain.db.run_phases).values(
            run_id=run_id, attempt=1, phase="pretrain", phase_order=0, status="done"
        )
    )
    phase = _phase_dir(settings, "pretrain")
    (phase / "model.bin").write_bytes(b"weights")
    (phase / "copy.bin").write_bytes(b"weights")
    utrain.store.consolidate(phase, settings)

    result = utrain.store.check(settings, session)

    assert (result.data_files, result.store_files, result.problems) == (2, 1, [])
