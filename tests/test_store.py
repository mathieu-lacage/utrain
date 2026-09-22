"""Unit tests for the content-addressed store.

`store.gc` and `store.check` both work purely off the filesystem plus the
attempts table, so they can be exercised by building a data directory by hand
-- no podman, so these run in CI, like test_query_layer.py.
"""

import hashlib
import os
import pathlib
import typing

import pytest
import sqlalchemy

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
    settings: utrain.config.Settings,
    run_id: str,
    attempt: int,
    phase: str,
    name: str,
    content: bytes,
) -> None:
    """Leave behind exactly what `orchestrator._deduplicate_data` leaves: a
    0o444 store file named after the content's hash, hardlinked into the
    phase's data dir."""
    sha = hashlib.sha256(content).hexdigest()
    store_path = settings.data_dir / "store" / sha
    store_path.parent.mkdir(parents=True, exist_ok=True)
    # As in `_deduplicate_data`: the store keeps one 0o444 file per distinct
    # content, so a second phase sharing it must not rewrite the file.
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
    _add_linked_data(settings, "a" * 32, 1, "tokenizer", "tok.txt", b"tok\n")
    _add_linked_data(settings, "a" * 32, 1, "tokenizer", "shared.txt", b"shared\n")
    _add_linked_data(settings, "a" * 32, 1, "pretrain", "shared.txt", b"shared\n")

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
    _add_linked_data(settings, run_id, 1, "pretrain", "model.bin", b"weights")
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
    _add_linked_data(settings, run_id, 1, "pretrain", "model.bin", b"weights")
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


def test_store_check_skips_attempts_that_are_not_done(
    root: tuple[utrain.config.Settings, sqlalchemy.orm.Session],
) -> None:
    settings, session = root
    run_id = "d" * 32
    _insert_run(session, run_id, status="running")
    _insert_attempt(session, run_id, 1, "running")
    # A running attempt's data is plain files: deduplication has not run yet.
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
    _add_linked_data(settings, "f" * 32, 1, "pretrain", "model.bin", b"weights")
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
    _add_linked_data(settings, run_id, 1, "tokenizer", "tok.txt", b"tok\n")
    # Two pretrain files share one store file: both get their own line.
    _add_linked_data(settings, run_id, 1, "pretrain", "shared.txt", b"shared\n")
    _add_linked_data(settings, run_id, 1, "pretrain", "shared2.txt", b"shared\n")

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
    _add_linked_data(settings, "ab" * 16, 1, "pretrain", "model.bin", b"weights")
    _add_linked_data(settings, "cd" * 16, 1, "pretrain", "model.bin", b"weights")

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
    _add_linked_data(settings, run_id, 1, "pretrain", "old.bin", b"old")
    _add_linked_data(settings, run_id, 2, "pretrain", "new.bin", b"new")

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
    _add_linked_data(settings, run_id, 1, "tokenizer", "tok.txt", b"tok\n")
    _add_linked_data(settings, run_id, 1, "pretrain", "model.bin", b"weights")

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
    _add_linked_data(settings, run_id, 1, "pretrain", "model.bin", b"weights")

    # The done attempt is checked, the running one skipped, as unscoped.
    result = utrain.store.check(settings, session, scope=run_id)
    assert result.attempts == 1
    assert result.data_files == 1

    # Asked about the running attempt directly, silence would read as ok.
    with pytest.raises(utrain.exceptions.UI, match=f"attempt 2 of run '{run_id}' is not done"):
        utrain.store.check(settings, session, scope=f"{run_id}/2")

    run_id = "6" * 32
    _insert_run(session, run_id, status="running")
    _insert_attempt(session, run_id, 1, "running")
    with pytest.raises(utrain.exceptions.UI, match=f"run '{run_id}' has no completed attempt"):
        utrain.store.check(settings, session, scope=run_id)
