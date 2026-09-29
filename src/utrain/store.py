import hashlib
import os
import pathlib

import sqlalchemy
import sqlalchemy.orm

from . import address, config, exceptions, types
from . import db as dbmod

# How much of a file `_sha256` reads per iteration: big enough that model
# snapshots do not turn into millions of tiny reads, small enough that a whole
# file never sits in memory twice over.
_HASH_CHUNK = 1024 * 1024


def _sha256(fpath: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with fpath.open("rb") as f:
        for chunk in iter(lambda: f.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def consolidate(data_dir: pathlib.Path, settings: config.Settings) -> None:
    """Hardlink every file under a phase's data dir into the store.

    Afterwards each file shares an inode with the 0o444 store file named after
    its content's sha256 -- what `check` verifies of a `done` phase. The
    orchestrator calls this as each phase succeeds, before recording it done.

    Only new files cost anything. A file carried forward from a phase already
    consolidated (`cp -rl`), or served from the cache, is a store inode
    already, and is recognised by that rather than hashed again. A new file
    whose content the store lacks becomes the store file itself, by a link
    rather than a copy; one whose content is there already is swapped for a
    link to it, atomically, so the data file is never missing.
    """
    store_dir = settings.data_dir / "store"
    store_dir.mkdir(parents=True, exist_ok=True)
    if not data_dir.exists():
        return
    in_store: set[tuple[int, int]] = set()
    for fpath in store_dir.iterdir():
        stat = fpath.stat()
        in_store.add((stat.st_dev, stat.st_ino))

    for fpath in sorted(data_dir.rglob("*")):
        if fpath.is_symlink() or not fpath.is_file():
            continue
        stat = fpath.stat()
        if (stat.st_dev, stat.st_ino) in in_store:
            continue
        store_path = store_dir / _sha256(fpath)
        try:
            os.link(fpath, store_path)
        except FileExistsError:
            # The content is there already, from another file, phase or run:
            # link to that one instead, replacing the data file in one step.
            tmp = fpath.with_name(f".{fpath.name}.utrain-link")
            tmp.unlink(missing_ok=True)
            os.link(store_path, tmp)
            os.replace(tmp, fpath)
        else:
            os.chmod(store_path, 0o444)
        linked = store_path.stat()
        in_store.add((linked.st_dev, linked.st_ino))


def _resolve_scope(
    scope: str,
    session: sqlalchemy.orm.Session,
) -> tuple[str, int | None, str | None]:
    """Parse a fully-qualified id into (run_id, attempt, phase).

    Three forms, the same addresses the CLI prints in its tables' first
    column: a run (`RUN_ID`, prefix ok), an attempt (`RUN_ID/N`) or a phase
    (`RUN_ID/N/PHASE`). Unlike the phase commands' addresses, a phase
    component always carries its attempt: the check needs to know which
    attempt's data dir to look in, so `RUN_ID/PHASE` is rejected rather than
    guessed. Which attempt and phase exist is checked against the database
    below; the run prefix has been resolved by the shared parser already.
    """
    run_id, attempt, phase = address.parse(scope, session, phase_needs_attempt=True)

    if attempt is not None:
        status = session.execute(
            sqlalchemy.select(dbmod.run_attempts.c.status).where(
                (dbmod.run_attempts.c.run_id == run_id) & (dbmod.run_attempts.c.attempt == attempt)
            )
        ).scalar_one_or_none()
        if status is None:
            raise exceptions.UI(f"attempt {attempt} of run '{run_id}' not found")

    if phase is not None:
        found = session.execute(
            sqlalchemy.select(dbmod.run_phases.c.phase).where(
                (dbmod.run_phases.c.run_id == run_id)
                & (dbmod.run_phases.c.attempt == attempt)
                & (dbmod.run_phases.c.phase == phase)
            )
        ).scalar_one_or_none()
        if found is None:
            raise exceptions.UI(f"phase '{phase}' not found in attempt {attempt} of run '{run_id}'")

    return run_id, attempt, phase


def summary(settings: config.Settings) -> types.StoreSummary:
    """How big the store is, and how much of it `gc` would reclaim.

    A stat per file and no hashing, so it is cheap enough to show on a screen
    that refreshes: `check` is the thorough one.
    """
    store_dir = settings.data_dir / "store"
    files = size = orphaned = orphaned_bytes = 0
    if store_dir.exists():
        for fpath in store_dir.iterdir():
            if not fpath.is_file():
                continue
            stat = fpath.stat()
            files += 1
            size += stat.st_size
            if stat.st_nlink == 1:
                orphaned += 1
                orphaned_bytes += stat.st_size
    return types.StoreSummary(
        files=files, bytes=size, orphaned=orphaned, orphaned_bytes=orphaned_bytes
    )


def gc(settings: config.Settings) -> types.GcResult:
    """Drop store files no run still hardlinks, and report what was reclaimed."""
    store_dir = settings.data_dir / "store"
    if not store_dir.exists():
        return types.GcResult(removed=0, reclaimed_bytes=0)

    removed = 0
    reclaimed = 0
    for fpath in store_dir.iterdir():
        if not fpath.is_file():
            continue
        stat = fpath.stat()
        if stat.st_nlink == 1:
            reclaimed += stat.st_size
            fpath.unlink()
            removed += 1

    return types.GcResult(removed=removed, reclaimed_bytes=reclaimed)


def check(
    settings: config.Settings,
    session: sqlalchemy.orm.Session,
    *,
    hash_contents: bool = True,
    verbose: bool = False,
    scope: str | None = None,
) -> types.StoreCheckResult:
    """Verify runs' data is hardlinked into the content-addressed store.

    This is the invariant `consolidate` establishes as each phase succeeds:
    every file under a done phase's data dir shares an inode with the store
    file named after its content's sha256, and every store file's content
    hashes to its own name.

    Both sides are checked, but the store is hashed only once: a data file
    that shares an inode with a verified store file is verified by
    transitivity, so phase data -- the bulk of a run -- is never read here.
    With ``hash_contents`` off only the links are checked; the names are then
    trusted to be hashes.

    Phases still going, failed or stopped are skipped: their data is
    legitimately not linked, because only a phase that succeeds is
    consolidated. The done phases of a failed or stopped attempt are checked
    like any other. An explicit ``scope`` that names nothing done is an error
    rather than a silent pass -- the caller asked about exactly that data, so
    "nothing to check" would read as ok.

    ``scope`` narrows the check to a run (`RUN_ID`, prefix ok), an attempt
    (`RUN_ID/N`) or a single phase (`RUN_ID/N/PHASE`); the default checks
    every run. With ``verbose``, the result also carries one ``StoreLink``
    per checked file, naming the store file it is hardlinked to.
    """
    data_dir = settings.data_dir
    store_dir = data_dir / "store"

    problems: list[types.StoreProblem] = []
    store_inodes: dict[tuple[int, int], str] = {}
    store_files = 0
    orphaned = 0
    if store_dir.exists():
        for fpath in sorted(store_dir.iterdir()):
            if not fpath.is_file():
                continue
            store_files += 1
            if hash_contents:
                digest = _sha256(fpath)
                if digest != fpath.name:
                    problems.append(
                        types.StoreProblem(
                            pathlib.Path("store") / fpath.name,
                            f"content hashes to {digest}, not its name",
                        )
                    )
            stat = fpath.stat()
            store_inodes[(stat.st_dev, stat.st_ino)] = fpath.name
            # The same criterion `gc` uses: nothing links this file any more.
            if stat.st_nlink == 1:
                orphaned += 1

    scope_run: str | None = None
    scope_attempt: int | None = None
    scope_phase: str | None = None
    if scope is not None:
        scope_run, scope_attempt, scope_phase = _resolve_scope(scope, session)

    query = sqlalchemy.select(
        dbmod.run_phases.c.run_id, dbmod.run_phases.c.attempt, dbmod.run_phases.c.phase
    ).where(dbmod.run_phases.c.status == "done")
    if scope_run is not None:
        query = query.where(dbmod.run_phases.c.run_id == scope_run)
    if scope_attempt is not None:
        query = query.where(dbmod.run_phases.c.attempt == scope_attempt)
    if scope_phase is not None:
        query = query.where(dbmod.run_phases.c.phase == scope_phase)
    done = session.execute(
        query.order_by(
            dbmod.run_phases.c.run_id, dbmod.run_phases.c.attempt, dbmod.run_phases.c.phase
        )
    ).all()

    if scope is not None and not done:
        not_yet = "its data is not in the store yet"
        if scope_phase is not None:
            raise exceptions.UI(
                f"phase '{scope_phase}' of attempt {scope_attempt} of run '{scope_run}' "
                f"is not done; {not_yet}"
            )
        if scope_attempt is not None:
            raise exceptions.UI(
                f"attempt {scope_attempt} of run '{scope_run}' has no completed phase; {not_yet}"
            )
        raise exceptions.UI(f"run '{scope_run}' has no completed phase; {not_yet}")

    checked_attempts: set[tuple[str, int]] = set()
    data_files = 0
    links: list[types.StoreLink] = []
    for run_id, attempt, phase in done:
        root = settings.runs_dir / str(run_id) / "attempt" / str(attempt) / "data" / str(phase)
        if not root.exists():
            continue
        checked_attempts.add((str(run_id), int(attempt)))
        for fpath in sorted(root.rglob("*")):
            if not fpath.is_file():
                continue
            data_files += 1
            stat = fpath.stat()
            store_name = store_inodes.get((stat.st_dev, stat.st_ino))
            if store_name is None:
                problems.append(
                    types.StoreProblem(fpath.relative_to(data_dir), "not a hardlink into the store")
                )
            elif verbose:
                links.append(
                    types.StoreLink(
                        data=fpath.relative_to(data_dir),
                        store=pathlib.Path("store") / store_name,
                    )
                )

    return types.StoreCheckResult(
        attempts=len(checked_attempts),
        data_files=data_files,
        store_files=store_files,
        orphaned=orphaned,
        problems=problems,
        links=links,
    )
