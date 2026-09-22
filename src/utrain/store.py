import hashlib
import pathlib

import sqlalchemy
import sqlalchemy.orm

from . import config, db as dbmod, exceptions, types

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


def _resolve_scope(
    scope: str,
    session: sqlalchemy.orm.Session,
) -> tuple[str, int | None, str | None]:
    """Parse a fully-qualified id into (run_id, attempt, phase).

    Three forms, the same addresses the CLI prints in its tables' first
    column: a run (`RUN_ID`, prefix ok), an attempt (`RUN_ID/N`) or a phase
    (`RUN_ID/N/PHASE`). Unlike phases' addresses, a phase component always
    carries its attempt: the check needs to know which attempt's data dir to
    look in, so `RUN_ID/PHASE` is rejected rather than guessed.
    """
    parts = scope.split("/")
    if len(parts) > 3:
        raise exceptions.UI(f"invalid address '{scope}'")

    run_id = dbmod.resolve_run_id(parts[0], session)

    attempt: int | None = None
    if len(parts) >= 2:
        if not parts[1].isdigit():
            raise exceptions.UI(f"expected attempt number, got '{parts[1]}'")
        attempt = int(parts[1])
        status = session.execute(
            sqlalchemy.select(dbmod.run_attempts.c.status).where(
                (dbmod.run_attempts.c.run_id == run_id)
                & (dbmod.run_attempts.c.attempt == attempt)
            )
        ).scalar_one_or_none()
        if status is None:
            raise exceptions.UI(f"attempt {attempt} of run '{run_id}' not found")

    phase: str | None = None
    if len(parts) == 3:
        phase = parts[2]
        found = session.execute(
            sqlalchemy.select(dbmod.run_phases.c.phase).where(
                (dbmod.run_phases.c.run_id == run_id)
                & (dbmod.run_phases.c.attempt == attempt)
                & (dbmod.run_phases.c.phase == phase)
            )
        ).scalar_one_or_none()
        if found is None:
            raise exceptions.UI(
                f"phase '{phase}' not found in attempt {attempt} of run '{run_id}'"
            )

    return run_id, attempt, phase


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

    This is the invariant `orchestrator._deduplicate_data` establishes once a
    run completes: every file under a done attempt's data dir shares an inode
    with the store file named after its content's sha256, and every store
    file's content hashes to its own name.

    Both sides are checked, but the store is hashed only once: a data file
    that shares an inode with a verified store file is verified by
    transitivity, so phase data -- the bulk of a run -- is never read here.
    With ``hash_contents`` off only the links are checked; the names are then
    trusted to be hashes.

    Attempts still going, failed or stopped are skipped: their data is
    legitimately not linked yet, because deduplication runs only when every
    phase of the attempt has finished. An explicit ``scope`` that names one of
    those attempts is an error rather than a silent pass -- the caller asked
    about exactly that data, so "nothing to check" would read as ok.

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
        dbmod.run_attempts.c.run_id, dbmod.run_attempts.c.attempt
    ).where(dbmod.run_attempts.c.status == "done")
    if scope_run is not None:
        query = query.where(dbmod.run_attempts.c.run_id == scope_run)
    if scope_attempt is not None:
        query = query.where(dbmod.run_attempts.c.attempt == scope_attempt)
    done = session.execute(query).all()

    if scope is not None and not done:
        # A validated attempt reaches here only when it is not done; a
        # run-only scope when none of its attempts are.
        if scope_attempt is not None:
            raise exceptions.UI(
                f"attempt {scope_attempt} of run '{scope_run}' is not done; "
                "its data is not in the store yet"
            )
        raise exceptions.UI(
            f"run '{scope_run}' has no completed attempt; its data is not in the store yet"
        )

    attempts = 0
    data_files = 0
    links: list[types.StoreLink] = []
    for run_id, attempt in done:
        attempt_data = settings.runs_dir / str(run_id) / "attempt" / str(attempt) / "data"
        # A phase scope looks in that phase's directory only; anything else
        # checks the whole attempt.
        roots = [attempt_data / scope_phase] if scope_phase is not None else [attempt_data]
        if not any(root.exists() for root in roots):
            continue
        attempts += 1
        for root in roots:
            if not root.exists():
                continue
            for fpath in sorted(root.rglob("*")):
                if not fpath.is_file():
                    continue
                data_files += 1
                stat = fpath.stat()
                store_name = store_inodes.get((stat.st_dev, stat.st_ino))
                if store_name is None:
                    problems.append(
                        types.StoreProblem(
                            fpath.relative_to(data_dir), "not a hardlink into the store"
                        )
                    )
                elif verbose:
                    links.append(
                        types.StoreLink(
                            data=fpath.relative_to(data_dir),
                            store=pathlib.Path("store") / store_name,
                        )
                    )

    return types.StoreCheckResult(
        attempts=attempts,
        data_files=data_files,
        store_files=store_files,
        orphaned=orphaned,
        problems=problems,
        links=links,
    )
