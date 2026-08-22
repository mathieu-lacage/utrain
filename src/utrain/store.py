from . import config, types


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
