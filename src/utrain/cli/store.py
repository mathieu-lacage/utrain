from .. import config
from . import output


def gc(settings: config.Settings) -> None:
    store_dir = settings.data_dir / "store"
    if not store_dir.exists():
        print("removed 0 file(s), 0.0 B reclaimed")
        return

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

    print(f"removed {removed} file(s), {output.human_size(reclaimed)} reclaimed")
