import os
import pathlib


def tail_lines(path: pathlib.Path, n: int) -> list[str]:
    """The last `n` lines of a file, read backwards from the end.

    Training logs run to hundreds of megabytes and every `run logs` or
    `phase show` re-reads one, so seek from the end in blocks rather than
    pulling the whole file into memory.

    The first block read may start mid-line, and mid-character; decoding is
    lenient and the partial leading line is dropped by the final slice, which
    is safe because reading stops only once more than `n` newlines are in hand.

    `n <= 0` means the whole file. That is what the callers have always done --
    a plain `lines[-n:]` slice, where `-0` is `0` and so selects everything.
    """
    if n <= 0:
        return path.read_text(errors="replace").splitlines()

    block = 64 * 1024
    chunks: list[bytes] = []
    newlines = 0
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        pos = f.tell()
        while pos > 0 and newlines <= n:
            size = min(block, pos)
            pos -= size
            f.seek(pos)
            chunk = f.read(size)
            chunks.append(chunk)
            newlines += chunk.count(b"\n")

    data = b"".join(reversed(chunks))
    return data.decode(errors="replace").splitlines()[-n:]
