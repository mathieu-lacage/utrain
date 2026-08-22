"""The `tokenizer` phase: build the character vocabulary from the corpus.

The corpus itself arrives from the `download` phase, through the data dir:
utrain seeds each phase's data dir from its predecessor's.
"""

import json

import wandb

from . import download, paths


def run(p: paths.Paths, run_id: str) -> bool:
    run = wandb.init(project="tokenizer", id=run_id, dir=str(p.run_dir))
    input_path = p.data_dir / download.CORPUS_FILENAME

    if not input_path.exists():
        print(
            f"{download.CORPUS_FILENAME} is missing from the data dir -- "
            "the download phase has to run first.",
            flush=True,
        )
        run.finish(exit_code=1)
        return False

    text = input_path.read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi_map = {c: i for i, c in enumerate(chars)}
    itos_map = {i: c for i, c in enumerate(chars)}
    vocab = {"chars": chars, "stoi": stoi_map, "itos": itos_map}
    vocab_path = p.data_dir / "vocab.json"
    vocab_path.write_text(json.dumps(vocab))

    vocab_size = len(chars)
    print(f"Vocab size: {vocab_size}, corpus length: {len(text):,} chars", flush=True)
    run.log({"vocab_size": float(vocab_size), "corpus_chars": float(len(text))})
    run.finish(exit_code=0)
    return True
