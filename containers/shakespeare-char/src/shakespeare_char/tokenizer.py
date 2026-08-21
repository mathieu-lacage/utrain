"""The `tokenizer` phase: fetch the corpus and build the character vocabulary."""

import json
import urllib.request

import wandb

from . import paths

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def run(run_id: str) -> bool:
    run = wandb.init(project="tokenizer", id=run_id, dir=str(paths.RUN_DIR))
    run.log({"_phase_event": "tokenizer/started"})
    paths.DATA_DIR.mkdir(parents=True, exist_ok=True)
    input_path = paths.DATA_DIR / "input.txt"

    print("Downloading Shakespeare dataset...", flush=True)
    try:
        urllib.request.urlretrieve(SHAKESPEARE_URL, str(input_path))
    except Exception as e:
        print(f"Download failed: {e}", flush=True)
        run.log({"_phase_event": "tokenizer/failed"})
        run.finish(exit_code=1)
        return False

    text = input_path.read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi_map = {c: i for i, c in enumerate(chars)}
    itos_map = {i: c for i, c in enumerate(chars)}
    vocab = {"chars": chars, "stoi": stoi_map, "itos": itos_map}
    vocab_path = paths.DATA_DIR / "vocab.json"
    vocab_path.write_text(json.dumps(vocab))

    vocab_size = len(chars)
    print(f"Vocab size: {vocab_size}, corpus length: {len(text):,} chars", flush=True)
    run.log({"vocab_size": float(vocab_size), "corpus_chars": float(len(text))})
    run.log({"_phase_event": "tokenizer/completed"})
    run.finish(exit_code=0)
    return True
