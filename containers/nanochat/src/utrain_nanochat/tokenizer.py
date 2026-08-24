"""The `tokenizer` phase: fit the BPE vocabulary on the downloaded corpus.

nanochat splits this in two -- `tok_train` fits the merges, `tok_eval` reports
the compression ratio it achieved against other tokenizers. Neither logs
metrics, so this phase owns the wandb run and logs the one number worth a curve.
"""

import wandb

from . import config, paths, upstream


def run(p: paths.Paths, run_id: str, cfg: dict[str, object]) -> bool:
    run = wandb.init(project="tokenizer", id=run_id, config=cfg, dir=str(p.run_dir))

    vocab_size = config.get_int(cfg, "vocab_size", 32768)
    max_chars = config.get_int(cfg, "max_chars", 2_000_000_000)
    doc_cap = config.get_int(cfg, "doc_cap", 10000)

    ok = upstream.run_script(
        p,
        "tokenizer",
        run_id,
        "scripts.tok_train",
        [
            f"--vocab-size={vocab_size}",
            f"--max-chars={max_chars}",
            f"--doc-cap={doc_cap}",
        ],
    )
    if not ok:
        run.finish(exit_code=1)
        return False

    run.log({"vocab_size": float(vocab_size), "training_chars": float(max_chars)})

    # Evaluation is reporting, not a gate: a tokenizer that trained but whose
    # eval fell over is still the artifact the next phase needs.
    if not upstream.run_script(p, "tokenizer", run_id, "scripts.tok_eval", []):
        print("tok_eval failed; the trained tokenizer is still in place.", flush=True)

    run.finish(exit_code=0)
    return True
