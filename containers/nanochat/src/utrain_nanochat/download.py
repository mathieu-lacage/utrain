"""The `download` phase: fetch the pretraining corpus and the SFT/RL datasets.

Not cacheable. `check-cache` would have to name the sha256 of every shard before
fetching it, and nanochat's ClimbMix shards are not hash-pinned upstream -- see
the note in `describe.py`.

The task datasets (SmolTalk, MMLU, GSM8K) are pulled here too, though nothing
needs them until `sft`. They are small next to the corpus, they land in the same
data dir and so carry forward for free, and fetching them now means a network
problem surfaces in the first minute of a run rather than an hour in.
"""

import wandb

from . import config, paths, upstream

# The exact (repo, subset, split) triples `chat_sft` and `chat_rl` ask for.
# Fetched through nanochat's own `load_hub_dataset`, which is what fills the
# cache, rather than by constructing the Task classes -- one less signature to
# track across an upstream bump.
_DATASETS = (
    ("HuggingFaceTB/smol-smoltalk", "default", "train"),
    ("HuggingFaceTB/smol-smoltalk", "default", "test"),
    ("cais/mmlu", "all", "auxiliary_train"),
    ("cais/mmlu", "all", "test"),
    ("openai/gsm8k", "main", "train"),
    ("openai/gsm8k", "main", "test"),
)

_PREFETCH = (
    "import tasks.common\n"
    f"for repo, subset, split in {_DATASETS!r}:\n"
    '    print(f"Prefetching {repo} {subset}/{split}...", flush=True)\n'
    "    rows = len(tasks.common.load_hub_dataset(repo, subset, split=split))\n"
    '    print(f"  {rows} rows", flush=True)\n'
)


def _corpus(p: paths.Paths) -> tuple[int, float]:
    """Shards on disk and their total size.

    Counted rather than taken from the config: nanochat fetches the validation
    shard on top of the `-n` training shards it was asked for.
    """
    root = upstream.base_dir(p) / "base_data_climbmix"
    if not root.exists():
        return 0, 0.0
    shards = [f for f in root.rglob("*.parquet") if f.is_file()]
    return len(shards), float(sum(f.stat().st_size for f in shards))


def run(p: paths.Paths, run_id: str, cfg: dict[str, object]) -> bool:
    # This phase owns the wandb run: nothing it launches logs metrics of its
    # own, so there is no second writer to collide with. The training phases do
    # the opposite -- see `pretrain.py`.
    run = wandb.init(project="download", id=run_id, config=cfg, dir=str(p.run_dir))

    num_shards = config.get_int(cfg, "num_shards", 8)
    num_workers = config.get_int(cfg, "num_workers", 4)

    ok = upstream.run_script(
        p,
        "download",
        run_id,
        "nanochat.dataset",
        ["-n", str(num_shards), "-w", str(num_workers)],
    )
    if not ok:
        run.finish(exit_code=1)
        return False

    shards, corpus_bytes = _corpus(p)
    print(f"Corpus on disk: {corpus_bytes:,.0f} bytes across {shards} shards", flush=True)
    run.log({"shards": float(shards), "corpus_bytes": corpus_bytes})

    if config.get_bool(cfg, "prefetch_tasks", True):
        if not upstream.run_code(p, "download", run_id, _PREFETCH):
            run.finish(exit_code=1)
            return False

    run.finish(exit_code=0)
    return True
