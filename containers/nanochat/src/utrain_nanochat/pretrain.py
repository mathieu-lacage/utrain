"""The `pretrain` phase: train nanochat's base model, then evaluate it.

Unlike `download` and `tokenizer`, this phase opens no wandb run of its own.
`scripts.base_train` already logs `train/loss`, `val/bpb`, `train/mfu` and the
CORE metric through wandb, and the bridge on the child's PYTHONPATH files those
under this phase and this run -- see `wandb_bridge/sitecustomize.py`. naw
refuses to open an rtsdb that already exists, so a second `wandb.init`
against the same phase and run id would fail the phase outright.

`--run` goes to the training script alone; nanochat skips wandb entirely when
it is left at `dummy`. `base_eval` takes no such flag and logs nothing through
wandb, so its numbers reach the phase log rather than a curve.
"""

from . import config, paths, upstream


def run(p: paths.Paths, run_id: str, cfg: dict[str, object]) -> bool:
    args = [
        f"--run={run_id}",
        f"--depth={config.get_int(cfg, 'depth', 8)}",
        f"--max-seq-len={config.get_int(cfg, 'max_seq_len', 2048)}",
        f"--device-batch-size={config.get_int(cfg, 'device_batch_size', 8)}",
        f"--target-param-data-ratio={config.get_float(cfg, 'target_param_data_ratio', 12.0)}",
        f"--eval-every={config.get_int(cfg, 'eval_every', 250)}",
        f"--eval-tokens={config.get_int(cfg, 'eval_tokens', 2097152)}",
        f"--core-metric-every={config.get_int(cfg, 'core_metric_every', 2000)}",
        f"--sample-every={config.get_int(cfg, 'sample_every', 2000)}",
    ]
    # Both are passed only when set: nanochat's own -1 means "work it out from
    # the model", and repeating that here would just duplicate its arithmetic.
    num_iterations = config.get_int(cfg, "num_iterations", -1)
    if num_iterations > 0:
        args.append(f"--num-iterations={num_iterations}")
    total_batch_size = config.get_int(cfg, "total_batch_size", -1)
    if total_batch_size > 0:
        args.append(f"--total-batch-size={total_batch_size}")

    if not upstream.run_script(p, "pretrain", run_id, "scripts.base_train", args):
        return False

    if config.get_bool(cfg, "run_eval", True):
        eval_args = [f"--device-batch-size={config.get_int(cfg, 'device_batch_size', 8)}"]
        if not upstream.run_script(p, "pretrain", run_id, "scripts.base_eval", eval_args):
            print("base_eval failed; the trained checkpoint is still in place.", flush=True)

    return True
