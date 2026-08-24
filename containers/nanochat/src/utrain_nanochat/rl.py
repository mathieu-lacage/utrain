"""The `rl` phase: GRPO on GSM8K over the fine-tuned model.

This is reinforcement learning against a *verifiable* reward -- whether the
model's arithmetic came out right -- not RLHF: there is no preference data and
no reward model, and nanochat's variant drops the trust region and the PPO
clipping too.

Metrics come from the child, for the reason spelled out in `pretrain.py`. The
ones worth watching are `reward` and `pass@1`; nanochat reports `pass@k` up to
the device batch size, so a larger batch simply adds curves.
"""

from . import config, paths, upstream


def run(p: paths.Paths, run_id: str, cfg: dict[str, object]) -> bool:
    args = [
        f"--run={run_id}",
        f"--num-epochs={config.get_int(cfg, 'num_epochs', 1)}",
        f"--num-samples={config.get_int(cfg, 'num_samples', 16)}",
        f"--examples-per-step={config.get_int(cfg, 'examples_per_step', 16)}",
        f"--device-batch-size={config.get_int(cfg, 'device_batch_size', 8)}",
        f"--max-new-tokens={config.get_int(cfg, 'max_new_tokens', 256)}",
        f"--eval-every={config.get_int(cfg, 'eval_every', 60)}",
        f"--eval-examples={config.get_int(cfg, 'eval_examples', 100)}",
    ]
    return upstream.run_script(p, "rl", run_id, "scripts.chat_rl", args)
