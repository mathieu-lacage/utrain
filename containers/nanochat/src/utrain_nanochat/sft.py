"""The `sft` phase: fine-tune the base model on conversations.

Teaches the model the chat special tokens, tool use and multiple choice, over a
mixture of SmolTalk, MMLU and GSM8K. The base checkpoint arrives through the
data dir from `pretrain`, and the datasets from the `task_data` cache `download`
warmed -- both under the same nanochat base dir.

Metrics come from the child, for the reason spelled out in `pretrain.py`.
"""

from . import config, paths, upstream


def run(p: paths.Paths, run_id: str, cfg: dict[str, object]) -> bool:
    args = [
        f"--run={run_id}",
        f"--max-seq-len={config.get_int(cfg, 'max_seq_len', 2048)}",
        f"--device-batch-size={config.get_int(cfg, 'device_batch_size', 4)}",
        f"--eval-every={config.get_int(cfg, 'eval_every', 200)}",
        f"--eval-tokens={config.get_int(cfg, 'eval_tokens', 2097152)}",
        f"--chatcore-every={config.get_int(cfg, 'chatcore_every', 200)}",
        f"--mmlu-epochs={config.get_int(cfg, 'mmlu_epochs', 3)}",
        f"--gsm8k-epochs={config.get_int(cfg, 'gsm8k_epochs', 4)}",
    ]
    # Both are passed only when set: nanochat's own -1 means "work it out from
    # the model", and repeating that here would just duplicate its arithmetic.
    num_iterations = config.get_int(cfg, "num_iterations", -1)
    if num_iterations > 0:
        args.append(f"--num-iterations={num_iterations}")
    total_batch_size = config.get_int(cfg, "total_batch_size", -1)
    if total_batch_size > 0:
        args.append(f"--total-batch-size={total_batch_size}")

    if not upstream.run_script(p, "sft", run_id, "scripts.chat_sft", args):
        return False

    if config.get_bool(cfg, "run_eval", True):
        if not upstream.run_script(p, "sft", run_id, "scripts.chat_eval", ["-i", "sft"]):
            print("chat_eval failed; the fine-tuned checkpoint is still in place.", flush=True)

    return True
