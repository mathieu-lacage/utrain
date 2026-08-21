"""The `pretrain` phase: train the character LM and checkpoint it into /utrain/data."""

import json
import math
import time

import torch
import torch.nn as nn
import wandb

from . import config, paths
from . import model as model_mod


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _estimate_mfu(n_params: int, tokens_per_sec: float, device: torch.device) -> float:
    """Rough MFU: 6 * N * tokens/s / peak_flops."""
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)  # type: ignore[reportUnknownMemberType]
        # fp16 peak flops
        sm_count: int = props.multi_processor_count  # type: ignore[reportUnknownMemberType]
        peak_flops = sm_count * 128 * 2 * 1e12  # rough Ampere estimate
    else:
        peak_flops = 1e12  # 1 TFLOP placeholder for CPU
    return 6.0 * n_params * tokens_per_sec / peak_flops


def run(run_id: str, cfg: dict[str, object]) -> bool:
    run = wandb.init(project="pretrain", id=run_id, config=cfg, dir=str(paths.RUN_DIR))
    run.log({"_phase_event": "pretrain/started"})

    # Config
    n_layer = config.get_int(cfg, "n_layer", 4)
    n_head = config.get_int(cfg, "n_head", 4)
    n_embd = config.get_int(cfg, "n_embd", 128)
    block_size = config.get_int(cfg, "block_size", 256)
    max_iters = config.get_int(cfg, "max_iters", 5000)
    batch_size = config.get_int(cfg, "batch_size", 64)
    learning_rate = config.get_float(cfg, "learning_rate", 3e-4)
    eval_interval = config.get_int(cfg, "eval_interval", 500)

    # Load vocab + data
    vocab_path = paths.DATA_DIR / "vocab.json"
    vocab = json.loads(vocab_path.read_text())
    stoi: dict[str, int] = vocab["stoi"]
    vocab_size = len(stoi)
    text = (paths.DATA_DIR / "input.txt").read_text(encoding="utf-8")
    data = torch.tensor([stoi[c] for c in text if c in stoi], dtype=torch.long)
    n_train = int(0.9 * len(data))
    train_data = data[:n_train]
    val_data = data[n_train:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    model = model_mod.CharLM(vocab_size, n_embd, n_layer, n_head, block_size).to(device)
    n_params = _count_params(model)
    print(f"Model parameters: {n_params:,}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
        src = train_data if split == "train" else val_data
        ix = torch.randint(len(src) - block_size, (batch_size,))
        x = torch.stack([src[i : i + block_size] for i in ix]).to(device)
        y = torch.stack([src[i + 1 : i + block_size + 1] for i in ix]).to(device)
        return x, y

    @torch.no_grad()
    def estimate_loss() -> tuple[float, float]:
        model.eval()
        losses = {"train": 0.0, "val": 0.0}
        for split in ("train", "val"):
            total = 0.0
            for _ in range(20):
                xb, yb = get_batch(split)
                _, eval_loss = model(xb, yb)
                assert eval_loss is not None
                total += eval_loss.item()
            losses[split] = total / 20
        model.train()
        return losses["train"], losses["val"]

    model.train()
    step_times: list[float] = []
    t0 = time.time()
    last_metric_time = time.time()

    for step in range(max_iters):
        if config.read_control() == "stop":
            run.log({"_phase_event": "pretrain/failed"})
            run.finish(exit_code=1)
            return False

        xb, yb = get_batch("train")
        _, loss_or_none = model(xb, yb)
        assert loss_or_none is not None
        loss = loss_or_none
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  # type: ignore[reportUnknownMemberType]

        t1 = time.time()
        step_times.append(t1 - t0)
        t0 = t1
        if len(step_times) > 50:
            step_times.pop(0)

        if step % eval_interval == 0 or step == max_iters - 1:
            train_loss, val_loss = estimate_loss()
            bpb = val_loss / math.log(2)
            tokens_per_sec = (batch_size * block_size) / (sum(step_times) / len(step_times))
            mfu = _estimate_mfu(n_params, tokens_per_sec, device)
            run.log(
                {
                    "loss": val_loss,
                    "bpb": bpb,
                    "mfu": mfu,
                },
                step=step,
                commit=True,
            )
            last_metric_time = time.time()
            print(
                f"step {step:5d}/{max_iters}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"bpb={bpb:.4f} mfu={mfu:.4f} tok/s={tokens_per_sec:.0f}",
                flush=True,
            )
            # Save checkpoint
            torch.save(
                {
                    "model": model.state_dict(),
                    "n_layer": n_layer,
                    "n_head": n_head,
                    "n_embd": n_embd,
                    "block_size": block_size,
                    "vocab_size": vocab_size,
                },
                str(paths.DATA_DIR / "model.pt"),
            )
        elif time.time() - last_metric_time >= 10.0:
            run.log({"loss": loss.item()}, step=step, commit=True)
            last_metric_time = time.time()

    run.log({"_phase_event": "pretrain/completed"})
    run.finish(exit_code=0)
    return True
