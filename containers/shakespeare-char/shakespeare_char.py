#!/usr/bin/env python3
"""Shakespeare character-level LLM preset implementing the utrain container contract."""

import argparse
import http.server
import json
import math
import pathlib
import sys
import time
import urllib.request

import torch
import torch.nn as nn
import wandb
import yaml

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)

DESCRIBE = {
    "name": "shakespeare-char",
    "version": "1.0.0",
    "phases": [
        {"name": "tokenizer", "label": "Download & Tokenize"},
        {"name": "pretrain", "label": "Train Character LM"},
    ],
    "phase_order": ["tokenizer", "pretrain"],
    "config_schema": {
        "globals": {
            "groups": [
                {
                    "name": "model",
                    "label": "Model Architecture",
                    "fields": [
                        {
                            "key": "n_layer",
                            "label": "Layers",
                            "type": "int",
                            "default": 4,
                            "min": 1,
                            "max": 24,
                            "description": "Number of transformer blocks",
                        },
                        {
                            "key": "n_head",
                            "label": "Attention Heads",
                            "type": "int",
                            "default": 4,
                            "min": 1,
                            "max": 16,
                            "description": "Number of attention heads",
                        },
                        {
                            "key": "n_embd",
                            "label": "Embedding Dim",
                            "type": "int",
                            "default": 128,
                            "min": 32,
                            "max": 1024,
                            "description": "Model embedding dimension",
                        },
                        {
                            "key": "block_size",
                            "label": "Context Length",
                            "type": "int",
                            "default": 256,
                            "min": 64,
                            "max": 2048,
                            "description": "Maximum sequence length",
                        },
                    ],
                }
            ]
        },
        "phases": {
            "pretrain": {
                "groups": [
                    {
                        "name": "training",
                        "label": "Training",
                        "fields": [
                            {
                                "key": "max_iters",
                                "label": "Training Steps",
                                "type": "int",
                                "default": 5000,
                                "min": 100,
                                "max": 100000,
                            },
                            {
                                "key": "batch_size",
                                "label": "Batch Size",
                                "type": "int",
                                "default": 64,
                                "min": 1,
                                "max": 512,
                            },
                            {
                                "key": "learning_rate",
                                "label": "Learning Rate",
                                "type": "float",
                                "default": 3e-4,
                            },
                            {
                                "key": "eval_interval",
                                "label": "Eval Every N Steps",
                                "type": "int",
                                "default": 500,
                                "min": 1,
                            },
                            {
                                "key": "generate_len",
                                "label": "Generate Length (serve)",
                                "type": "int",
                                "default": 200,
                                "min": 10,
                                "max": 2000,
                            },
                        ],
                    }
                ]
            }
        },
    },
    "can_serve": True,
}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class _CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int) -> None:
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        head_size = C // self.n_head
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (head_size**-0.5)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)  # type: ignore[no-any-return]


class _Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = _CausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class CharLM(nn.Module):
    def __init__(
        self, vocab_size: int, n_embd: int, n_layer: int, n_head: int, block_size: int
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[_Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.blocks(tok + pos)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.8
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_control(control_path: pathlib.Path) -> str:
    try:
        data = json.loads(control_path.read_text())
        return str(data.get("action", "continue"))
    except Exception:
        return "continue"


def _load_config(run_dir: pathlib.Path) -> dict[str, object]:
    cfg_path = run_dir / "config.yaml"
    if cfg_path.exists():
        return dict(yaml.safe_load(cfg_path.read_text()) or {})
    return {}


def _flatten_config(cfg: dict[str, object], phase: str) -> dict[str, object]:
    """Merge the utrain config into the flat namespace this phase reads.

    utrain writes config.yaml with keys grouped under a ``globals`` section and
    a per-phase ``phases.<phase>`` section. Within a section a value may be a
    field group (a dict, e.g. ``globals.model``) whose members are the real
    keys, or a bare field. The container reads flat keys, so for the phase being
    run we merge top-level scalars + ``globals`` + ``phases.<phase>``, expanding
    any group dict one level. Phase values win over globals on collision.
    """
    flat: dict[str, object] = {k: v for k, v in cfg.items() if k not in ("globals", "phases")}

    def _merge(section: object) -> None:
        if not isinstance(section, dict):
            return
        for key, value in section.items():
            if isinstance(value, dict):
                flat.update(value)
            else:
                flat[key] = value

    _merge(cfg.get("globals"))
    phases_section = cfg.get("phases")
    if isinstance(phases_section, dict):
        _merge(phases_section.get(phase))

    return flat


def _get_int(cfg: dict[str, object], key: str, default: int) -> int:
    val = cfg.get(key, default)
    return int(val)  # type: ignore[arg-type]


def _get_float(cfg: dict[str, object], key: str, default: float) -> float:
    val = cfg.get(key, default)
    return float(val)  # type: ignore[arg-type]


def _get_str(cfg: dict[str, object], key: str, default: str) -> str:
    val = cfg.get(key, default)
    return str(val)


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _estimate_mfu(n_params: int, tokens_per_sec: float, device: torch.device) -> float:
    """Rough MFU: 6 * N * tokens/s / peak_flops."""
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        # fp16 peak flops
        peak_flops = props.multi_processor_count * 128 * 2 * 1e12  # rough Ampere estimate
    else:
        peak_flops = 1e12  # 1 TFLOP placeholder for CPU
    return 6.0 * n_params * tokens_per_sec / peak_flops


# ---------------------------------------------------------------------------
# Phase implementations
# ---------------------------------------------------------------------------


def _run_tokenizer(run_dir: pathlib.Path, run_id: str) -> bool:
    run = wandb.init(project="tokenizer", id=run_id, dir=str(run_dir))
    run.log({"_phase_event": "tokenizer/started"})
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    input_path = data_dir / "input.txt"

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
    vocab_path = data_dir / "vocab.json"
    vocab_path.write_text(json.dumps(vocab))

    vocab_size = len(chars)
    print(f"Vocab size: {vocab_size}, corpus length: {len(text):,} chars", flush=True)
    run.log({"vocab_size": float(vocab_size), "corpus_chars": float(len(text))})
    run.log({"_phase_event": "tokenizer/completed"})
    run.finish(exit_code=0)
    return True


def _run_pretrain(run_dir: pathlib.Path, run_id: str, cfg: dict[str, object]) -> bool:
    run = wandb.init(project="pretrain", id=run_id, config=cfg, dir=str(run_dir))
    run.log({"_phase_event": "pretrain/started"})

    # Config
    n_layer = _get_int(cfg, "n_layer", 4)
    n_head = _get_int(cfg, "n_head", 4)
    n_embd = _get_int(cfg, "n_embd", 128)
    block_size = _get_int(cfg, "block_size", 256)
    max_iters = _get_int(cfg, "max_iters", 5000)
    batch_size = _get_int(cfg, "batch_size", 64)
    learning_rate = _get_float(cfg, "learning_rate", 3e-4)
    eval_interval = _get_int(cfg, "eval_interval", 500)

    # Load vocab + data
    vocab_path = run_dir / "data" / "vocab.json"
    vocab = json.loads(vocab_path.read_text())
    stoi: dict[str, int] = vocab["stoi"]
    vocab_size = len(stoi)
    text = (run_dir / "data" / "input.txt").read_text(encoding="utf-8")
    data = torch.tensor([stoi[c] for c in text if c in stoi], dtype=torch.long)
    n_train = int(0.9 * len(data))
    train_data = data[:n_train]
    val_data = data[n_train:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    model = CharLM(vocab_size, n_embd, n_layer, n_head, block_size).to(device)
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
        if _read_control(run_dir / "control.json") == "stop":
            run.log({"_phase_event": "pretrain/failed"})
            run.finish(exit_code=1)
            return False

        xb, yb = get_batch("train")
        _, loss_or_none = model(xb, yb)
        assert loss_or_none is not None
        loss = loss_or_none
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
                str(run_dir / "model.pt"),
            )
        elif time.time() - last_metric_time >= 10.0:
            run.log({"loss": loss.item()}, step=step, commit=True)
            last_metric_time = time.time()

    run.log({"_phase_event": "pretrain/completed"})
    run.finish(exit_code=0)
    return True


# ---------------------------------------------------------------------------
# Serve
# ---------------------------------------------------------------------------


def _load_model(run_dir: pathlib.Path) -> tuple[CharLM, dict[str, int], dict[str, int], int]:
    ckpt = torch.load(str(run_dir / "model.pt"), map_location="cpu", weights_only=True)
    model = CharLM(
        ckpt["vocab_size"], ckpt["n_embd"], ckpt["n_layer"], ckpt["n_head"], ckpt["block_size"]
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    vocab = json.loads((run_dir / "data" / "vocab.json").read_text())
    return model, vocab["stoi"], vocab["itos"], ckpt["block_size"]


class _ChatHandler(http.server.BaseHTTPRequestHandler):
    model: CharLM
    stoi: dict[str, int]
    itos: dict[str, int]  # actually int keys as strings from JSON
    generate_len: int

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_POST(self) -> None:
        if self.path != "/chat":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode()
        try:
            msg = json.loads(body).get("message", "")
        except Exception:
            msg = ""

        # Encode prompt → generate → decode
        prompt = msg if msg else "\n"
        encoded = [self.stoi.get(c, 0) for c in prompt]
        idx = torch.tensor([encoded], dtype=torch.long)
        with torch.no_grad():
            out = self.model.generate(idx, self.generate_len)
        generated = "".join(self.itos.get(str(i), "?") for i in out[0].tolist()[len(encoded) :])
        reply = json.dumps({"reply": generated})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply.encode())


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


def cmd_describe() -> None:
    print(json.dumps(DESCRIBE))


def cmd_check_compat() -> None:
    cuda_ok = torch.cuda.is_available()
    details = f"CUDA available: {cuda_ok}. Preset runs on CPU or GPU."
    print(json.dumps({"compatible": True, "details": details}))


def cmd_run(run_dir: pathlib.Path, phase: str) -> None:
    cfg = _flatten_config(_load_config(run_dir), phase)
    run_id = _get_str(cfg, "run_id", "")
    if phase == "tokenizer":
        ok = _run_tokenizer(run_dir, run_id)
    elif phase == "pretrain":
        ok = _run_pretrain(run_dir, run_id, cfg)
    else:
        print(f"unknown phase: {phase}", file=sys.stderr)
        sys.exit(1)
    sys.exit(0 if ok else 1)


def cmd_serve(run_dir: pathlib.Path, port: int) -> None:
    cfg = _flatten_config(_load_config(run_dir), "pretrain")
    generate_len = _get_int(cfg, "generate_len", 200)
    model, stoi, itos, _ = _load_model(run_dir)

    _ChatHandler.model = model
    _ChatHandler.stoi = stoi
    _ChatHandler.itos = itos  # type: ignore[assignment]
    _ChatHandler.generate_len = generate_len

    print(f"Serving on port {port}", flush=True)
    server = http.server.HTTPServer(("", port), _ChatHandler)
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("describe")
    sub.add_parser("check-compat")
    run_p = sub.add_parser("run")
    run_p.add_argument("run_dir", type=pathlib.Path)
    run_p.add_argument("--phase", required=True)
    serve_p = sub.add_parser("serve")
    serve_p.add_argument("run_dir", type=pathlib.Path)
    serve_p.add_argument("--port", type=int, default=8080)

    args = parser.parse_args()
    if args.cmd == "describe":
        cmd_describe()
    elif args.cmd == "check-compat":
        cmd_check_compat()
    elif args.cmd == "run":
        cmd_run(args.run_dir, args.phase)
    elif args.cmd == "serve":
        cmd_serve(args.run_dir, args.port)


if __name__ == "__main__":
    main()
