"""The `serve` command: a tiny HTTP chat endpoint over the trained checkpoint."""

import http.server
import json

import torch

from . import config, paths
from . import model as model_mod


def _load_model() -> tuple[model_mod.CharLM, dict[str, int], dict[str, str], int]:
    ckpt = torch.load(str(paths.DATA_DIR / "model.pt"), map_location="cpu", weights_only=True)
    model = model_mod.CharLM(
        ckpt["vocab_size"], ckpt["n_embd"], ckpt["n_layer"], ckpt["n_head"], ckpt["block_size"]
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    vocab = json.loads((paths.DATA_DIR / "vocab.json").read_text())
    return model, vocab["stoi"], vocab["itos"], ckpt["block_size"]


class _ChatHandler(http.server.BaseHTTPRequestHandler):
    model: model_mod.CharLM
    stoi: dict[str, int]
    itos: dict[str, str]  # JSON turned the int keys of the vocab into strings
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
        token_ids: list[int] = out[0].tolist()  # type: ignore[reportUnknownMemberType]
        generated = "".join(self.itos.get(str(i), "?") for i in token_ids[len(encoded) :])
        reply = json.dumps({"reply": generated})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply.encode())


def serve(port: int) -> None:
    cfg = config.flatten(config.load(), "pretrain")
    generate_len = config.get_int(cfg, "generate_len", 200)
    model, stoi, itos, _ = _load_model()

    _ChatHandler.model = model
    _ChatHandler.stoi = stoi
    _ChatHandler.itos = itos
    _ChatHandler.generate_len = generate_len

    print(f"Serving on port {port}", flush=True)
    server = http.server.HTTPServer(("", port), _ChatHandler)
    server.serve_forever()
