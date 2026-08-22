# Building a container

This walks through writing the smallest possible container that satisfies
the utrain [container contract](container-contract.md), running entirely
with fake/synthetic data so there's nothing to download and no GPU needed.
By the end you'll have a real image that `utrain` can pull, run, and plot
metrics for.

## 1. Write the container script

Create `demo_container.py`:

```python
#!/usr/bin/env python3
import argparse
import json
import pathlib
import random
import sys
import time

import wandb
import yaml

# Every path utrain gives you hangs off one root, at fixed locations beneath it.
# The root arrives as --utrain-root (utrain passes /utrain); the default lets you
# run this script straight from your shell, against ./run.
DEFAULT_ROOT = "run"

DESCRIBE = {
    "name": "demo",
    "version": "1.0.0",
    "phases": [{"name": "train", "label": "Training"}],
    "phase_order": ["train"],
    "config_schema": {
        "phases": {
            "train": {
                "groups": [
                    {
                        "name": "training",
                        "label": "Training",
                        "fields": [
                            {
                                "key": "steps",
                                "label": "Steps",
                                "type": "int",
                                "default": 50,
                            }
                        ],
                    }
                ]
            }
        }
    },
    "can_serve": False,
}


def _flatten_config(cfg: dict, phase: str) -> dict:
    flat = {k: v for k, v in cfg.items() if k not in ("globals", "phases")}

    def merge(section):
        if not isinstance(section, dict):
            return
        for key, value in section.items():
            if isinstance(value, dict):
                flat.update(value)
            else:
                flat[key] = value

    merge(cfg.get("globals"))
    phases = cfg.get("phases")
    if isinstance(phases, dict):
        merge(phases.get(phase))
    return flat


def _read_control(root: pathlib.Path) -> str:
    try:
        return json.loads((root / "control.json").read_text()).get("action", "continue")
    except Exception:
        return "continue"


def cmd_describe():
    print(json.dumps(DESCRIBE))


def cmd_check_compat():
    print(json.dumps({"compatible": True, "details": "always compatible"}))


def _run_train(root: pathlib.Path, cfg: dict) -> bool:
    run = wandb.init(project="demo", id=str(cfg.get("run_id", "")), dir=str(root))
    steps = int(cfg.get("steps", 50))
    ok = True
    for step in range(steps):
        if _read_control(root) == "stop":
            ok = False
            break
        loss = 3.0 * (0.95 ** step) + random.gauss(0, 0.02)
        run.log({"loss": loss}, step=step, commit=True)
        time.sleep(0.05)
    if ok:
        # Hand the result to later phases (and `serve`) through the data dir.
        (root / "data" / "model.txt").write_text(f"trained for {steps} steps\n")
    run.finish(exit_code=0 if ok else 1)
    return ok


def cmd_run(root: pathlib.Path, phase: str):
    if phase != "train":
        print(f"unknown phase: {phase}", file=sys.stderr)
        sys.exit(1)
    # Missing config.yaml is fine: fall back to the `describe` defaults. That is
    # what lets this run locally with nothing prepared.
    cfg_path = root / "config.yaml"
    raw_cfg = (yaml.safe_load(cfg_path.read_text()) or {}) if cfg_path.exists() else {}
    cfg = _flatten_config(raw_cfg, phase)
    sys.exit(0 if _run_train(root, cfg) else 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--utrain-root", type=pathlib.Path, default=pathlib.Path.cwd() / DEFAULT_ROOT
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("describe")
    sub.add_parser("check-compat")
    run_p = sub.add_parser("run")
    run_p.add_argument("--phase", required=True)

    args = parser.parse_args()
    if args.cmd == "describe":
        cmd_describe()
    elif args.cmd == "check-compat":
        cmd_check_compat()
    elif args.cmd == "run":
        root = args.utrain_root
        # utrain mounts these already; creating them is what makes a local run work.
        (root / "data").mkdir(parents=True, exist_ok=True)
        (root / "wandb").mkdir(parents=True, exist_ok=True)
        cmd_run(root, args.phase)


if __name__ == "__main__":
    main()
```

This is the minimum viable version of the contract: one phase, one config
field, `describe`/`check-compat`/`run` but no `serve`. It reads
`<root>/config.yaml`, checks `<root>/control.json` for a stop request,
writes its output to `<root>/data`, and logs a `loss` metric through
`wandb` — utrain will capture that transparently, no wandb account needed.
Phase status comes from the exit code. The *layout* under the root is a
fixed part of the contract, and only `data/` and the metrics dir are
writable; the root itself is whatever `--utrain-root` says, which is
`/utrain` under utrain and `./run` when you run the script yourself. See the
[container reference](container-contract.md) for what each of these does
and why.

## 2. Run it before you build it

Nothing here needs a container yet, so try the phase directly:

```console
$ pip install pyyaml wandb
$ WANDB_MODE=offline python demo_container.py run --phase train
$ cat run/data/model.txt
trained for 50 steps
```

`WANDB_MODE=offline` keeps the real wandb from wanting an account.

## 3. Write the Containerfile

```dockerfile
FROM docker.io/python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir pyyaml wandb
COPY demo_container.py /app/demo_container.py
ENTRYPOINT ["python", "/app/demo_container.py"]
```

## 4. Build and sanity-check it

```console
$ podman build -t localhost/utrain-demo:utrain -f Containerfile .
$ podman run --rm localhost/utrain-demo:utrain describe
```

The second command should print the JSON `describe` blob back to you. If
it does, utrain will be able to parse your container.

## 5. Register it with utrain

```console
$ utrain image add podman://localhost/utrain-demo:utrain
```

`podman://` tells `utrain image add` the image is already local — it just
tags it as a utrain preset rather than pulling.

```console
$ utrain image list
```

should now show `utrain-demo`.

## 6. Create and start a run

```console
$ RID=$(utrain run create --name demo --image utrain-demo --compute cpu --print-id)
$ utrain run start "$RID"
$ utrain run show "$RID" --wait
```

## 7. Look at the metrics with `naw`

```console
$ naw metrics runs/$RID/attempt/1/wandb/train/*.rtsdb
$ naw plot -y loss --lines runs/$RID/attempt/1/wandb/train/*.rtsdb
```

You should see `loss` decreasing across steps.

## Next steps

- Read the [container reference](container-contract.md) for the full
  contract, including `serve` (the OpenAI-compatible protocol behind
  `utrain run chat`) and the exact mount points and podman invocation
  utrain uses.
- Look at `containers/shakespeare-char/` in this repository for a
  real, published container that trains an actual small language model
  end to end. The tutorial above deliberately stays in one file; that one
  is packaged the way a container you maintain should be — a `pyproject.toml`
  and a `src/` layout, with the CLI in its own module and the model, phases,
  and serve loop beside it.
