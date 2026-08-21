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


def _read_control(path: pathlib.Path) -> str:
    try:
        return json.loads(path.read_text()).get("action", "continue")
    except Exception:
        return "continue"


def cmd_describe():
    print(json.dumps(DESCRIBE))


def cmd_check_compat():
    print(json.dumps({"compatible": True, "details": "always compatible"}))


def _run_train(run_dir: pathlib.Path, cfg: dict) -> bool:
    run = wandb.init(project="demo", id=str(cfg.get("run_id", "")), dir=str(run_dir))
    run.log({"_phase_event": "train/started"})
    steps = int(cfg.get("steps", 50))
    ok = True
    for step in range(steps):
        if _read_control(run_dir / "control.json") == "stop":
            run.log({"_phase_event": "train/failed"})
            ok = False
            break
        loss = 3.0 * (0.95 ** step) + random.gauss(0, 0.02)
        run.log({"loss": loss}, step=step, commit=True)
        time.sleep(0.05)
    if ok:
        run.log({"_phase_event": "train/completed"})
    run.finish(exit_code=0 if ok else 1)
    return ok


def cmd_run(run_dir: pathlib.Path, phase: str):
    raw_cfg = yaml.safe_load((run_dir / "config.yaml").read_text()) or {}
    cfg = _flatten_config(raw_cfg, phase)
    if phase != "train":
        print(f"unknown phase: {phase}", file=sys.stderr)
        sys.exit(1)
    sys.exit(0 if _run_train(run_dir, cfg) else 1)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("describe")
    sub.add_parser("check-compat")
    run_p = sub.add_parser("run")
    run_p.add_argument("run_dir", type=pathlib.Path)
    run_p.add_argument("--phase", required=True)

    args = parser.parse_args()
    if args.cmd == "describe":
        cmd_describe()
    elif args.cmd == "check-compat":
        cmd_check_compat()
    elif args.cmd == "run":
        cmd_run(args.run_dir, args.phase)


if __name__ == "__main__":
    main()
```

This is the minimum viable version of the contract: one phase, one config
field, `describe`/`check-compat`/`run` but no `serve`. It reads
`config.yaml`, checks `control.json` for a stop request, and logs a
`loss` metric plus `_phase_event` markers through `wandb` — utrain will
capture those transparently, no wandb account needed. See the
[container reference](container-contract.md) for what each of these does
and why.

## 2. Write the Containerfile

```dockerfile
FROM docker.io/python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir pyyaml wandb
COPY demo_container.py /app/demo_container.py
ENTRYPOINT ["python", "/app/demo_container.py"]
```

## 3. Build and sanity-check it

```console
$ podman build -t localhost/utrain-demo:utrain -f Containerfile .
$ podman run --rm localhost/utrain-demo:utrain describe
```

The second command should print the JSON `describe` blob back to you. If
it does, utrain will be able to parse your container.

## 4. Register it with utrain

```console
$ utrain image add podman://localhost/utrain-demo:utrain
```

`podman://` tells `utrain image add` the image is already local — it just
tags it as a utrain preset rather than pulling.

```console
$ utrain image list
```

should now show `utrain-demo`.

## 5. Create and start a run

```console
$ RID=$(utrain run create --name demo --image utrain-demo --compute cpu --print-id)
$ utrain run start "$RID"
$ utrain run show "$RID" --wait
```

## 6. Look at the metrics with `naw`

```console
$ naw metrics runs/$RID/attempt/1/wandb/train/*.rtsdb
$ naw plot -y loss --lines runs/$RID/attempt/1/wandb/train/*.rtsdb
```

You should see `loss` decreasing across steps.

## Next steps

- Read the [container reference](container-contract.md) for the full
  contract, including `serve` (live chat testing) and the exact mount
  points and podman invocation utrain uses.
- Look at `containers/shakespeare-char/` in this repository for a
  real, published container that trains an actual small language model
  end to end.
