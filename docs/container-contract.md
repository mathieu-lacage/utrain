# Container reference

A utrain "training container" is an ordinary OCI image, built and run with
`podman`, whose `ENTRYPOINT` is a small CLI. utrain never inspects an
image's internals; it only ever invokes that CLI, so anything satisfying
this contract works with utrain, regardless of what training code or
framework lives inside.

The CLI must implement three subcommands — `describe`, `check-compat`,
`run` — plus `serve` if the image supports live chat testing.
This page documents each one precisely. If you just want to see it work,
follow [Building a container](building-a-container.md) first and come back
here for details.

## `describe`

```console
$ podman run --rm <image> describe
```

Takes no arguments. Prints a single line of JSON to stdout describing the
image. utrain calls this once, right after `utrain image add`, to learn
what the container trains and what it's configurable with.

The JSON must match this shape (see `src/utrain/container/schema.py`):

```yaml
name: str                # container name, e.g. "shakespeare-char"
version: str              # defaults to "1.0.0" if omitted
phases:                   # every phase the container knows about
  - name: str
    label: str             # human-readable, shown in the UI/CLI
phase_order: [str, ...]   # the subset (and order) of `phases` to run
config_schema:
  globals:
    groups:
      - name: str
        label: str
        fields: [FieldSchema, ...]
  phases:
    <phase_name>:
      groups: [FieldGroup, ...]
can_serve: bool            # defaults to false
```

A `FieldSchema` describes one configurable value:

| key | type | notes |
|---|---|---|
| `key` | `str` | the config key, e.g. `learning_rate` |
| `label` | `str` | human-readable |
| `type` | `"int"\|"float"\|"str"\|"bool"\|"enum"` | |
| `default` | matches `type`, or `None` | |
| `description` | `str` | optional |
| `required` | `bool` | defaults to `false` |
| `min` / `max` | `int`/`float`, optional | for numeric types |
| `options` | `list[str]` | for `enum` |

`config_schema.globals` holds fields that apply to every phase (e.g. model
architecture); `config_schema.phases.<phase>` holds fields specific to one
phase (e.g. `batch_size` for a `pretrain` phase). Fields are grouped under
named `groups` purely for display — a group named `model` with fields
`n_layer`, `n_embd` becomes `globals.model.n_layer` / `globals.model.n_embd`
in the generated config (see [Config protocol](#config-protocol) below).

`phase_order` must be non-empty — `utrain run create` refuses to create a
run against an image whose `describe` returns an empty `phase_order`.

## `check-compat`

```console
$ podman run --rm <image> check-compat
```

Takes no arguments. Prints `{"compatible": bool, "details": str}`. Intended
for a container to report whether the current host can run it (e.g. GPU
compute capability, driver version). Not currently invoked by utrain itself,
but part of the contract every reference container implements — a
reasonable place to put your own compatibility checks even if nothing calls
it yet.

## `run`

```console
$ podman run ... <image> run <run_dir> --phase <phase_name>
```

Runs exactly one phase and exits. This is what utrain's orchestrator
invokes once per phase, in `phase_order`. `run_dir` is a positional
argument — utrain passes the path where it mounted the run directory
(always `/utrain` in practice, see [Filesystem contract](#filesystem-contract)),
so don't hardcode a path inside your container; read the argument.

Inside `run`, your container must:

1. Read `<run_dir>/config.yaml` and flatten it for this phase (see
   [Config protocol](#config-protocol)).
2. Periodically read `<run_dir>/control.json` and stop cleanly if
   `{"action": "stop"}` is set (see [Graceful stop](#graceful-stop)).
3. Log progress through `wandb` and emit `_phase_event` markers (see
   [Metrics and phase events](#metrics-and-phase-events)).
4. Exit `0` on success, non-zero on failure or stop.

## `serve`

```console
$ podman run ... <image> serve <run_dir> [--port PORT]
```

Optional — only required if `describe` reports `can_serve: true`. Starts an
HTTP server for quick manual testing of the trained model. Reference
containers default to port 8080 and expose:

```
POST /chat
  body:  {"message": "..."}
  reply: {"reply": "..."}
```

utrain's CLI doesn't currently launch `serve` for you; run it directly with
`podman run`, mounting the same `run_dir` your training phases wrote their
output to.

## Config protocol

Before starting a run, utrain writes `<run_dir>/config.yaml` from your
`config_schema` defaults (`src/utrain/cli/runs.py`, `_write_config`), and
the user may hand-edit it before starting:

```yaml
run_id: <uuid>
output_dir: /run
compute: cpu | gpu<N>
globals:
  <group_name>:
    <field_key>: <value>
phases:
  <phase_name>:
    <field_key>: <value>
```

Your container must flatten this itself for the phase it's running: start
from any top-level scalars, merge in `globals` (expanding each group dict
one level so `globals.model.n_layer` becomes a bare `n_layer` key), then
merge in `phases.<phase>` the same way — phase-specific values win over
globals on key collision. This exact merge is implemented identically in
both reference containers and is safe to copy verbatim:

- `tests/containers/fake/fake.py` — `_flatten_config`
- `containers/shakespeare-char/shakespeare_char.py` — `_flatten_config`

## Graceful stop

`<run_dir>/control.json` holds `{"action": "continue"}` or
`{"action": "stop"}`. utrain writes `"stop"` when the user stops a run.
Your training loop should check this periodically (e.g. once per step or
every few seconds) and, on seeing `"stop"`, log a `<phase>/failed` event,
finish the wandb run, and exit non-zero. Treat a missing or unparseable
file as `"continue"`.

## Metrics and phase events

Your container must `import wandb` and use the standard wandb API:
`wandb.init(...)`, `run.log({...}, step=..., commit=True)`,
`run.finish(exit_code=...)`. At runtime, utrain transparently shadows the
real `wandb` package with its own shim backed by `naw`'s rtsdb format —
metrics never leave the host and no wandb account or network access is
needed. This means your image should still declare and install `wandb` as
a normal dependency (for the API surface / import to resolve at build
time); what actually executes at run time is utrain's shim.

Alongside your real metrics, log a `_phase_event` key at these points so
utrain can track phase status independent of your process's exit code:

| when | value |
|---|---|
| phase begins | `"<phase>/started"` |
| phase finishes successfully | `"<phase>/completed"` |
| phase fails or is stopped | `"<phase>/failed"` |

## Filesystem contract

utrain runs your image roughly as:

```console
$ podman run --rm --network=host --security-opt=label=disable \
    [gpu device args if compute is gpuN] \
    -v <naw>:/opt/utrain-py/naw:ro -v <shim>:/opt/utrain-py/wandb:ro \
    -e PYTHONPATH=/opt/utrain-py \
    -v <attempt_dir>:/utrain \
    -v <phase_data_dir>:/data \
    localhost/<image>:utrain \
    run /utrain --phase <phase>
```

- **`/utrain`** — the run directory, passed to your container as the
  `run_dir` positional argument. Contains `config.yaml`, `control.json`,
  and (once you write to it) `wandb/` and `logs/`.
- **`/data`** — a phase-specific scratch/output directory. Anything you
  write here is available to later phases: utrain hardlink-copies the
  previous phase's `/data` contents forward as the starting point of the
  next phase's `/data`, and deduplicates everything into a content-addressed
  store once the run finishes.
- Exit code `0` means success; anything else means failure.

You don't need to reproduce this invocation yourself — it's shown so you
can build the same command by hand with `podman run` while developing and
debugging a container outside of utrain.

## Building and publishing

utrain has no `build` or `validate` subcommand — a container is just a
`Containerfile` you build with `podman build` (or any OCI-compatible
builder) like any other image. This repository's CI builds every
`containers/<name>/Containerfile` it finds and pushes the image to the
project's registry on tagged releases; adding a new officially-published
preset is as simple as adding a new `containers/<name>/Containerfile`.

To sanity-check a container by hand at any point, run its `describe`
subcommand directly:

```console
$ podman run --rm localhost/utrain-<name>:utrain describe
```

## Reference implementations

- `tests/containers/fake/fake.py` — the smallest complete implementation of
  this contract; good starting point to copy.
- `tests/containers/fake-gpu/fake_gpu.py` — implements only `describe` and
  `run`, showing that `serve`/`check-compat` are only needed if you use
  them.
- `containers/shakespeare-char/` — a real, published container (character-level
  GPT training), showing the full contract used end to end.
