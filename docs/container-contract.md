# Container reference

A utrain "training container" is an ordinary OCI image, built and run with
`podman`, whose `ENTRYPOINT` is a small CLI. utrain never inspects an
image's internals; it only ever invokes that CLI, so anything satisfying
this contract works with utrain, regardless of what training code or
framework lives inside.

The CLI must implement three subcommands — `describe`, `check-compat`,
`run` — plus `serve` if the image supports live chat testing, and
`check-cache` if any phase declares itself `cacheable`.
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
    cacheable: bool        # defaults to false; see `check-cache` below
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

## `check-cache`

```console
$ podman run ... <image> --utrain-root <root> check-cache --phase <phase_name>
```

Only required for phases whose `describe` entry sets `cacheable: true`.
Takes the same arguments and filesystem mounts as `run` (see
[Filesystem contract](#filesystem-contract)), with one difference:
`<root>/data` is mounted **read-only**, so the no-write rule below is
enforced rather than merely requested. It is already populated with
whatever the previous phase carried forward.

Prints a single line of JSON to stdout describing the exact set of files
this phase *would* produce if run right now, given the current config and
`<root>/data` contents — without doing the phase's real (expensive) work:

```yaml
files:
  - path: str      # relative to <root>/data
    sha256: str     # content hash of the file this phase would write there
```

utrain uses this to skip actually running the phase: if every declared
`sha256` is already present in its content-addressed store, it populates
`<root>/data` from the store directly and never launches your container
for that phase. This only makes sense for phases whose output is a deterministic
function of their config and input data (e.g. a fixed-parameter
preprocessing step) — don't mark a phase `cacheable` if its output can vary
run to run (e.g. anything driven by an unseeded random process). utrain
trusts the declared hashes; it never verifies them against what `run` would
actually produce, so an incorrect `check-cache` answer silently serves stale
or wrong data from the store.

`check-cache` must not write to `<root>/data` and should return quickly (no
GPU work, no heavy compute) — it runs on every attempt of a cacheable phase, not
just cache hits. Exit non-zero (or time out) if you can't predict the
manifest for some reason; utrain treats that as a cache miss and runs the
phase normally.

## `run`

```console
$ podman run ... <image> --utrain-root <root> run --phase <phase_name>
```

Runs exactly one phase and exits. This is what utrain's orchestrator
invokes once per phase, in `phase_order`. Every path your container needs
lives at a fixed location under `<root>`, so `--utrain-root` and `--phase`
are the only inputs it receives on the command line — see
[Filesystem contract](#filesystem-contract) for the layout. The fundamental
one is `<root>/data`: this is where a phase reads what earlier phases
produced and writes its own output for later phases to build on.

Inside `run`, your container must:

1. Read `<root>/config.yaml` and flatten it for this phase (see
   [Config protocol](#config-protocol)).
2. Read whatever earlier phases left in `<root>/data` and write this
   phase's output back to `<root>/data` — this is the sole channel phases
   use to hand off state to each other.
3. Periodically read `<root>/control.json` and stop cleanly if
   `{"action": "stop"}` is set (see [Graceful stop](#graceful-stop)).
4. Log progress through `wandb` and emit `_phase_event` markers (see
   [Metrics and phase events](#metrics-and-phase-events)).
5. Exit `0` on success, non-zero on failure or stop.

## `serve`

```console
$ podman run ... <image> --utrain-root <root> serve [--port PORT]
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
`podman run`, mounting the same dirs the training phases used — in
particular `<root>/data`, since that is where the trained model lives.

## Config protocol

Before starting a run, utrain writes `<root>/config.yaml` from your
`config_schema` defaults (`src/utrain/cli/runs.py`, `_write_config`), and
the user may hand-edit it before starting:

```yaml
run_id: <uuid>
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
- `containers/shakespeare-char/src/shakespeare_char/config.py` — `flatten`

## Graceful stop

`<root>/control.json` holds `{"action": "continue"}` or
`{"action": "stop"}`. utrain writes `"stop"` when the user stops a run.
Your training loop should check this periodically (e.g. once per step or
every few seconds) and, on seeing `"stop"`, log a `<phase>/failed` event,
finish the wandb run, and exit non-zero. Treat a missing or unparsable
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

Everything utrain gives your container lives under a single root, at fixed
paths beneath it. The *layout* is the contract and there is nothing to discover
about it; the root itself is passed in as `--utrain-root`, so the same code can
run against a mount inside a container and against a plain directory on a
developer's machine.

`--utrain-root` is a global option — it comes before the subcommand — and every
subcommand must accept it, because utrain passes it unconditionally, including
to subcommands that never touch the filesystem. Its default must be `run`,
relative to the working directory, which is what makes the local invocation
below work with no arguments at all. `run` and `serve` should create the
writable dirs (`data/`, `wandb/`) if they are missing, so a local run needs no
setup; `check-cache` must not, since it is forbidden to write.

utrain always passes `/utrain`, and runs your image roughly as:

```console
$ podman run --rm --network=host --security-opt=label=disable \
    [gpu device args if compute is gpuN] \
    -v <naw>:/opt/utrain-py/naw:ro -v <shim>:/opt/utrain-py/wandb:ro \
    -e PYTHONPATH=/opt/utrain-py \
    -v <attempt_dir>/mnt:/utrain:ro \
    -v <phase_data_dir>:/utrain/data \
    -v <attempt_dir>/wandb:/utrain/wandb \
    localhost/<image>:utrain \
    --utrain-root /utrain run --phase <phase>
```

| path | mode | what it is |
|---|---|---|
| `<root>/config.yaml` | ro | this run's config (see [Config protocol](#config-protocol)) |
| `<root>/control.json` | ro | the stop flag (see [Graceful stop](#graceful-stop)) |
| `<root>/data` | rw | the phase's data dir — inputs from earlier phases, and your output |
| `<root>/wandb` | rw | where the wandb shim writes metrics; you never touch it directly |

The root itself is mounted read-only, and `data/` and `wandb/` are the only two
places a phase may write. That is deliberate: it makes `<root>/data` provably
the sole channel phases hand state through. utrain's own bookkeeping for the
run — logs, the orchestrator's state, other phases' data dirs — is not mounted
and is not visible to your container at all.

**`<root>/data`** is a phase-specific scratch/output directory. Anything you
write here is available to later phases: utrain hardlink-copies the previous
phase's contents forward as the starting point of the next phase's data dir,
and deduplicates everything into a content-addressed store once the run
finishes. For a `cacheable` phase, utrain may skip `run` entirely and populate
it straight from that store — see [`check-cache`](#check-cache). The trained
model belongs here too, so that `serve` and later phases can find it.

Exit code `0` means success; anything else means failure.

The top level of the root is reserved for utrain: don't create your own files
or directories beside `config.yaml` and `data/` (under utrain it's read-only,
so you can't), and expect utrain to add entries there in future versions. Keep
everything of your own inside `<root>/data`.

### Running a phase without a container

Because the root is an argument rather than a hardcoded `/utrain`, a phase runs
unmodified straight from a checkout, against `./run`:

```console
$ shakespeare-char run --phase tokenizer
$ ls run/data
input.txt  vocab.json
```

No config.yaml is required — a container should treat a missing one as an empty
config and fall back to its `describe` defaults, and a missing or unparsable
control.json as `"continue"`. Drop a `run/config.yaml` in place to exercise the
[config protocol](#config-protocol) itself.

The one thing that differs from a real run is `wandb`: utrain substitutes its
shim for you inside the container, but on the host `import wandb` finds whatever
you installed. Either put utrain's shim first on `PYTHONPATH`
(`src/utrain/container/wandb_shim`, alongside `naw`) to get the same rtsdb files
utrain would collect, or set `WANDB_MODE=offline` if you only care about the
phase's real work.

You don't need to reproduce the `podman run` invocation above yourself — it's
shown so you can build the same command by hand while debugging a built image,
once the local loop is no longer enough.

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
  GPT training), showing the full contract used end to end. It is a normal
  Python project (`pyproject.toml` plus `src/shakespeare_char/`) whose CLI
  entry point, in `cli.py`, is the only module that knows about the
  subcommands above; the phases, model, and serve handler are plain modules
  next to it.
