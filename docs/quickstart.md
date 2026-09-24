# Train a tiny Shakespeare model

This walks through training a character-level Shakespeare model end to end.
It uses one of our prebuilt images which is designed to produce
a working model within a couple of minutes on a laptop GPU: `shakespeare-char`.

## 1. Download an image

Tagged releases publish classic models to the project's GitLab container registry:

```console
$ utrain image add docker://registry.gitlab.inria.fr/mlacage/utrain/shakespeare-char:latest
```

## 2. Create a run

Create a run on the first GPU (`gpu0`; use `cpu` if you have no GPU). `--print-id`
prints just the new run id so you can capture it:

```console
$ RID=$(utrain run create --name shake --image utrain-shakespeare-char --compute gpu0 --print-id)
```

`run create` writes a default `runs/$RID/config.yaml` you can edit before starting.

## 3. Shrink the model

Edit `runs/$RID/config.yaml` down to a small useful size:

```yaml
globals:
  model:
    n_layer: 1
    n_head: 2
    n_embd: 64
    block_size: 128
phases:
  pretrain:
    max_iters: 2000
    batch_size: 32
    learning_rate: 0.001
    eval_interval: 100
```

That is a ~0.1M-parameter model — it trains in a couple of minutes on an Ada
mobile GPU while still showing a clearly decreasing loss. You can go smaller: the
schema minimums are `n_layer 1`, `n_head 1`, `n_embd 32`, `block_size 64`.

## 4. Start and monitor

```console
$ utrain run start "$RID"
$ utrain run show "$RID" --wait      # blocks until the run finishes
```

At any time, `utrain phase show "$RID/pretrain"` prints the latest `loss`, `bpb`
and `mfu`, plus the tail of the training log.

## 5. Look at the loss with `naw`

Each phase logs its metrics as a `naw` time-series file under the run directory.
Point `naw` at the pretrain metrics to plot the loss right in your terminal:

```console
$ naw metrics runs/$RID/attempt/1/wandb/pretrain/*/*.rtsdb              # list metrics
$ naw plot -y loss --lines runs/$RID/attempt/1/wandb/pretrain/*/*.rtsdb # display a plot in-terminal
$ naw watch runs/$RID/attempt/1/wandb/pretrain/*/*.rtsdb                # live tail while training
```

`naw plot ...` also supports `--output png`/`--output svg`/`--output csv` if you
want to save the curve instead of drawing it in the terminal.

## 6. Use the model

`utrain run chat` starts an interactive session against what the run produced.
Type a prompt and the model's continuation streams back a character at a time:

```console
$ utrain run chat "$RID"
serving shake (utrain-shakespeare-char, phase 'pretrain')
endpoint: http://127.0.0.1:42317/v1  (OpenAI-compatible)
model: context_window=128, id=shakespeare-char, owned_by=utrain
commands:
  /reset   forget the conversation so far
  /quit    end the session (or Ctrl-D)
  /help    this message
anything else is sent to the model as the next turn.
> ROMEO:
O, she doth teach the torches to burn bright...
```

Each turn continues the previous one, so the conversation builds up; `/reset`
starts over. `Ctrl-C` interrupts a reply that has run on too long without ending
the session, and `--max-tokens`/`--temperature` set how long and how adventurous
the replies are.

## 7. Or skip utrain entirely

That `endpoint:` line is not decoration. The container serves the OpenAI
`/v1/chat/completions` API on a port the kernel picks, so while the session
above is running, anything that speaks to OpenAI speaks to your model:

```console
$ curl -s http://127.0.0.1:42317/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"ROMEO:"}]}'
```

```python
client = openai.OpenAI(base_url="http://127.0.0.1:42317/v1", api_key="not-needed")
client.chat.completions.create(
    model="shakespeare-char", messages=[{"role": "user", "content": "ROMEO:"}]
)
```

The container serves this with FastAPI, so it also describes itself. Point a
browser at `http://127.0.0.1:42317/docs` while a session is running for a
browsable schema of exactly what the endpoint accepts, or fetch
`/openapi.json` for the machine-readable version.

You can also start the server without utrain in the loop at all — run the image
directly with `serve --port 0` and read the port it prints (an image with more
than one servable phase wants a `--phase <name>` too). See the
[`serve` contract](container-contract.md#serve) for that, and for what to
implement if you are building your own container.

Only images whose `describe` marks a phase `can_serve: true` can be chatted
with, and only once the run has finished.

A run with more than one training phase produces more than one model, and each
is reachable by name — every phase's data dir is a snapshot of the run as it
stood when that phase ended:

```console
$ utrain run chat "$RID/pretrain"    # the base model
$ utrain run chat "$RID/sft"         # after fine-tuning
```

Without a phase in the address you get the newest servable phase that
completed, which for a one-model run like this one is the only one there is.
In the TUI, `t` from the phases pane talks to the phase under the cursor.

## 8. Use the TUI

`utrain tui` is the whole of utrain in one screen: the runs you have, the phases
each is made of, the config it was started with, its curves as they fill, and
its log. This is a recording of one run from nothing to a conversation with the
model it produced.

<div id="tour" data-cast="../assets/tour.cast" data-poster="npt:0:45"></div>

### What goes past

**`i` and `c`** are the other two things utrain knows about: the images you have
pulled, and the CPU and GPUs it can put a run on. Each opens as a panel over
whatever you were looking at -- the run stays where it was underneath it -- and
`escape` closes it again.

**`n`** creates a run. A run needs three things — a name, an image and something
to run on — and everything else about it is config.

**`2`** is the config utrain wrote from the image's schema. `down` moves the
cursor, `e` edits the field it is on, and `enter` writes it straight to
`runs/<id>/config.yaml`. Here the model is taken from nanochat's default depth
of 8 down to 6, which is the first thing to do to a run that has one GPU rather
than eight.

**`s`** starts it, and `enter` on the run opens the phases it is made of —
download, tokenizer, pretrain, sft, rl — which go from pending to running to
done in order.

**`3`** is the plots. Each phase declares which curves are worth drawing, and
they follow the run live. `m` opens the metric picker: `space` draws a metric or
stops drawing it, `y` solos one, `l` switches to a log y axis and `b` swaps the
braille dots for half-blocks, for a font that has no braille. `E` writes
the curve out as csv, png, svg or pdf.

**`4`** is the phase's log, tailed as it is written.

**`t`** talks to what the run produced. The phase under the cursor is the one it
serves and the reply streams back from an OpenAI-compatible
endpoint the container brings up for as long as the screen is open. `escape`
ends the session and stops the container.

**`?`** lists every key, and `q` quits.
