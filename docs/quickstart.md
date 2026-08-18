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
$ RID=$(utrain run create --name tiny-shakespeare \
    --image utrain-shakespeare-char --compute gpu0 --print-id)
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
$ naw metrics runs/$RID/attempt/1/wandb/pretrain/*.rtsdb              # list metrics
$ naw plot -y loss --lines runs/$RID/attempt/1/wandb/pretrain/*.rtsdb # display a plot in-terminal
$ naw watch runs/$RID/attempt/1/wandb/pretrain/*.rtsdb                # live tail while training
```

`naw plot ...` also supports `--output png`/`--output svg`/`--output csv` if you
want to save the curve instead of drawing it in the terminal.
