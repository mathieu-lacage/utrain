# Get started

utrain trains small language models on your machine with your GPU.

Training a model end to end normally means gluing together a pile of moving
parts: a container with the right CUDA and PyTorch versions, a training script,
a config file, somewhere to put checkpoints, and some way to tell whether the
loss is going down. utrain manages all this so you can concentrate on the
model.

A *run* is the unit you work with. You pick a training image and the compute to
use, utrain generates a config file to edit, and from then on the run is
something you can start, monitor, stop, restart and inspect:

```console
$ utrain image add docker://gitlab.inria.fr:5050/mlacage/utrain/shakespeare-char:latest
$ RID=$(utrain run create --name tiny --image utrain-shakespeare-char --compute gpu0 --print-id)
$ utrain run start "$RID"
$ utrain run show "$RID"
```

Each run is made of ordered *phases* — tokenizing the dataset, pretraining, and
whatever else the image declares. Phases run one after another, each one logging
its own metrics, so a failed run can be restarted from the phase that broke
instead of from the beginning.

## What you get

- **Images.** Training environments are ordinary OCI images pulled with podman;
  the project publishes a few ready to use. Nothing is installed into your
  Python environment or onto your host.
- **Configuration.** Every run gets a `config.yaml` under `runs/<id>/` with the
  image's defaults filled in, validated against the image's schema before the
  run starts rather than crashing ten minutes in.
- **Monitoring.** `utrain run show` and `utrain phase show` report status,
  progress and the latest metrics; `--wait` blocks until a run finishes, which
  makes runs easy to drive from a script.
- **Metrics.** Phases log time series that [`naw`](https://pypi.org/project/naw/)
  can plot, tail live, or export to CSV, PNG or SVG.

## Scope

utrain is deliberately single-host: it runs containers on your CPU or your local
GPUs, keeps its state in one SQLite database and one runs directory, and has no
scheduler, no cluster and no account to create. It is meant for the models you
can train in minutes to hours on hardware you already have — teaching,
experimenting, and getting a pipeline working before it goes anywhere bigger.

Ready to try it? [Install utrain](installation.md), then walk through the
[quickstart](quickstart.md) to train a character-level Shakespeare model.
