# Install

First, make sure you install [podman](https://podman.io/docs/installation) and
[container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
either from packages or from source.

Then install utrain. We recommend the use of pipx:
```console
$ pipx install utrain naw[plot]
```

To also get `utrain tui`, the terminal UI for adding images, configuring,
starting and stopping runs, plotting their metrics, exporting a plot as a csv,
png, svg or pdf and chatting with what a run trained, ask for the `tui` extra:
```console
$ pipx install "utrain[tui]" naw[plot]
```

`utrain` groups its commands as `compute`, `image`, `run`, `attempt`, `phase`
and `store`, plus `tui`. Run `utrain --help` for the full list.
