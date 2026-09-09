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

The TUI draws its curves with braille dots, which pack four points into the
height of a cell where half-blocks manage two. Whether the font in front of you
has those glyphs is the one thing a terminal program cannot ask, so it is
guessed: braille unless the encoding is not Unicode, or the terminal is one with
no font to configure at all -- the Linux virtual console, say. If the guess is
wrong the curves come out as boxes; `b` switches to half-blocks for the session,
and this pins it:
```yaml
# utrain.yaml
tui_charset: block
```
`UTRAIN_TUI_CHARSET=block` does the same, and `braille` forces it back on.

`utrain` groups its commands as `compute`, `image`, `run`, `attempt`, `phase`
and `store`, plus `tui`. Run `utrain --help` for the full list.
