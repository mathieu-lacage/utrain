# Install

First, make sure you install [podman](https://podman.io/docs/installation) and
[container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
either from packages or from source.

Then install utrain. We recommend the use of pipx:
```console
$ pipx install utrain naw
```

`utrain` groups its commands as `compute`, `image`, `run`, `attempt`, `phase`
and `store`. Run `utrain --help` for the full list.
