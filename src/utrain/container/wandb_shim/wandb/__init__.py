"""Runtime shim: makes ``import wandb`` resolve to naw's wandb-compatible API.

This package is mounted into training sandboxes ahead of the image's real
``wandb`` on ``PYTHONPATH``, so a container written against upstream wandb
transparently logs to naw's rtsdb backend instead. Every attribute access is
forwarded to :mod:`naw.wandb` via PEP 562 ``__getattr__`` so that mutable
module globals (``run``, ``config``) reflect live state.
"""

import naw.wandb


def __getattr__(name: str) -> object:
    return getattr(naw.wandb, name)
