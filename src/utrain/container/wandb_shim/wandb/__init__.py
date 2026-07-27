"""Runtime shim: makes ``import wandb`` resolve to baw's wandb-compatible API.

This package is mounted into training sandboxes ahead of the image's real
``wandb`` on ``PYTHONPATH``, so a container written against upstream wandb
transparently logs to baw's rtsdb backend instead. Every attribute access is
forwarded to :mod:`baw.wandb` via PEP 562 ``__getattr__`` so that mutable
module globals (``run``, ``config``) reflect live state.
"""

import baw.wandb


def __getattr__(name: str) -> object:
    return getattr(baw.wandb, name)
