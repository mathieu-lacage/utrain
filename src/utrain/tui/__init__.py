"""Textual front-end for utrain: a second client of the query layer.

Importing this imports `textual` and `uniplot`, which are optional (the `tui`
extra). `cli/main` guards the import and turns a missing one into a UI error.
"""

from . import app, data, render, screens, widgets

__all__ = ["app", "data", "render", "screens", "widgets"]
