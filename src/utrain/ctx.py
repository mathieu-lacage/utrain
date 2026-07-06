import contextlib
import contextvars
import typing

import sqlalchemy.orm

from . import config

T = typing.TypeVar("T")


class _Var(typing.Generic[T]):
    def __init__(self, name: str) -> None:
        self._var: contextvars.ContextVar[T] = contextvars.ContextVar(name)

    def get(self) -> T:
        return self._var.get()

    @contextlib.contextmanager
    def __call__(self, value: T) -> typing.Iterator[None]:
        token = self._var.set(value)
        try:
            yield
        finally:
            self._var.reset(token)


settings: _Var[config.Settings] = _Var("settings")
db: _Var[sqlalchemy.orm.Session] = _Var("db")
