"""The CLI's fully-qualified ids: ``RUN[/N][/PHASE]``.

Every command that takes an address takes it in this one grammar, parsed by
`parse`, so what a table's first column prints is always exactly what another
command accepts: ``utrain run logs 7/2/pretrain`` is the phase row
``7/2/pretrain`` spelled where it is used. ``RUN`` is a run-id prefix, a
numeric second component is an attempt, and a non-numeric one names a phase of
the run's latest attempt.
"""

import typing

import sqlalchemy.orm

from . import db as dbmod
from . import exceptions


class Address(typing.NamedTuple):
    """A parsed address: the run it names, plus optional attempt and phase."""

    run_id: str
    attempt: int | None
    phase: str | None


def parse(
    addr: str,
    session: sqlalchemy.orm.Session,
    *,
    phase_needs_attempt: bool = False,
) -> Address:
    """Parse ``RUN[/N][/PHASE]`` into an Address.

    ``RUN`` is resolved through the usual id-prefix rules; a second component
    that is all digits is an attempt, one that is not is a phase name (the
    run's latest attempt is implied). A third component is a phase and its
    attempt both. With `phase_needs_attempt` -- what `store check` wants, since
    the check has to know which attempt's data dir to look in -- ``RUN/PHASE``
    is rejected rather than guessed at.
    """
    parts = addr.split("/")
    if len(parts) > 3:
        raise exceptions.UI(f"invalid address '{addr}'")

    run_id = dbmod.resolve_run_id(parts[0], session)

    attempt: int | None = None
    phase: str | None = None
    if len(parts) == 2:
        token = parts[1]
        if token.isdigit():
            attempt = int(token)
        elif phase_needs_attempt:
            raise exceptions.UI(f"expected attempt number, got '{token}'")
        else:
            phase = token
    elif len(parts) == 3:
        if not parts[1].isdigit():
            raise exceptions.UI(f"expected attempt number, got '{parts[1]}'")
        attempt = int(parts[1])
        phase = parts[2]

    return Address(run_id, attempt, phase)
