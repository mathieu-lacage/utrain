"""`utrain.address.parse`, the one parser behind every address-taking command.

The grammar is ``RUN[/N][/PHASE]`` with a run-id prefix for ``RUN``. These
tests pin the disambiguation that makes the grammar writable by hand: a
numeric second component is an attempt, a non-numeric one is a phase of the
run's latest attempt, and `phase_needs_attempt` (what `store check` wants)
turns the latter into a refusal.
"""

import pathlib
import typing

import pytest
import sqlalchemy.orm

import utrain.address
import utrain.db


@pytest.fixture()
def session(tmp_path: pathlib.Path) -> typing.Iterator[sqlalchemy.orm.Session]:
    with utrain.db.with_db(utrain.config.Settings(data_dir=tmp_path)) as s:
        yield s


def _add_run(session: sqlalchemy.orm.Session, run_id: str) -> None:
    import sqlalchemy

    session.execute(
        sqlalchemy.insert(utrain.db.runs).values(
            id=run_id, name="r", image="img", compute="cpu", status="done", created_at=0.0
        )
    )
    session.execute(
        sqlalchemy.insert(utrain.db.run_attempts).values(
            run_id=run_id, attempt=2, status="done", started_at=0.0
        )
    )


def test_bare_run_id_resolves_the_prefix(session: sqlalchemy.orm.Session) -> None:
    run_id = "a" * 32
    _add_run(session, run_id)

    a = utrain.address.parse(run_id[:6], session)
    assert a == utrain.address.Address(run_id=run_id, attempt=None, phase=None)


def test_numeric_second_component_is_an_attempt(session: sqlalchemy.orm.Session) -> None:
    run_id = "b" * 32
    _add_run(session, run_id)

    a = utrain.address.parse(f"{run_id[:4]}/2", session)
    assert a == utrain.address.Address(run_id=run_id, attempt=2, phase=None)


def test_non_numeric_second_component_is_a_phase(session: sqlalchemy.orm.Session) -> None:
    run_id = "c" * 32
    _add_run(session, run_id)

    a = utrain.address.parse(f"{run_id[:4]}/pretrain", session)
    assert a == utrain.address.Address(run_id=run_id, attempt=None, phase="pretrain")


def test_three_components_are_attempt_and_phase(session: sqlalchemy.orm.Session) -> None:
    run_id = "d" * 32
    _add_run(session, run_id)

    a = utrain.address.parse(f"{run_id[:4]}/2/pretrain", session)
    assert a == utrain.address.Address(run_id=run_id, attempt=2, phase="pretrain")


def test_unknown_run_prefix(session: sqlalchemy.orm.Session) -> None:
    with pytest.raises(utrain.exceptions.UI, match="run 'zz' not found"):
        utrain.address.parse("zz", session)


def test_ambiguous_run_prefix(session: sqlalchemy.orm.Session) -> None:
    _add_run(session, "e" * 32)
    _add_run(session, "f" * 32)

    with pytest.raises(utrain.exceptions.UI, match="id prefix '' is ambiguous"):
        utrain.address.parse("", session)


def test_too_many_components(session: sqlalchemy.orm.Session) -> None:
    run_id = "9" * 32
    _add_run(session, run_id)

    with pytest.raises(utrain.exceptions.UI, match="invalid address"):
        utrain.address.parse(f"{run_id}/1/pretrain/extra", session)


def test_non_numeric_middle_component_is_an_error(session: sqlalchemy.orm.Session) -> None:
    run_id = "8" * 32
    _add_run(session, run_id)

    with pytest.raises(utrain.exceptions.UI, match="expected attempt number, got 'pretrain'"):
        utrain.address.parse(f"{run_id}/pretrain/sft", session)


def test_phase_needs_attempt_rejects_run_phase(session: sqlalchemy.orm.Session) -> None:
    run_id = "7" * 32
    _add_run(session, run_id)

    with pytest.raises(utrain.exceptions.UI, match="expected attempt number, got 'pretrain'"):
        utrain.address.parse(f"{run_id[:4]}/pretrain", session, phase_needs_attempt=True)


def test_phase_needs_attempt_still_takes_attempts_and_phases(
    session: sqlalchemy.orm.Session,
) -> None:
    run_id = "6" * 32
    _add_run(session, run_id)

    a = utrain.address.parse(f"{run_id[:4]}/2/pretrain", session, phase_needs_attempt=True)
    assert a == utrain.address.Address(run_id=run_id, attempt=2, phase="pretrain")
