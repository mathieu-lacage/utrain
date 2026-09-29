import argparse
import datetime
import importlib.metadata
import json
import os
import pathlib
import signal
import subprocess
import sys
import typing

import sqlalchemy.orm

from .. import (
    address,
    attempts,
    compute,
    config,
    dispatcher,
    exceptions,
    images,
    orchestrator,
    phases,
    runs,
    store,
    sweeps,
)
from .. import db as dbmod
from . import chat, debug, output, render


def db_command(
    f: typing.Callable[[sqlalchemy.orm.Session, argparse.Namespace], None],
) -> typing.Callable[[argparse.Namespace], None]:
    def inner(args: argparse.Namespace) -> None:
        settings = config.Settings()
        with dbmod.with_db(settings) as session:
            return f(session, args)

    return inner


# How much of a phase's stdout log `phase show` echoes.
_PHASE_LOG_TAIL = 20

# How long `run start`, `run restart` and `run stop` wait for the dispatcher to
# act on them before printing the run as it stands. Long enough for a stopped
# container to exit, which a restart of a running run waits for as well.
_DISPATCH_WAIT_SECONDS = 30.0


def _run_id(arg: str) -> str:
    """A positional that has to be a bare run id, not a longer address.

    `run show` and friends have nothing to say to an attempt or a phase, so a
    slash in the positional is answered with a clear refusal rather than the
    silent mis-resolution of taking everything before the slash as the prefix.
    """
    if "/" in arg:
        raise exceptions.UI(f"expected a run id, got '{arg}' (this command takes RUN only)")
    return arg


def _print_run_row(run_id: str, session: sqlalchemy.orm.Session) -> None:
    run = runs.get_run(run_id, session)
    print(render.run_row(run, runs.list_run_ids(session)))


def _cmd_compute_list(args: argparse.Namespace) -> None:
    info = compute.collect_compute()
    cpu = info.cpu
    mem_used = cpu.mem_total_gb - cpu.mem_available_gb

    headers = [
        "ID",
        "KIND",
        "NAME",
        "CORES",
        "POWER (W)",
        "COMPUTE",
        "MEM_USED (GB)",
        "MEM_TOTAL (GB)",
        "MEM_USED (%)",
    ]
    rows: list[list[str]] = [
        [
            "cpu",
            "cpu",
            cpu.name,
            str(cpu.cores),
            "--",
            "--",
            f"{mem_used:.1f}",
            f"{cpu.mem_total_gb:.1f}",
            f"{100 * mem_used / cpu.mem_total_gb:.1f}%" if cpu.mem_total_gb else "--",
        ]
    ]
    for gpu in info.gpus:
        mem_used_gb = gpu.mem_used_mb / 1024
        mem_total_gb = gpu.mem_total_mb / 1024
        mem_pct = 100 * mem_used_gb / mem_total_gb if mem_total_gb else 0
        rows.append(
            [
                f"gpu{gpu.index}",
                "gpu",
                gpu.name,
                "--",
                f"{gpu.power_draw:.1f}/{gpu.power_limit:.1f}",
                f"{gpu.util}%",
                f"{mem_used_gb:.1f}",
                f"{mem_total_gb:.1f}",
                f"{mem_pct:.1f}%",
            ]
        )
    print(output.format_table(headers, rows))


@db_command
def _cmd_image_list(session: sqlalchemy.orm.Session, args: argparse.Namespace) -> None:
    imgs = images.list_images(session)
    if args.quiet:
        for i in imgs:
            print(i.name)
        return
    headers = ["NAME", "SIZE", "RUNS"]
    rows = [[i.name, i.size_str, str(i.run_count)] for i in imgs]
    print(output.format_table(headers, rows))


@db_command
def _cmd_image_add(session: sqlalchemy.orm.Session, args: argparse.Namespace) -> None:
    name = images.add_image(args.url)
    if args.print_id:
        print(name, end="")
    else:
        imgs = images.list_images(session)
        matching = [i for i in imgs if i.name == name]
        if matching:
            headers = ["NAME", "SIZE", "RUNS"]
            print(
                output.format_table(
                    headers, [[matching[0].name, matching[0].size_str, str(matching[0].run_count)]]
                )
            )


@db_command
def _cmd_image_remove(session: sqlalchemy.orm.Session, args: argparse.Namespace) -> None:
    for name in args.names:
        images.remove_image(name, session, force=args.force)


@db_command
def _cmd_run_list(session: sqlalchemy.orm.Session, args: argparse.Namespace) -> None:
    if args.quiet:
        for run_id in runs.list_run_ids(session):
            print(run_id)
        return

    rows = runs.list_runs(session)
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "id": r.id,
                        "name": r.name,
                        "image": r.image,
                        "image_id": r.image_id,
                        "compute": r.compute,
                        "status": r.status,
                        "created_at": datetime.datetime.fromtimestamp(r.created_at).isoformat(),
                    }
                    for r in rows
                ]
            )
        )
    else:
        print(render.run_table(rows))


@db_command
def _cmd_run_show(session: sqlalchemy.orm.Session, args: argparse.Namespace) -> None:
    if args.edit:
        config_path = runs.config_path(_run_id(args.id), session)
        editor = os.environ.get("EDITOR", "vi")
        with runs.writable(config_path):
            subprocess.run([editor, str(config_path)])
        return

    detail = runs.get_run_detail(_run_id(args.id), session, wait=args.wait, timeout=args.timeout)
    print(render.run_detail(detail))


@db_command
def _cmd_run_create(session: sqlalchemy.orm.Session, args: argparse.Namespace) -> None:
    settings = config.Settings()
    run_id = runs.create_run(
        name=args.name,
        image=args.image,
        compute_spec=args.compute,
        settings=settings,
        session=session,
    )
    if args.print_id:
        print(run_id)
    else:
        _print_run_row(run_id, session)


@db_command
def _cmd_run_start(session: sqlalchemy.orm.Session, args: argparse.Namespace) -> None:
    run_id = runs.start_run(_run_id(args.id), session)
    runs.wait_until_started(run_id, session, _DISPATCH_WAIT_SECONDS)
    _print_run_row(run_id, session)


@db_command
def _cmd_run_stop(session: sqlalchemy.orm.Session, args: argparse.Namespace) -> None:
    run_id = runs.stop_run(_run_id(args.id), session)
    runs.wait_until_ended(run_id, session, _DISPATCH_WAIT_SECONDS)
    _print_run_row(run_id, session)


@db_command
def _cmd_run_restart(session: sqlalchemy.orm.Session, args: argparse.Namespace) -> None:
    a = address.parse(args.addr, session)
    if a.attempt is not None and a.phase is None:
        raise exceptions.UI(f"restart takes RUN or RUN/PHASE, got '{args.addr}'")
    run_id = runs.restart_run(a.run_id, a.phase, session)
    runs.wait_until_started(run_id, session, _DISPATCH_WAIT_SECONDS)
    _print_run_row(run_id, session)


def _cmd_run_delete(args: argparse.Namespace) -> None:
    for run_id_prefix in args.ids:
        settings = config.Settings()
        with dbmod.with_db(settings) as session:
            run_id = runs.delete_run(_run_id(run_id_prefix), force=args.force, session=session)
            print(f"removed run {run_id}")


@db_command
def _cmd_run_logs(session: sqlalchemy.orm.Session, args: argparse.Namespace) -> None:
    a = address.parse(args.addr, session)
    # The whole log: training output is what this command exists for, and a
    # viewer who wants less pipes through `tail`. `tail_lines` seeks backwards
    # in blocks, so an unread prefix costs nothing.
    lines = runs.read_log_tail(
        a.run_id,
        session,
        attempt=a.attempt,
        phase=a.phase,
        tail=0,
    )
    for line in lines:
        print(line)


@db_command
def _cmd_run_chat(session: sqlalchemy.orm.Session, args: argparse.Namespace) -> None:
    a = address.parse(args.addr, session)
    chat.chat_run(
        a.run_id,
        session,
        phase=a.phase,
        attempt=a.attempt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )


@db_command
def _cmd_attempt_list(session: sqlalchemy.orm.Session, args: argparse.Namespace) -> None:
    if args.quiet:
        for addr in attempts.list_attempt_ids(args.addr, session):
            print(addr)
    else:
        print(render.attempt_table(attempts.list_attempts(args.addr, session)))


@db_command
def _cmd_attempt_show(session: sqlalchemy.orm.Session, args: argparse.Namespace) -> None:
    a = address.parse(args.addr, session)
    if a.attempt is None:
        print(f"abort: expected <RUN_ID>/<N>, got '{args.addr}'", file=sys.stderr)
        sys.exit(2)
    print(render.attempt_detail(attempts.show_attempt(a.run_id, a.attempt, session)))


@db_command
def _cmd_phase_list(session: sqlalchemy.orm.Session, args: argparse.Namespace) -> None:
    if args.quiet:
        for addr in phases.list_phase_ids(args.addr, session):
            print(addr)
    else:
        print(render.phase_list_table(phases.list_phases(args.addr, session)))


@db_command
def _cmd_phase_show(session: sqlalchemy.orm.Session, args: argparse.Namespace) -> None:
    metric = getattr(args, "metric", None)
    if metric is not None:
        series = phases.read_metric(
            args.addr, session, metric, since_step=getattr(args, "since_step", 0)
        )
        print(render.metric_series(series))
        return

    detail = phases.show_phase(args.addr, session)
    print(
        render.phase_detail(detail, phases.read_log_tail(detail, _PHASE_LOG_TAIL), _PHASE_LOG_TAIL)
    )


@db_command
def _cmd_phase_restart(session: sqlalchemy.orm.Session, args: argparse.Namespace) -> None:
    run_id = phases.restart_phase(args.addr, session)
    runs.wait_until_started(run_id, session, _DISPATCH_WAIT_SECONDS)
    _print_run_row(run_id, session)


def _axes_args(values: list[str] | None) -> dict[str, object]:
    axes: dict[str, object] = {}
    for arg in values or []:
        path, raw = sweeps.parse_axis_arg(arg)
        if path in axes:
            raise exceptions.UI(f"axis '{path}' given twice")
        axes[path] = raw
    return axes


def _compute_arg(value: str | None) -> list[str]:
    return [c.strip() for c in value.split(",") if c.strip()] if value else []


@db_command
def _cmd_sweep_create(session: sqlalchemy.orm.Session, args: argparse.Namespace) -> None:
    # A spec file gives the whole sweep; flags fill in or override its fields,
    # so a file can be reused with a different name or compute.
    spec = (
        sweeps.read_spec_file(pathlib.Path(args.spec))
        if args.spec
        else sweeps.SpecFile(None, None, None, {}, [], [])
    )
    axes = dict(spec.axes)
    axes.update(_axes_args(args.axis))
    name = args.name or spec.name
    if name is None:
        raise exceptions.UI("a sweep needs a name (--name, or 'name:' in the spec file)")
    sweep_id = sweeps.create_sweep(
        name=name,
        axes=axes,
        compute=_compute_arg(args.compute) or spec.compute,
        settings=config.Settings(),
        session=session,
        image=args.image or spec.image,
        base=args.base or spec.base,
        replicate=(args.replicate or []) + spec.replicate,
    )
    if args.start:
        sweeps.start_sweep(sweep_id, session)
    if args.print_id:
        print(sweep_id)
    else:
        print(render.sweep_detail(sweeps.get_sweep(sweep_id, session)))


@db_command
def _cmd_sweep_list(session: sqlalchemy.orm.Session, args: argparse.Namespace) -> None:
    if args.quiet:
        for sweep_id in sweeps.list_sweep_ids(session):
            print(sweep_id)
        return
    print(render.sweep_table(sweeps.list_sweeps(session)))


@db_command
def _cmd_sweep_show(session: sqlalchemy.orm.Session, args: argparse.Namespace) -> None:
    print(render.sweep_detail(sweeps.get_sweep(args.sweep, session)))


def _print_sweep_row(sweep_id: str, session: sqlalchemy.orm.Session) -> None:
    rows = [s for s in sweeps.list_sweeps(session) if s.id == sweep_id]
    print(render.sweep_table(rows))


@db_command
def _cmd_sweep_start(session: sqlalchemy.orm.Session, args: argparse.Namespace) -> None:
    _print_sweep_row(sweeps.start_sweep(args.sweep, session), session)


@db_command
def _cmd_sweep_pause(session: sqlalchemy.orm.Session, args: argparse.Namespace) -> None:
    _print_sweep_row(sweeps.pause_sweep(args.sweep, session), session)


@db_command
def _cmd_sweep_cancel(session: sqlalchemy.orm.Session, args: argparse.Namespace) -> None:
    _print_sweep_row(sweeps.cancel_sweep(args.sweep, session), session)


@db_command
def _cmd_sweep_retry(session: sqlalchemy.orm.Session, args: argparse.Namespace) -> None:
    sweep_id, n = sweeps.retry_sweep(args.sweep, session)
    print(f"requeued {n} run(s)")
    _print_sweep_row(sweep_id, session)


@db_command
def _cmd_sweep_extend(session: sqlalchemy.orm.Session, args: argparse.Namespace) -> None:
    sweep_id, n = sweeps.extend_sweep(args.sweep, _axes_args(args.axis), config.Settings(), session)
    print(f"added {n} run(s)")
    _print_sweep_row(sweep_id, session)


@db_command
def _cmd_sweep_delete(session: sqlalchemy.orm.Session, args: argparse.Namespace) -> None:
    for ref in args.sweeps:
        sweep_id, n = sweeps.delete_sweep(ref, force=args.force, session=session)
        print(f"removed sweep {sweep_id} and its {n} run(s)")


def _cmd_dispatch(args: argparse.Namespace) -> None:
    dispatcher.run(config.Settings())


def _cmd_store_gc(args: argparse.Namespace) -> None:
    settings = config.Settings()
    print(render.gc_result(store.gc(settings)))


@db_command
def _cmd_store_check(session: sqlalchemy.orm.Session, args: argparse.Namespace) -> None:
    settings = config.Settings()
    result = store.check(
        settings,
        session,
        hash_contents=not args.no_hash,
        verbose=args.verbose,
        scope=args.addr,
    )
    print(render.store_check(result))
    if result.problems:
        sys.exit(1)


def _cmd_tui(args: argparse.Namespace) -> None:
    # Imported here, not at module scope, so that every other command keeps
    # working when the optional `tui` extra is not installed.
    try:
        from .. import tui
    except ImportError as e:
        raise exceptions.UI(
            "the tui needs textual and uniplot. Run: pip install 'utrain[tui]'"
        ) from e

    tui.app.run()


def _cmd_orchestrate(args: argparse.Namespace) -> None:
    settings = config.Settings()
    orchestrator.run_orchestrator(
        run_id=args.run_id,
        attempt=int(args.attempt),
        from_phase=getattr(args, "from_phase", None),
        settings=settings,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="utrain")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {importlib.metadata.version('utrain')}",
    )
    parser.add_argument("-d", "--debug", action="count", default=0)
    parser.add_argument("--log-filename", help="Filename where logs will be written", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    # compute
    compute_p = sub.add_parser("compute", help="CPU/GPU on host")
    compute_sub = compute_p.add_subparsers(dest="compute_command", required=True)
    compute_sub.add_parser("list", help="List compute resources on host").set_defaults(
        func=_cmd_compute_list
    )

    # image
    image_p = sub.add_parser("image", help="Container images")
    image_sub = image_p.add_subparsers(dest="image_command", required=True)
    img_list = image_sub.add_parser("list", help="List images from local store")
    img_list.add_argument("-q", "--quiet", action="store_true")
    img_list.set_defaults(func=_cmd_image_list)
    img_add = image_sub.add_parser("add", help="Add image to local store")
    img_add.add_argument("url")
    img_add.add_argument("--print-id", action="store_true", dest="print_id")
    img_add.set_defaults(func=_cmd_image_add)
    img_rm = image_sub.add_parser("remove", help="Remove one or more images from local store")
    img_rm.add_argument("names", nargs="+", metavar="NAME")
    img_rm.add_argument("--force", action="store_true")
    img_rm.set_defaults(func=_cmd_image_remove)

    # run
    run_p = sub.add_parser("run", help="Experiment runs")
    run_sub = run_p.add_subparsers(dest="run_command", required=True)

    run_list = run_sub.add_parser("list", help="List runs")
    run_list.add_argument("--json", action="store_true")
    run_list.add_argument("-q", "--quiet", action="store_true")
    run_list.set_defaults(func=_cmd_run_list)

    run_show = run_sub.add_parser("show", help="Show details about a run")
    run_show.add_argument("id", metavar="RUN", help="a run id, or a unique prefix of one")
    run_show.add_argument("--wait", action="store_true")
    run_show.add_argument("--timeout", type=int, default=600)
    run_show.add_argument("--edit", action="store_true")
    run_show.set_defaults(func=_cmd_run_show)

    run_create = run_sub.add_parser("create", help="Create a run")
    run_create.add_argument("--name", required=True)
    run_create.add_argument("--image", required=True)
    run_create.add_argument("--compute", required=True)
    run_create.add_argument("--print-id", action="store_true", dest="print_id")
    run_create.set_defaults(func=_cmd_run_create)

    run_start = run_sub.add_parser("start", help="Start a run")
    run_start.add_argument("id", metavar="RUN", help="a run id, or a unique prefix of one")
    run_start.set_defaults(func=_cmd_run_start)

    run_stop = run_sub.add_parser("stop", help="Stop a running run")
    run_stop.add_argument("id", metavar="RUN", help="a run id, or a unique prefix of one")
    run_stop.set_defaults(func=_cmd_run_stop)

    run_restart = run_sub.add_parser(
        "restart", help="Restart a run from a phase (create a new attempt)"
    )
    run_restart.add_argument(
        "addr", metavar="ADDR", help="RUN for a full restart, or RUN/PHASE to restart from there"
    )
    run_restart.set_defaults(func=_cmd_run_restart)

    run_delete = run_sub.add_parser("delete", help="Delete one or more runs")
    run_delete.add_argument("ids", nargs="+", metavar="ID")
    run_delete.add_argument("--force", action="store_true")
    run_delete.set_defaults(func=_cmd_run_delete)

    run_logs = run_sub.add_parser(
        "logs",
        help="Show logs for a run (the whole log; pipe through tail(1) to limit it)",
    )
    run_logs.add_argument(
        "addr",
        metavar="ADDR",
        help="RUN (the orchestrator log), RUN/ATTEMPT, RUN/PHASE or RUN/ATTEMPT/PHASE",
    )
    run_logs.set_defaults(func=_cmd_run_logs)

    run_chat = run_sub.add_parser("chat", help="Chat interactively with a run's trained model")
    run_chat.add_argument(
        "addr",
        metavar="ADDR",
        help="RUN, RUN/PHASE, RUN/ATTEMPT or RUN/ATTEMPT/PHASE (a phase names whose"
        " snapshot to talk to; default: the newest servable one)",
    )
    run_chat.add_argument("--max-tokens", type=int, default=200, dest="max_tokens")
    run_chat.add_argument("--temperature", type=float, default=0.8)
    run_chat.set_defaults(func=_cmd_run_chat)

    # attempt
    attempt_p = sub.add_parser("attempt")
    attempt_sub = attempt_p.add_subparsers(dest="attempt_command", required=True)
    att_list = attempt_sub.add_parser("list")
    att_list.add_argument("addr", metavar="ADDR", help="RUN, or RUN/ATTEMPT for one attempt")
    att_list.add_argument("-q", "--quiet", action="store_true")
    att_list.set_defaults(func=_cmd_attempt_list)
    att_show = attempt_sub.add_parser("show")
    att_show.add_argument("addr", metavar="ADDR", help="RUN/ATTEMPT")
    att_show.set_defaults(func=_cmd_attempt_show)

    # phase
    phase_p = sub.add_parser("phase", help="Individual phases of a run")
    phase_sub = phase_p.add_subparsers(dest="phase_command", required=True)
    ph_list = phase_sub.add_parser("list", help="List all phases of a run")
    ph_list.add_argument("addr", metavar="ADDR", help="RUN, or RUN/ATTEMPT for one attempt")
    ph_list.add_argument("-q", "--quiet", action="store_true")
    ph_list.set_defaults(func=_cmd_phase_list)
    ph_show = phase_sub.add_parser("show", help="Show a single phase")
    ph_show.add_argument("addr", metavar="ADDR", help="RUN/PHASE or RUN/ATTEMPT/PHASE")
    ph_show.add_argument("--metric", default=None)
    ph_show.add_argument("--since-step", type=int, default=0, dest="since_step")
    ph_show.set_defaults(func=_cmd_phase_show)
    ph_restart = phase_sub.add_parser(
        "restart", help="Restart a run from one of its phases (create a new attempt)"
    )
    ph_restart.add_argument("addr", metavar="ADDR", help="RUN/PHASE or RUN/ATTEMPT/PHASE")
    ph_restart.set_defaults(func=_cmd_phase_restart)

    # sweep
    sweep_p = sub.add_parser("sweep", help="Grid searches: one run per combination of values")
    sweep_sub = sweep_p.add_subparsers(dest="sweep_command", required=True)
    sweep_ref_help = "a sweep's name, or a unique prefix of its id"

    sw_create = sweep_sub.add_parser(
        "create", help="Create a sweep and one queued run per point of its grid"
    )
    sw_create.add_argument(
        "spec", nargs="?", default=None, metavar="SPEC", help="a sweep spec file (YAML)"
    )
    sw_create.add_argument("--name")
    sw_create.add_argument("--image", help="the image to sweep (or use --from)")
    sw_create.add_argument(
        "--from", dest="base", metavar="RUN", help="copy the image and config from this run"
    )
    sw_create.add_argument(
        "--axis",
        action="append",
        metavar="PATH=VALUES",
        help="e.g. phases.pretrain.learning_rate=log:1e-4:1e-2:5 or globals.model.n_layer=4,6,8;"
        " '*' sweeps every value of a bool or enum field. Repeatable.",
    )
    sw_create.add_argument(
        "--compute", help="comma-separated computes; runs are spread across them"
    )
    sw_create.add_argument(
        "--replicate",
        action="append",
        metavar="PATH",
        help="an axis whose runs are averaged, not told apart, when compared (e.g. a seed)",
    )
    sw_create.add_argument("--start", action="store_true", help="start dispatching right away")
    sw_create.add_argument("--print-id", action="store_true", dest="print_id")
    sw_create.set_defaults(func=_cmd_sweep_create)

    sw_list = sweep_sub.add_parser("list", help="List sweeps")
    sw_list.add_argument("-q", "--quiet", action="store_true")
    sw_list.set_defaults(func=_cmd_sweep_list)

    sw_show = sweep_sub.add_parser("show", help="Show a sweep and its runs")
    sw_show.add_argument("sweep", metavar="SWEEP", help=sweep_ref_help)
    sw_show.set_defaults(func=_cmd_sweep_show)

    for name, func, text in (
        ("start", _cmd_sweep_start, "Start dispatching a sweep's queued runs"),
        ("resume", _cmd_sweep_start, "Resume a paused sweep"),
        ("pause", _cmd_sweep_pause, "Stop starting new runs; running ones carry on"),
        ("cancel", _cmd_sweep_cancel, "Stop a sweep's running runs and drop its queued ones"),
        ("retry", _cmd_sweep_retry, "Queue a sweep's failed and stopped runs again"),
    ):
        p = sweep_sub.add_parser(name, help=text)
        p.add_argument("sweep", metavar="SWEEP", help=sweep_ref_help)
        p.set_defaults(func=func)

    sw_extend = sweep_sub.add_parser("extend", help="Add values to a sweep's axes")
    sw_extend.add_argument("sweep", metavar="SWEEP", help=sweep_ref_help)
    sw_extend.add_argument("--axis", action="append", required=True, metavar="PATH=VALUES")
    sw_extend.set_defaults(func=_cmd_sweep_extend)

    sw_delete = sweep_sub.add_parser("delete", help="Delete sweeps and all of their runs")
    sw_delete.add_argument("sweeps", nargs="+", metavar="SWEEP")
    sw_delete.add_argument("--force", action="store_true")
    sw_delete.set_defaults(func=_cmd_sweep_delete)

    # store
    store_p = sub.add_parser("store", help="Data store")
    store_sub = store_p.add_subparsers(dest="store_command", required=True)
    store_gc = store_sub.add_parser("gc")
    store_gc.set_defaults(func=_cmd_store_gc)
    store_check = store_sub.add_parser(
        "check", help="Verify run data is hardlinked into the store by content hash"
    )
    store_check.add_argument(
        "addr",
        nargs="?",
        default=None,
        metavar="ADDR",
        help="Scope: RUN_ID, RUN_ID/ATTEMPT or RUN_ID/ATTEMPT/PHASE (default: every run)",
    )
    store_check.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="List each checked file beside the store file it is hardlinked to",
    )
    store_check.add_argument(
        "--no-hash",
        action="store_true",
        help="Skip re-hashing store files; check only that the hardlinks are in place",
    )
    store_check.set_defaults(func=_cmd_store_check)

    sub.add_parser("tui", help="Browse runs, phases and metrics interactively").set_defaults(
        func=_cmd_tui
    )

    # hidden _orchestrate subcommand
    orch = sub.add_parser("_orchestrate")
    orch.add_argument("run_id")
    orch.add_argument("attempt", type=int)
    orch.add_argument("--from-phase", dest="from_phase", default=None)
    orch.set_defaults(func=_cmd_orchestrate)

    # hidden _dispatch subcommand: the dispatcher process
    dispatch = sub.add_parser("_dispatch")
    dispatch.set_defaults(func=_cmd_dispatch)

    sub.metavar = "{compute,image,run,attempt,phase,sweep,store,tui}"

    return parser


def main() -> None:
    # Die on SIGPIPE the way every other Unix tool does. Python installs SIG_IGN
    # for it at startup, which turns `utrain run show | head -7` into a
    # BrokenPipeError traceback once head exits: the writes that follow raise
    # instead of killing us. Output buffering hides this -- buffered, the whole
    # output flushes in one write while head is still reading -- so it only
    # surfaces under PYTHONUNBUFFERED or once output outgrows the buffer.
    #
    # SIG_IGN is also inherited across exec, so this hands the default
    # disposition to the podman children the orchestrator spawns rather than
    # passing them Python's.
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    parser = build_parser()
    args = parser.parse_args()

    debug.setup(args.debug, args.log_filename)

    try:
        args.func(args)
    except exceptions.UI as e:
        # The query layer raises plain messages; the "abort:" framing is this
        # CLI's, so that another front end can present the same failure its way.
        print(f"abort: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
