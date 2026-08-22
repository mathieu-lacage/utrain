import argparse
import datetime
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import typing

import sqlalchemy.orm

from .. import attempts, compute, config, exceptions, images, orchestrator, phases, runs, store
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
        config_path = runs.config_path(args.id, session)
        os.chmod(config_path, 0o644)
        editor = os.environ.get("EDITOR", "vi")
        subprocess.run([editor, str(config_path)])
        os.chmod(config_path, 0o444)
        return

    detail = runs.get_run_detail(args.id, session, wait=args.wait, timeout=args.timeout)
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
    _print_run_row(runs.start_run(args.id, session), session)


@db_command
def _cmd_run_stop(session: sqlalchemy.orm.Session, args: argparse.Namespace) -> None:
    _print_run_row(runs.stop_run(args.id, session), session)


@db_command
def _cmd_run_restart(session: sqlalchemy.orm.Session, args: argparse.Namespace) -> None:
    _print_run_row(runs.restart_run(args.id, getattr(args, "from_phase", None), session), session)


def _cmd_run_delete(args: argparse.Namespace) -> None:
    for run_id_prefix in args.ids:
        settings = config.Settings()
        with dbmod.with_db(settings) as session:
            run_id = runs.delete_run(run_id_prefix, force=args.force, session=session)
            print(f"removed run {run_id}")


@db_command
def _cmd_run_logs(session: sqlalchemy.orm.Session, args: argparse.Namespace) -> None:
    lines = runs.read_log_tail(
        args.id,
        session,
        attempt=getattr(args, "attempt", None),
        phase=getattr(args, "phase", None),
        stderr=getattr(args, "stderr", False),
        tail=getattr(args, "tail", 200),
    )
    for line in lines:
        print(line)


@db_command
def _cmd_run_chat(session: sqlalchemy.orm.Session, args: argparse.Namespace) -> None:
    chat.chat_run(
        args.id,
        session,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )


@db_command
def _cmd_attempt_list(session: sqlalchemy.orm.Session, args: argparse.Namespace) -> None:
    if args.quiet:
        for addr in attempts.list_attempt_ids(args.run_id, session):
            print(addr)
    else:
        print(render.attempt_table(attempts.list_attempts(args.run_id, session)))


@db_command
def _cmd_attempt_show(session: sqlalchemy.orm.Session, args: argparse.Namespace) -> None:
    parts = args.addr.split("/")
    if len(parts) != 2 or not parts[1].isdigit():
        print(f"abort: expected <RUN_ID>/<N>, got '{args.addr}'", file=sys.stderr)
        sys.exit(2)
    print(render.attempt_detail(attempts.show_attempt(parts[0], int(parts[1]), session)))


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


def _cmd_store_gc(args: argparse.Namespace) -> None:
    settings = config.Settings()
    print(render.gc_result(store.gc(settings)))


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
    compute_sub = compute_p.add_subparsers(dest="compute_command")
    compute_sub.add_parser("list", help="List compute resources on host").set_defaults(
        func=_cmd_compute_list
    )

    # image
    image_p = sub.add_parser("image", help="Container images")
    image_sub = image_p.add_subparsers(dest="image_command")
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
    run_sub = run_p.add_subparsers(dest="run_command")

    run_list = run_sub.add_parser("list", help="List runs")
    run_list.add_argument("--json", action="store_true")
    run_list.add_argument("-q", "--quiet", action="store_true")
    run_list.set_defaults(func=_cmd_run_list)

    run_show = run_sub.add_parser("show", help="Show details about a run")
    run_show.add_argument("id")
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
    run_start.add_argument("id")
    run_start.set_defaults(func=_cmd_run_start)

    run_stop = run_sub.add_parser("stop", help="Stop a running run")
    run_stop.add_argument("id")
    run_stop.set_defaults(func=_cmd_run_stop)

    run_restart = run_sub.add_parser(
        "restart", help="Restart a run from a phase (create a new attempt)"
    )
    run_restart.add_argument("id")
    run_restart.add_argument("--from-phase", dest="from_phase", default=None)
    run_restart.set_defaults(func=_cmd_run_restart)

    run_delete = run_sub.add_parser("delete", help="Delete one or more runs")
    run_delete.add_argument("ids", nargs="+", metavar="ID")
    run_delete.add_argument("--force", action="store_true")
    run_delete.set_defaults(func=_cmd_run_delete)

    run_logs = run_sub.add_parser("logs", help="Show logs for a run")
    run_logs.add_argument("id")
    run_logs.add_argument("--attempt", type=int, default=None)
    run_logs.add_argument("--phase", default=None)
    run_logs.add_argument("--stderr", action="store_true")
    run_logs.add_argument("--tail", type=int, default=200)
    run_logs.set_defaults(func=_cmd_run_logs)

    run_chat = run_sub.add_parser("chat", help="Chat interactively with a run's trained model")
    run_chat.add_argument("id")
    run_chat.add_argument("--max-tokens", type=int, default=200, dest="max_tokens")
    run_chat.add_argument("--temperature", type=float, default=0.8)
    run_chat.set_defaults(func=_cmd_run_chat)

    # attempt
    attempt_p = sub.add_parser("attempt")
    attempt_sub = attempt_p.add_subparsers(dest="attempt_command")
    att_list = attempt_sub.add_parser("list")
    att_list.add_argument("run_id")
    att_list.add_argument("-q", "--quiet", action="store_true")
    att_list.set_defaults(func=_cmd_attempt_list)
    att_show = attempt_sub.add_parser("show")
    att_show.add_argument("addr")
    att_show.set_defaults(func=_cmd_attempt_show)

    # phase
    phase_p = sub.add_parser("phase", help="Individual phases of a run")
    phase_sub = phase_p.add_subparsers(dest="phase_command")
    ph_list = phase_sub.add_parser("list", help="List all phases of a run")
    ph_list.add_argument("addr", help="Run id")
    ph_list.add_argument("-q", "--quiet", action="store_true")
    ph_list.set_defaults(func=_cmd_phase_list)
    ph_show = phase_sub.add_parser("show", help="Show a single phase")
    ph_show.add_argument("addr")
    ph_show.add_argument("--metric", default=None)
    ph_show.add_argument("--since-step", type=int, default=0, dest="since_step")
    ph_show.set_defaults(func=_cmd_phase_show)

    # store
    store_p = sub.add_parser("store", help="Data store")
    store_sub = store_p.add_subparsers(dest="store_command")
    store_gc = store_sub.add_parser("gc")
    store_gc.set_defaults(func=_cmd_store_gc)

    # hidden _orchestrate subcommand
    orch = sub.add_parser("_orchestrate")
    orch.add_argument("run_id")
    orch.add_argument("attempt", type=int)
    orch.add_argument("--from-phase", dest="from_phase", default=None)
    orch.set_defaults(func=_cmd_orchestrate)

    sub.metavar = "{compute,image,run,attempt,phase,store}"

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
