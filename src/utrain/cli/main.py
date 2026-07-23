import argparse
import sys

from .. import config
from . import attempts, compute, images, output, phases, runs, store
from . import db as dbmod


def _cmd_compute_list(args: argparse.Namespace) -> None:
    info = compute.collect_compute()
    cpu = info.cpu
    mem_used = cpu.mem_total_gb - cpu.mem_available_gb

    headers = ["KIND", "NAME", "CORES", "POWER", "COMPUTE", "MEM_USED", "MEM_TOTAL", "MEM_PCT"]
    rows: list[list[str]] = [
        [
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
                "gpu",
                gpu.name,
                "--",
                f"{gpu.power_draw:.1f}/{gpu.power_limit:.1f} W",
                f"{gpu.util}%",
                f"{mem_used_gb:.1f}",
                f"{mem_total_gb:.1f}",
                f"{mem_pct:.1f}%",
            ]
        )
    print(output.format_table(headers, rows))


def _cmd_image_list(args: argparse.Namespace) -> None:
    settings = config.Settings()
    with dbmod.with_db(settings) as session:
        imgs = images.list_images(session)
    headers = ["NAME", "SIZE", "RUNS"]
    rows = [[i.name, i.size_str, str(i.run_count)] for i in imgs]
    print(output.format_table(headers, rows))


def _cmd_image_add(args: argparse.Namespace) -> None:
    name = images.add_image(args.url)
    if args.print_id:
        print(name, end="")
    else:
        settings = config.Settings()
        with dbmod.with_db(settings) as session:
            imgs = images.list_images(session)
        matching = [i for i in imgs if i.name == name]
        if matching:
            headers = ["NAME", "SIZE", "RUNS"]
            print(
                output.format_table(
                    headers, [[matching[0].name, matching[0].size_str, str(matching[0].run_count)]]
                )
            )


def _cmd_image_remove(args: argparse.Namespace) -> None:
    settings = config.Settings()
    with dbmod.with_db(settings) as session:
        images.remove_image(args.name, session, force=args.force)


def _cmd_run_list(args: argparse.Namespace) -> None:
    settings = config.Settings()
    with dbmod.with_db(settings) as session:
        if args.json:
            runs.list_runs_json(session)
        else:
            runs.list_runs(session, short=getattr(args, "short", False))


def _cmd_run_show(args: argparse.Namespace) -> None:
    settings = config.Settings()
    with dbmod.with_db(settings) as session:
        runs.show_run(
            args.id,
            session,
            wait=args.wait,
            timeout=args.timeout,
            edit=args.edit,
        )


def _cmd_run_create(args: argparse.Namespace) -> None:
    settings = config.Settings()
    with dbmod.with_db(settings) as session:
        runs.create_run(
            name=args.name,
            image=args.image,
            gpu_spec=args.gpu,
            settings=settings,
            session=session,
            print_id=args.print_id,
        )


def _cmd_run_start(args: argparse.Namespace) -> None:
    settings = config.Settings()
    with dbmod.with_db(settings) as session:
        runs.start_run(args.id, session)


def _cmd_run_stop(args: argparse.Namespace) -> None:
    settings = config.Settings()
    with dbmod.with_db(settings) as session:
        runs.stop_run(args.id, session)


def _cmd_run_restart(args: argparse.Namespace) -> None:
    settings = config.Settings()
    with dbmod.with_db(settings) as session:
        runs.restart_run(args.id, getattr(args, "from_phase", None), session)


def _cmd_run_delete(args: argparse.Namespace) -> None:
    settings = config.Settings()
    with dbmod.with_db(settings) as session:
        runs.delete_run(args.id, force=args.force, session=session)


def _cmd_run_logs(args: argparse.Namespace) -> None:
    settings = config.Settings()
    with dbmod.with_db(settings) as session:
        runs.logs_run(
            args.id,
            session,
            attempt=getattr(args, "attempt", None),
            phase=getattr(args, "phase", None),
            stderr=getattr(args, "stderr", False),
            tail=getattr(args, "tail", 200),
        )


def _cmd_attempt_list(args: argparse.Namespace) -> None:
    settings = config.Settings()
    with dbmod.with_db(settings) as session:
        attempts.list_attempts(args.run_id, session)


def _cmd_attempt_show(args: argparse.Namespace) -> None:
    settings = config.Settings()
    parts = args.addr.split("/")
    if len(parts) != 2 or not parts[1].isdigit():
        print(f"abort: expected <RUN_ID>/<N>, got '{args.addr}'", file=sys.stderr)
        sys.exit(2)
    with dbmod.with_db(settings) as session:
        attempts.show_attempt(parts[0], int(parts[1]), session)


def _cmd_phase_list(args: argparse.Namespace) -> None:
    settings = config.Settings()
    with dbmod.with_db(settings) as session:
        phases.list_phases(args.addr, session)


def _cmd_phase_read(args: argparse.Namespace) -> None:
    settings = config.Settings()
    with dbmod.with_db(settings) as session:
        phases.read_phase(
            args.addr,
            session,
            metric=getattr(args, "metric", None),
            since_step=getattr(args, "since_step", 0),
        )


def _cmd_store_gc(args: argparse.Namespace) -> None:
    settings = config.Settings()
    store.gc(settings)


def _cmd_orchestrate(args: argparse.Namespace) -> None:
    from . import orchestrator

    settings = config.Settings()
    orchestrator.run_orchestrator(
        run_id=args.run_id,
        attempt=int(args.attempt),
        from_phase=getattr(args, "from_phase", None),
        settings=settings,
    )


def _cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from .. import main as web_main

    settings = config.Settings()
    app = web_main.create_app(settings)
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="utrain")
    sub = parser.add_subparsers(dest="command")

    # serve
    sub.add_parser("serve", help="Start the web server")

    # compute
    compute_p = sub.add_parser("compute")
    compute_sub = compute_p.add_subparsers(dest="compute_command")
    compute_sub.add_parser("list")

    # image
    image_p = sub.add_parser("image")
    image_sub = image_p.add_subparsers(dest="image_command")
    image_sub.add_parser("list")
    img_add = image_sub.add_parser("add")
    img_add.add_argument("url")
    img_add.add_argument("--print-id", action="store_true", dest="print_id")
    img_rm = image_sub.add_parser("remove")
    img_rm.add_argument("name")
    img_rm.add_argument("--force", action="store_true")

    # run
    run_p = sub.add_parser("run")
    run_sub = run_p.add_subparsers(dest="run_command")

    run_list = run_sub.add_parser("list")
    run_list.add_argument("--json", action="store_true")
    run_list.add_argument("--short", action="store_true")

    run_show = run_sub.add_parser("show")
    run_show.add_argument("id")
    run_show.add_argument("--wait", action="store_true")
    run_show.add_argument("--timeout", type=int, default=600)
    run_show.add_argument("--edit", action="store_true")
    run_show.add_argument("--json", action="store_true")

    run_create = run_sub.add_parser("create")
    run_create.add_argument("--name", required=True)
    run_create.add_argument("--image", required=True)
    run_create.add_argument("--gpu", default="auto")
    run_create.add_argument("--print-id", action="store_true", dest="print_id")

    run_start = run_sub.add_parser("start")
    run_start.add_argument("id")

    run_stop = run_sub.add_parser("stop")
    run_stop.add_argument("id")

    run_restart = run_sub.add_parser("restart")
    run_restart.add_argument("id")
    run_restart.add_argument("--from-phase", dest="from_phase", default=None)

    run_delete = run_sub.add_parser("delete")
    run_delete.add_argument("id")
    run_delete.add_argument("--force", action="store_true")

    run_logs = run_sub.add_parser("logs")
    run_logs.add_argument("id")
    run_logs.add_argument("--attempt", type=int, default=None)
    run_logs.add_argument("--phase", default=None)
    run_logs.add_argument("--stderr", action="store_true")
    run_logs.add_argument("--tail", type=int, default=200)

    # attempt
    attempt_p = sub.add_parser("attempt")
    attempt_sub = attempt_p.add_subparsers(dest="attempt_command")
    att_list = attempt_sub.add_parser("list")
    att_list.add_argument("run_id")
    att_show = attempt_sub.add_parser("show")
    att_show.add_argument("addr")

    # phase
    phase_p = sub.add_parser("phase")
    phase_sub = phase_p.add_subparsers(dest="phase_command")
    ph_list = phase_sub.add_parser("list")
    ph_list.add_argument("addr")
    ph_read = phase_sub.add_parser("read")
    ph_read.add_argument("addr")
    ph_read.add_argument("--metric", default=None)
    ph_read.add_argument("--since-step", type=int, default=0, dest="since_step")

    # store
    store_p = sub.add_parser("store")
    store_sub = store_p.add_subparsers(dest="store_command")
    store_sub.add_parser("gc")

    # hidden _orchestrate subcommand
    orch = sub.add_parser("_orchestrate")
    orch.add_argument("run_id")
    orch.add_argument("attempt", type=int)
    orch.add_argument("--from-phase", dest="from_phase", default=None)

    sub.metavar = "{serve,compute,image,run,attempt,phase,store}"

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "serve":
            _cmd_serve(args)
        elif args.command == "compute":
            if args.compute_command == "list":
                _cmd_compute_list(args)
            else:
                parser.print_help()
                sys.exit(2)
        elif args.command == "image":
            if args.image_command == "list":
                _cmd_image_list(args)
            elif args.image_command == "add":
                _cmd_image_add(args)
            elif args.image_command == "remove":
                _cmd_image_remove(args)
            else:
                parser.print_help()
                sys.exit(2)
        elif args.command == "run":
            if args.run_command == "list":
                _cmd_run_list(args)
            elif args.run_command == "show":
                _cmd_run_show(args)
            elif args.run_command == "create":
                _cmd_run_create(args)
            elif args.run_command == "start":
                _cmd_run_start(args)
            elif args.run_command == "stop":
                _cmd_run_stop(args)
            elif args.run_command == "restart":
                _cmd_run_restart(args)
            elif args.run_command == "delete":
                _cmd_run_delete(args)
            elif args.run_command == "logs":
                _cmd_run_logs(args)
            else:
                parser.print_help()
                sys.exit(2)
        elif args.command == "attempt":
            if args.attempt_command == "list":
                _cmd_attempt_list(args)
            elif args.attempt_command == "show":
                _cmd_attempt_show(args)
            else:
                parser.print_help()
                sys.exit(2)
        elif args.command == "phase":
            if args.phase_command == "list":
                _cmd_phase_list(args)
            elif args.phase_command == "read":
                _cmd_phase_read(args)
            else:
                parser.print_help()
                sys.exit(2)
        elif args.command == "store":
            if args.store_command == "gc":
                _cmd_store_gc(args)
            else:
                parser.print_help()
                sys.exit(2)
        elif args.command == "_orchestrate":
            _cmd_orchestrate(args)
        else:
            parser.print_help()
            sys.exit(2)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr, end="")
            sys.exit(1)
        raise


if __name__ == "__main__":
    main()
