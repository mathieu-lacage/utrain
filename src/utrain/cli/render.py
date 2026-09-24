"""Turns query-layer results into the CLI's text output.

Everything here is CLI-specific: table widths, timestamp formats, and the
truncation of run ids to a unique prefix. The query layer returns raw values so
that another client can make different choices.
"""

from .. import types
from . import output

_RUN_HEADERS = ["ID", "NAME", "IMAGE", "COMPUTE", "STATUS", "ATTEMPT", "PHASE", "CREATED"]
_PHASE_HEADERS = ["PHASE", "ORDER", "STATUS", "STARTED", "ENDED"]


def min_prefix_len(run_ids: list[str]) -> int:
    """Shortest prefix that still tells these run ids apart."""
    if not run_ids:
        return 1
    for prefix_len in range(1, 33):
        prefixes = {rid[:prefix_len] for rid in run_ids}
        if len(prefixes) == len(run_ids):
            return prefix_len
    return 32


def _run_cells(run: types.RunRow, prefix_len: int) -> list[str]:
    return [
        run.id[:prefix_len],
        run.name,
        run.image,
        run.compute,
        run.status,
        str(run.attempt) if run.attempt is not None else "--",
        run.phase if run.phase is not None else "--",
        output.format_time(run.created_at),
    ]


def _phase_cells(phase: types.PhaseRow) -> list[str]:
    return [
        # The fully-qualified address, so what the table shows can be pasted
        # straight into `phase show` or `phase restart`.
        phase.address,
        str(phase.phase_order),
        phase.status,
        output.format_time(phase.started_at),
        output.format_time(phase.ended_at),
    ]


def run_table(runs: list[types.RunRow]) -> str:
    prefix_len = min_prefix_len([r.id for r in runs])
    return output.format_table(_RUN_HEADERS, [_run_cells(r, prefix_len) for r in runs])


def run_row(run: types.RunRow, all_run_ids: list[str]) -> str:
    """One run in table form, with the id truncated as the full listing would."""
    prefix_len = min_prefix_len(all_run_ids)
    return output.format_table(_RUN_HEADERS, [_run_cells(run, prefix_len)])


def gc_result(result: types.GcResult) -> str:
    return (
        f"removed {result.removed} file(s), {output.human_size(result.reclaimed_bytes)} reclaimed"
    )


def store_check(result: types.StoreCheckResult) -> str:
    lines = [f"{p.path}: {p.problem}" for p in result.problems]
    if result.orphaned:
        lines.append(
            f"note: {result.orphaned} orphaned store file(s) ('utrain store gc' reclaims them)"
        )
    # The verbose listing: each checked data file beside the store file it
    # shares an inode with.
    lines += [f"{link.data}  {link.store}" for link in result.links]
    summary = (
        f"checked {result.data_files} data file(s) in {result.attempts} attempt(s) "
        f"against {result.store_files} store file(s)"
    )
    if result.problems:
        summary += f": {len(result.problems)} problem(s)"
    else:
        summary += ": ok"
    lines.append(summary)
    return "\n".join(lines)


def phase_table(phases: list[types.PhaseRow]) -> str:
    return output.format_table(_PHASE_HEADERS, [_phase_cells(p) for p in phases])


def run_detail(detail: types.RunDetail) -> str:
    run = detail.run
    attempts_str = (
        f"{detail.n_attempts} (latest: {detail.latest_attempt_status})"
        if detail.n_attempts
        else "0"
    )

    lines = [
        f"id:       {run.id}",
        f"name:     {run.name}",
        # The frozen image id, so what a run is pinned to is visible; a run
        # that predates the freeze shows the name only.
        f"image:    {run.image}" + (f" ({run.image_id[:12]})" if run.image_id is not None else ""),
        f"compute:  {run.compute}",
        f"status:   {run.status}",
        f"attempts: {attempts_str}",
        f"created:  {output.format_time(run.created_at)}",
    ]

    logs_dir = detail.logs_dir
    if logs_dir is not None:
        lines += [
            "",
            phase_table(detail.phases),
            "",
            f"config: {detail.config_path}",
            f"logs:   {logs_dir}",
        ]

    return "\n".join(lines)


def attempt_table(attempts: list[types.AttemptRow]) -> str:
    headers = ["ATTEMPT", "FROM_PHASE", "STATUS", "STARTED", "ENDED"]
    rows = [
        [
            a.address,
            a.from_phase if a.from_phase else "--",
            a.status,
            output.format_time(a.started_at),
            output.format_time(a.ended_at),
        ]
        for a in attempts
    ]
    return output.format_table(headers, rows)


def attempt_detail(detail: types.AttemptDetail) -> str:
    a = detail.attempt
    lines = [
        f"run:        {detail.run_id} ({detail.run_name})",
        f"attempt:    {a.attempt}",
        f"from_phase: {a.from_phase or '--'}",
        f"status:     {a.status}",
        f"started:    {output.format_time(a.started_at)}",
        f"ended:      {output.format_time(a.ended_at)}",
    ]
    if detail.phases:
        lines += ["", phase_table(detail.phases)]
    lines += ["", f"logs: {detail.logs_dir}", f"data: {detail.data_dir}"]
    return "\n".join(lines)


def phase_list_table(entries: list[types.PhaseListEntry]) -> str:
    rows = [
        [
            e.address,
            str(e.phase_order),
            # A `--from-phase` restart does not re-run the phases before it,
            # and those keep the status they finished with in the attempt that
            # did -- which is the attempt their address names.
            e.status if e.status is not None else "--",
            output.format_time(e.started_at),
            output.format_time(e.ended_at),
        ]
        for e in entries
    ]
    return output.format_table(_PHASE_HEADERS, rows)


def metric_series(metrics: list[types.Metric]) -> str:
    return output.format_table(["STEP", "VALUE"], [[str(m.step), str(m.value)] for m in metrics])


def phase_detail(detail: types.PhaseDetail, log_tail: list[str], tail_n: int) -> str:
    lines = [
        f"phase:   {detail.phase_label}",
        f"attempt: {detail.attempt}",
        f"status:  {detail.status}",
        f"run:     {detail.run_id} ({detail.run_name})",
        f"started: {output.format_time(detail.started_at)}",
        f"ended:   {output.format_time(detail.ended_at)}",
    ]

    if detail.last_metrics:
        rows = [[m.name, str(round(m.value, 3)), str(m.step)] for m in detail.last_metrics]
        lines += ["", output.format_table(["METRIC", "LAST", "STEP"], rows)]

    if detail.log_file is not None:
        lines += ["", f"--- logs/{detail.log_file.name} (tail {tail_n}) ---", *log_tail]

    return "\n".join(lines)
