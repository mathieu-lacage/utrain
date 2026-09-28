# Sweeps, run comparison, and TUI navigation

Status: proposal, for discussion. Nothing here is implemented.

This note designs two features for `utrain tui`, **sweeps** (grid search:
create, start and monitor a set of runs whose configs cover combinations of
values) and **inter-run comparison** (plots and tables across runs). It then
re-examines the TUI's navigation as a whole. The options run from extending
the current lazygit layout to replacing it: a DOS-style menu bar,
workspaces, a k9s-style command line, Miller columns, tiling, and
VisiData-style sheets.

---

## 0. Summary

* **A sweep is a named, generated group of ordinary runs, plus a
  dispatcher.** Sweep runs are real runs, so every existing screen and CLI
  command (config, plots, logs, restart, chat) works on them unchanged. What
  is new: a `sweeps` table, `sweep_id` and `sweep_point` columns on `runs`,
  a `queued` run status, and a detached dispatcher that starts queued runs
  as compute slots free up.
* **A comparison is a lens over a *run set*.** A sweep is the most common
  source of a run set but not the only one; marked runs and filter results
  are others. This one concept ties the two features together: a sweep is a
  run set with a generator, and "compare" works on any set.
* **The current navigation is full.** The footer is truncated at 120
  columns (§1), about 25 single-letter keys are taken, and the whole screen
  is built around one run at a time. Sweeps and comparison add roughly 15
  commands and two views that cover many runs at once.
* **Two separate decisions** are bundled in the word "navigation":
  1. **Structure:** how views are arranged and what "where am I" means.
  2. **Command surface:** how the user invokes and discovers actions and
     jumps between places.

  §4 covers seven options across both dimensions.
* **Recommendation (§5):** use **workspaces** for the structure (`Runs`,
  `Sweeps`, `Compare`, `System`), shown as **one DOS-style top row whose
  titles are both the workspaces and their menus**, so every command can be
  found without a second bar. Add a **`:` goto line** that accepts the
  CLI's own addresses. All of these are driven by a single
  **command registry**. A global **run tray** holds the set being compared.
  The `Runs` workspace keeps today's list-and-plots idea, reorganised as a
  sweep ▸ run ▸ phase tree beside fixed plot and log panes.

---

## 1. The current navigation

Captured from the test fixtures at 120×34:

```
 ⭘                                          utrain
╭─ 1 runs ───────────────────────────────╮    2 Config     3 Plots     4 Logs
│ ID  NAME              STATUS           │╭─ live  --  utrain-fake on cpu  --  running ─────────╮
│ c   live              running          ││ run                                                 │
│ b   draft             configuring      ││ id                          cccccccccccccccc        │
│ a   tiny-shakespeare  done             ││ name                        live                    │
│                                        ││ ...                                                 │
╰────────────────────────────────────────╯╰─────────────────────────────────────────────────────╯
 ⏎ open  r refresh  ? help  i images  c compute  e edit  s start  S stop  n new  d delete  R restart  t chat▏^p palette
```

On a phase with plots, the footer runs out of room earlier still:

```
 l log y  b charset  E export  r refresh  ? help  i images  c compute  e edit  s start  S stop  d delete  R ▏^p palette
```

The current model (see the module docstring of `tui/screens.py`):

| Element | Mechanism |
|---|---|
| Runs → phases | One sidebar list with two levels. `enter` drills in, `escape` goes back up. It remembers the last phase visited per run. |
| Config / Plots / Logs | Content tabs selected with `2`/`3`/`4`, remembered per phase |
| Metrics | A drawer (`m`) that picks which curves are drawn |
| Images, compute | Modal panels over the main screen (`i`, `c`) |
| Chat | A separate full screen (`t`) |
| Commands | Single-letter bindings in the footer; `?` help and the `^p` palette |

**What works and should be kept:**

* Navigation is selection, not a screen stack. One snapshot fills every
  pane, so moving the cursor down a list works as a flip-book (the content
  follows the cursor).
* The layout is a live monitor: the list shows status, and the content
  shows the current phase.
* Panels are modal and suspend the screen below them, so escape returns to
  exactly where you were.

**What breaks as features are added:**

1. **The footer is full.** Sweeps add `new sweep`, `pause`, `resume`,
   `extend`, `retry failed`, `cancel`, `mark`, `compare`, `color by`,
   `reducer` and more. They cannot all be single keys in one footer.
2. **The key namespace is nearly used up.** `s S n d R r e i c t m l b E x
   y space 1-4` are taken. Many of the natural sweep letters (`s` for sweep,
   `c` for compare, `r` for retry) already mean something else.
3. **Everything is scoped to one run.** Comparison needs a view over a set
   of runs, which is a different subject for the screen, not another tab.
4. **The hierarchy gains levels and becomes a graph.**
   `sweep → run → phase → {metrics, log}`, plus `run → image`,
   `run → compute`, `run set ← {sweep, marks, filter}`. A two-level list
   cannot hold all of that.
5. **The sidebar is 42 cells wide.** Sweep runs differ by their parameters
   (`lr=3e-4 L=6`), and that text competes with the name for those cells.
6. **Knowledge is written down three times:** the footer bindings,
   `_HELP`, and the gating sets (`_EDIT_MODE_OFF`, `_LIFECYCLE`,
   `check_action`). Every new command has to be added to all three.

---

## 2. Object model: what there is to navigate

```
                    ┌─────────┐        ┌─────────┐
                    │  image  │        │ compute │
                    └────▲────┘        └────▲────┘
                         │ frozen id        │ assigned at dispatch
 ┌─────────┐ generates ┌─┴──────────────────┴─┐  has  ┌─────────┐  has  ┌─────────────────┐
 │  sweep  ├──────────►│          run          ├──────►│ attempt ├──────►│      phase      │
 └────┬────┘           └───────────▲───────────┘       └─────────┘       │ metrics · log   │
      │ is a                       │ member of                           └─────────────────┘
      ▼                            │
 ┌─────────┐   ◄── marks / filter / sweep
 │ run set ├─────────────► compare lenses: curves · table · response · heatmap · diff
 └─────────┘
```

It helps to separate **nouns**, the things you select, from **lenses**, the
ways you look at a selection:

| Noun | Lenses |
|---|---|
| run | config, plots, logs, chat |
| phase | plots, log, metrics |
| sweep | grid/status matrix, queue, spec |
| run set | curves, table (leaderboard), response, heatmap, config diff |
| compute | utilisation, what is running where |
| image | description, phases, runs using it |

The navigation options in §4 differ mainly in how they lay out nouns
(tree, tabs, stack, columns) and whether lenses are tabs, panes or tiles.

---

## 3. Feature design

### 3.1 Sweeps

#### Spec

A sweep is a YAML spec. The CLI accepts it as a file and the TUI edits it
as a form:

```yaml
name: lr-depth
image: shakespeare-char        # the image id is frozen once, at sweep creation (cf. #40)
base: 7                        # optional: copy this run's config; otherwise image defaults
axes:
  phases.pretrain.learning_rate: {log: [1.0e-4, 1.0e-2], num: 5}
  globals.model.n_layer: [4, 6, 8]
  phases.pretrain.seed: {values: [0, 1], replicate: true}
mode: grid                     # v1; later: zip, random {n: 20}
compute: [gpu0, gpu1]          # pool; "cpu" allowed with parallel > 1
parallel: 2                    # max concurrent runs of this sweep
```

* **Axis paths** use the config's own shape (`globals.<group>.<key>`,
  `phases.<phase>.<key>`). They are validated against the image's
  `config_schema`, so values are coerced and range-checked by the existing
  `runs._coerce`, and an enum axis may only list that field's `options`.
* **Value forms:** `[a, b, c]`, `{lin: [lo, hi], num}`, `{log: [lo, hi], num}`
  and `{values: [...]}`. A bool axis defaults to `[false, true]`, and an enum
  axis to all of its options.
* **`replicate: true`** marks an axis, usually a seed, whose runs should be
  aggregated rather than told apart when compared (mean with a min/max
  band). This one flag makes comparisons of noisy runs much easier to read.
* **Grid size** is shown live as the spec is edited (`5 × 3 × 2 = 30
  runs`). If a base run has finished, the preview also estimates total time
  from that run's phase durations divided by `parallel`.

#### Storage

```
sweeps:   id, name, image, image_id, spec (json), status, created_at
runs:   + sweep_id  (nullable, FK)
        + sweep_point (json: {axis path: value}, the point's coordinates)
```

This follows the same `ALTER TABLE ... ADD COLUMN` approach as the existing
`db._migrate`. Runs are created eagerly, one per grid point, with status
`queued`. The whole grid is therefore visible, addressable and deletable
from the start, and `utrain run list` shows it without any sweep-specific
code.

Sweep status is derived from its runs plus a dispatch flag:
`draft → running ⇄ paused → done | cancelled`. A sweep with failed runs is
still `done`; the failures are shown on the runs.

#### Dispatch

Today a run is started right away on a compute fixed at creation. A sweep
needs a queue:

* `utrain sweep start` spawns a **detached dispatcher**
  (`utrain sweep dispatch <id>`), following the same pattern as the
  per-run orchestrator: detached, holding a lock, with all state in the DB.
* Each tick, the dispatcher reconciles, finds free slots, assigns a queued
  run to a compute, and calls `runs.start_run`. A slot counts as free when
  **no run at all** is running on that compute, so the user's manual runs
  are respected.
* **Compute is assigned when the run is dispatched**, which relaxes the
  "frozen at creation" rule for `queued` runs only. The image id stays
  frozen for the whole sweep, so every point runs the same bits. That
  matters for comparability.
* **Crash handling:** `reconcile` already handles an orchestrator that has
  died. It gains a rule that restarts the dispatcher for a sweep that is
  `running` but has no live dispatcher.
* **Dispatch order:** by default, the first point of each distinct
  upstream config goes first. See the open question in §7 on cacheable
  phases: today deduplication into the store runs only when the whole run
  completes, so parallel siblings cannot share a cacheable `tokenizer`
  phase until one of them finishes.

#### Lifecycle commands (CLI and TUI)

| Command | Effect |
|---|---|
| `sweep create <spec.yaml>` / `--from RUN --axis PATH=VALUES ...` | Create the sweep and its queued runs |
| `sweep start` / `pause` / `resume` | Start or stop dispatching. Running runs continue. |
| `sweep cancel` | Stop the running runs, drop the queued ones |
| `sweep extend --axis PATH=MORE` | Add grid points. Existing points are kept, matched by `sweep_point`. |
| `sweep retry` | Requeue failed runs |
| `sweep show` / `list` / `delete` | As for runs |

Addresses could extend the CLI's grammar with a sweep form such as
`@lr-depth` or `@lr-depth/lr=3e-4,n_layer=6`. This would give the TUI goto
line (§4.D) and the CLI one shared way to name sweeps and their points.

### 3.2 Comparison

#### Run sets

A **run set** is an ordered list of runs, plus the axes along which they
differ. It can come from:

* a **sweep**: all of its runs, with the sweep's axes;
* **marks**: `space` on runs in any list, collected in a global *tray*
  shown in the top row as `4 marked`;
* a **filter** (later): `status=done image=shakespeare-char`.

When a set does not come from a sweep, its "axes" are computed as **the
config keys whose values differ across the set**. The same computation
drives the config-diff lens and the default table columns, so ad-hoc
comparisons are as readable as sweep ones.

Sets start out ephemeral. Saving a set as a named *comparison* is a later,
optional step (§7).

#### Alignment

* Phases are matched **by name** across runs, and metrics by name within
  the phase. If the set spans several images, the phase picker lists the
  union and marks where each phase is missing.
* The x axis is shared: `step` or `elapsed`, as today.
* A **reducer** turns a curve into a number for tables, response plots and
  heatmaps: `last`, `min`, `max`, or `mean of last k`. The default comes
  from the name: `min` for anything matching `loss`, otherwise `last`.
  The user can override it.

#### Lenses

1. **Curves.** All runs overlaid on one plot. uniplot supports several
   series with colours and a legend, but a terminal offers only about six
   distinguishable ANSI colours, and in braille mode one cell shows one
   colour. So:
   * **Colour by axis:** colour encodes the value of one chosen axis
     (`color by: lr`), so 30 runs need 5 colours. Line style cannot vary in
     a text plot, so the other axes go in the legend and the table.
   * **Highlight:** the cursor's run is drawn in the accent colour and the
     rest in dim grey. Moving the cursor through the set flips through the
     runs.
   * **Replicates** collapse to a mean, optionally with a min/max band.
   * **Small multiples:** a grid of mini-plots, one per run or per axis
     value, sharing the y scale. This is the fallback when overlays get too
     crowded.
2. **Table (leaderboard).** One row per run. Columns are the axes (or
   differing keys), status, duration, and the reduced value of each chosen
   metric. It can be sorted by any column, and `enter` jumps to the run.
   This is the most-used lens.
3. **Response.** x is one axis (for example `lr`, log-scaled when the axis
   is), y is the reduced metric, and there is one series per value of a
   second axis. This is the standard learning-rate sweep plot.
4. **Heatmap.** Two axes as rows and columns, with the reduced metric in
   each cell as a number on a colour ramp. Terminals render this well. For
   a live sweep, the same matrix shows **status glyphs** in cells that have
   not finished, so the sweep monitor and its result are one view.
5. **Config diff.** Only the keys that differ, one column per run. With two
   runs, this is a side-by-side view.

#### Data

`data.Data.snapshot` is built for one run and one phase, keeping at most
eight `metrics.Tail`s open. Comparison needs a batch read over many runs:

* Add a `CompareSnapshot`: one worker reads **only the selected metric
  columns** for every run in the set, caching by `(path, mtime, size)` so
  finished runs are read once.
* Add a **`run_metric_summary` table** (`run, attempt, phase, metric, last,
  min, max, n`), written when a phase ends and filled lazily for runs that
  predate it. Table, response and heatmap then cost one query even for
  hundreds of runs. Only the curves lens reads full series.

---

## 4. Navigation options

For each option: a mockup (≈100×30), how it handles the six **reference
tasks**, and trade-offs.

> **T1** check a live run's loss · **T2** launch an lr × depth sweep ·
> **T3** monitor the sweep and the GPUs · **T4** find the best run of the
> sweep and read its log · **T5** compare a baseline run from last week
> with the sweep's best · **T6** retry the sweep's failed runs

Remember the two dimensions. Options **A, C, E, F, G** are *structures*.
Options **B and D** are *command surfaces* and combine with any structure.

### A. Lazygit+ (extend what exists)

The sidebar gets a grouping level: a sweep is a collapsible row with a
progress bar, and runs outside any sweep sit below. `space` marks runs. A
fifth content tab, `5 Compare`, shows the marked set, or the sweep under
the cursor when nothing is marked. The footer becomes strictly contextual.

```
╭─ 1 runs ─────────────────────────────╮  2 Config   3 Plots   4 Logs  [5 Compare]
│    NAME                  STATUS      │╭─ lr-depth · 12 runs · 4 done 2 running 6 queued ───╮
│ ▾  lr-depth              ▰▰▰▱▱▱ 4/12 ││ val/loss vs step · pretrain · colour: lr           │
│ ◆  ├ lr=1e-4 L=4         done        ││ 3.2┤⠑⢄                          ── 1e-4            │
│ ◆  ├ lr=1e-4 L=8         done        ││    │ ⠈⠢⡀⠑⢄                      ── 3e-4            │
│    ├ lr=1e-3 L=4         running     ││    │    ⠈⠑⠢⢄⡀⠉⠒⠤⣀                ── 1e-3            │
│    └ lr=1e-3 L=8         queued      ││ 1.2┤         ⠉⠉⠒⠒⠤⠤⣀⣀⣀⣀⣀⣀⣀⣀                    │
│ ▸  warmup                ▰▰▰▰▰▰ 8/8  ││    └────────────────────────────── step        │
│    live                  running     │╰────────────────────────────────────────────────────╯
│    draft                 configuring │
╰──────────────────────────────────────╯
 ◆ 2 marked  space mark  enter open  5 compare  ? help                            ^p palette
```

* T1 is unchanged. T2 is `N` (new sweep) on the list, then a form. T3 uses
  the sweep row, but GPUs are still behind the `c` modal, so the two are
  never on screen together. T4: `5`, then a table sub-view, then enter
  drills. T5: mark the baseline, mark the best, `5`. T6: `R` on the sweep
  row.
* **For:** cheapest option. Muscle memory survives, and the flip-book
  behaviour extends to sweeps naturally.
* **Against:** it does not fix the command-surface problem (§1.1–1.2),
  only reorders it. Compare squeezed into one tab of a single-run layout
  loses the full width that the table and heatmap want. `MainScreen` is
  already about 1,400 lines and would absorb both features.

### B. DOS / Turbo Vision menu bar (a command surface)

A permanent one-row menu at the top. `F10` or a click opens it, arrows move
through it, `enter` picks, and `alt+letter` jumps straight to a menu. Every
item shows its accelerator key on the right, which is how the menu teaches
the shortcuts. The footer becomes a thin status line.

```
 File  Run  Sweep  Compare  View  Window  Help                  gpu0 ▰▰▰▰▱ 97%  gpu1 ▰▱▱▱▱ 12%
       ┌──────────────────────────┐
       │ New sweep…            N  │
       │ New sweep from run…   ^N │
       │ ──────────────────────── │
       │ Pause dispatch        p  │
       │ Resume dispatch       P  │
       │ Extend grid…          +  │
       │ Retry failed runs        │
       │ Cancel sweep…            │   ← greyed: no sweep selected
       │ ──────────────────────── │
       │ Compare sweep runs    C  │
       └──────────────────────────┘
 ◆ 2 marked │ lr-depth running 4/12 │ F1 help  F10 menu  : goto                      12:04
```

Taken further, this becomes a **Turbo Vision desktop**: every view opens
as a numbered window, the `Window` menu offers Tile, Cascade, Next (`F6`)
and Zoom (`F5`), and `alt+1..9` jumps to a window. With tiling only
(Textual has no overlapping window manager), a sweep session could look
like this:

```
 File  Run  Sweep  Compare  View  Window  Help
╔═[■]═ 1 Sweep lr-depth ═══════════════════════╗┌─ 2 Curves: val/loss · lr-depth ───────────┐
║            L=4       L=6       L=8           ║│ ⠑⢄                                        │
║ lr=1e-4    ● 1.41    ● 1.38    ◐ …           ║│  ⠈⠢⡀⠑⢄                                    │
║ lr=3e-4    ● 1.29    ◐ …       ○             ║│     ⠈⠑⠢⢄⣀                                 │
║ lr=1e-3    ✗         ○         ○             ║│          ⠉⠉⠒⠒⠤⠤⣀⣀⣀⣀⣀                      │
╚══════════════════════════════════════════════╝└───────────────────────────────────────────┘
┌─ 3 Log: a3/pretrain ─────────────────────────────────────────────────────────────────────┐
│ step 4100 loss 1.402 lr 3.0e-4 tok/s 41233                                                │
└───────────────────────────────────────────────────────────────────────────────────────────┘
 F1 Help  F5 Zoom  F6 Next  F10 Menu  Alt-X Exit
```

* **For:** solves discoverability outright, with room for dozens of
  commands without key collisions. Greyed items explain what is possible
  without hiding it (unlike today's footer, which hides out-of-context
  keys). Mouse-friendly. It combines with every structure below.
* **Against:** menus are slow for frequent actions, which still need
  direct keys. A menu is a way to issue commands, not a map, so it does not
  answer "where am I". `alt+letter` is unreliable in some terminals and
  multiplexers, and `escape` doubles as the alt prefix, which conflicts
  with escape-as-back. So `F10` and the mouse are the primary ways in, and
  `alt` is a bonus. The full window manager (desktop, tile, cascade) is a
  large build in Textual and worth it only in the tiling form.
* **Implementation:** a `MenuBar` widget docked at the top, and a
  `MenuScreen` modal drawn at the menu's x offset holding an `OptionList`.
  Items, enablement and accelerators come from the command registry (§5.2).

### C. Workspaces (top-level tabs)

Top-level destinations, each with its own layout and its own remembered
state, one key apart. This follows the model of btop's views, htop's F-key
screens, and browser or vim tab pages.

```
 utrain  F1 Runs  [F2 Sweeps]  F3 Compare  F4 System        ◆ 2   gpu0 97%  gpu1 12%
╭─ sweeps ──────────────────╮╭─ lr-depth · shakespeare-char@3f2a · gpu0,gpu1 ×2 · min val/loss ─╮
│ lr-depth   running  4/12  ││             n_layer=4     n_layer=6     n_layer=8                 │
│ warmup     done     8/8   ││ lr=1e-4     ● 1.41        ● 1.38        ◐ 1.52 ↓                  │
│ seeds      paused   3/10  ││ lr=3e-4     ● 1.29 ★      ◐ 1.40 ↓      ○ queued                  │
│                           ││ lr=1e-3     ✗ failed      ○ queued      ○ queued                  │
│                           ││ lr=3e-3     ○ queued      ○ queued      ○ queued                  │
│                           │├─ queue ─────────────────────────────────────────────────────────┤
│                           ││ gpu0  lr=3e-4 L=6   pretrain  step 4100/5000   eta 6m            │
│                           ││ gpu1  lr=1e-4 L=8   pretrain  step 1200/5000   eta 21m           │
╰───────────────────────────╯╰────────────────────────────────────────────────────────────────╯
 enter open run  space mark  p pause  + extend  R retry failed  C compare   ? help
```

```
 utrain  F1 Runs  F2 Sweeps  [F3 Compare]  F4 System        ◆ 2   gpu0 97%  gpu1 12%
╭─ set: lr-depth (12) · phase pretrain · metric val/loss · reduce min · colour lr ──────────────╮
│ [curves]  table   response   heatmap   diff                                                   │
│  RUN  lr      n_layer  STATUS   val/loss ▲  train/loss  tok/s   DURATION                      │
│ ▶ a4  3e-4    4        done     1.29        1.11        41k     42m                           │
│   a2  1e-4    6        done     1.38        1.24        33k     58m                           │
│   a1  1e-4    4        done     1.41        1.30        41k     44m                           │
│   ... (the curves pane sits above the table and highlights the ▶ row)                        │
╰───────────────────────────────────────────────────────────────────────────────────────────────╯
```

* T1: `F1`, which is today's screen unchanged. T2: `F2`, then `N`. T3: `F2`
  shows the matrix, queue and GPU meters in the header. T4: `F3` sorted by
  the metric, then `enter` jumps to `F1` on that run's pretrain log. T5:
  mark the baseline in `F1`, mark the best in `F2`, then `F3` shows the
  tray. T6: `R` in `F2`.
* **For:** matches the user's actual modes (watching, running a sweep,
  analysing, housekeeping). Each screen stays simple and is built for its
  job. Today's `MainScreen` becomes `Runs` with little change. Textual
  supports this directly through `App.MODES` and `switch_mode`, with one
  screen stack per mode, so a chat or modal opened in one workspace stays
  there. Only the visible workspace needs to poll.
* **Against:** moving between workspaces needs explicit **jump** commands
  ("show this run in Runs", "compare this sweep"), and those must carry
  the selection across. Two places now show runs (the Runs list and inside
  sweeps). Compute and images move from quick modals to a full workspace;
  the modal shortcuts `i` and `c` can stay as glances.

### D. Command line and resource browser (k9s / vim `:`)

Every view is a table of one resource kind, with `enter` to drill down,
`escape` to go up, and a breadcrumb trail. A `:` command line (with
completion) jumps to any resource kind or address, and `/` filters the
current table.

```
 utrain ▸ sweeps ▸ lr-depth ▸ runs  (12)                          gpu0 97% gpu1 12%   ◆ 2
 :runs sweep=lr-depth status!=done█
  ┌──────────────────────────────────────┐
  │ :runs      runs (filter k=v …)       │
  │ :sweeps    sweeps                    │
  │ :compare   the tray, or @sweep       │
  │ :compute   GPUs and CPU              │
  │ :images    images                    │
  │ :a3/pretrain   jump to a phase       │
  └──────────────────────────────────────┘
   ID   lr     n_layer  STATUS    PHASE      STEP        val/loss
   a3   1e-3   4        running   pretrain   2210/5000   1.77
   a6   1e-4   8        running   pretrain   1200/5000   1.95
   a7   1e-3   6        queued    –          –           –
 <enter> open  <space> mark  <c> compare  </> filter  <:> command  <?> help
```

* T1: `:c` (a run-id prefix) and `enter`. T2: `:sweep new`, which opens the
  form. T3: `:sweeps`, then `enter`. T4: `:compare @lr-depth`, sort, then
  `enter`. T5: `:compare 7 @lr-depth/best`. T6: `:sweeps`, then `R`.
* **For:** scales to any number of resource kinds with **one** generic
  `ResourceScreen(kind, filter)`. It is the fastest option for experts.
  The TUI and CLI share one vocabulary: `address.parse` already gives
  every run, attempt and phase a canonical name, and a TUI that accepts
  `7/2/pretrain` teaches the CLI and the other way round. Filters in the
  command line become the "filter" source of run sets for free.
* **Against:** you cannot see what you do not know to type. The completion
  popup and the existing `^p` palette help, but it is still recall rather
  than recognition. A stack of full-screen tables also loses the
  everything-at-once monitor that the current design was built around.
  **As a secondary surface (a goto line) this is excellent; as the primary
  structure it gives up too much.**

### E. Miller columns (ranger, nnn, Finder)

The hierarchy laid out as sliding columns, with the parent always visible
on the left and a preview on the right. `h`/`l` move between levels and
`j`/`k` move within one.

```
 sweeps            │ runs · lr-depth         │ phases · a3     │ pretrain · val/loss vs step
   lr-depth    4/12│   a1 lr=1e-4 L=4  done  │   tokenizer ✓   │ 3.2┤⠑⢄
   warmup      8/8 │   a2 lr=1e-4 L=8  done  │ ▶ pretrain  ◐   │    │ ⠈⠢⡀
   seeds       3/10│ ▶ a3 lr=1e-3 L=4  run   │   eval      ○   │    │    ⠈⠑⠢⢄⣀
   (no sweep)      │   a4 lr=1e-3 L=8  queue │                 │ 1.2┤        ⠉⠉⠒⠒⠤⠤⣀⣀⣀
                   │   …                     │                 │    └──────────── step 4100
 ◆ 2   h/l level  j/k move  space mark  tab lens: plot/log/config  C compare  ? help
```

* **For:** the tree `sweep → run → phase` is always visible as context,
  which a two-level sidebar cannot do. One consistent gesture works
  everywhere. The flip-book behaviour becomes the core idea: moving down
  the runs column is comparing siblings one at a time.
* **Against:** images, compute and run sets are not part of the tree, so
  they need a second mechanism anyway. Columns spend width on context that
  plots and logs want, so it needs about 140 columns to be comfortable.
  Comparison is an operation on a set, and columns have no way to show
  one. It is a good fit for monitoring and browsing, and a poor fit for
  analysis.

### F. Tiling dashboard (tmux / i3 / Grafana)

The screen is split into tiles, and each tile holds any lens bound to any
noun. A tile is either **pinned** (a specific run, sweep or set) or
**follows** the selection. Layouts are named and saved.

```
┌ queue · lr-depth ──────────────────┬ val/loss · lr-depth · colour lr ────────────────────────┐
│ gpu0  a3 lr=1e-3 L=4  4100/5000 6m │ ⠑⢄                                                      │
│ gpu1  a6 lr=1e-4 L=8  1200/5000 21m│  ⠈⠢⡀⠑⢄                                        ── 1e-4   │
├ compute ───────────────────────────┤     ⠈⠑⠢⢄⣀⠉⠒⠤⣀                                  ── 3e-4   │
│ gpu0 ▰▰▰▰▰▰▰▰▰▱ 97%  22.1/24 GB    │          ⠉⠉⠒⠒⠤⠤⣀⣀⣀⣀⣀⣀                        ── 1e-3   │
│ gpu1 ▰▱▱▱▱▱▱▱▱▱ 12%   3.4/24 GB    │                                                         │
├ log · follows ▶ (a3/pretrain) ─────┴─────────────────────────────────────────────────────────┤
│ step 4100 loss 1.402 lr 1.0e-3 tok/s 41233                                                   │
└ layout: sweep-watch   ctrl+w s/v split  ctrl+w hjkl move  ctrl+w p pin  ctrl+w L layouts ─────┘
```

* **For:** the best possible monitoring: queue, GPUs, overlaid curves and
  the failing run's log on one screen. Power users will like it.
* **Against:** configuration is left to the user, and the defaults decide
  whether anyone uses it. Implementing a split tree, focus and
  pin-or-follow linking in Textual is the most work of any option. It is
  worth building as a **saved "board" inside a workspace** (for example
  Compare or Sweeps), not as the app's main navigation.

### G. Sheets (VisiData)

Everything is a table ("sheet"), and every operation derives a new sheet
pushed onto a stack: filter, sort, frequency, **pivot**. Plotting a column
is a lens on a sheet.

```
 runs ▸ sweep=lr-depth ▸ pivot lr × n_layer → min(val/loss)
              n_layer=4    n_layer=6    n_layer=8
 lr=1e-4      1.41         1.38         1.52
 lr=3e-4      1.29 ★       1.40         –
 lr=1e-3      ✗            –            –
 lr=3e-3      –            –            –
 sheets: runs › lr-depth › pivot   W pivot  F frequency  . plot column  enter rows behind cell
```

* **For:** analysing a sweep is analysing a table, and here it is fully
  general: the leaderboard, heatmap and response plot are a sort, a pivot
  and a plot of the same runs sheet. `enter` on a pivot cell opens the runs
  behind it, including replicates. Filters and derived sheets can be
  composed arbitrarily.
* **Against:** logs, config forms and chat are not sheets. The learning
  curve is steep, and VisiData's users like it for that reason, but most
  users will not. **Its ideas belong inside Compare** (sort, pivot, and
  drilling from a cell to its runs), not as the app shell.

### Scorecard

`++` strong, `+` good, `0` neutral, `–` weak, `– –` poor.

| | A Lazygit+ | B Menu bar | C Workspaces | D Command line | E Miller | F Tiling | G Sheets |
|---|---|---|---|---|---|---|---|
| Discoverability | 0 | ++ | + | – | + | – | – – |
| Expert speed | + | – (alone) | + | ++ | + | + | ++ |
| Live monitoring at a glance | + | 0 | + | – | + | ++ | – |
| Fit for sweeps | 0 | 0 | ++ | + | + | + | + |
| Fit for comparison | – | 0 | ++ | + | – – | + | ++ |
| Reuse of today's code | ++ | + | + | – | – | – | – – |
| Build cost (lower is better) | ++ | + | + | + | 0 | – – | – |
| Combinable | – | ++ | + | ++ | 0 | + | 0 |

---

## 5. Recommendation

### 5.1 Shape: one bar of workspaces that are also menus, a goto line and a tray

Choose **C** as the structure and **B plus D (goto line only)** as the
command surface, with B's menu bar and C's workspace tabs **merged into a
single row**. Borrow **G's** sort, pivot and cell drill-down inside
Compare. Keep **F** as a possible later "board".

A separate menu bar and tab strip would be two rows naming the same
nouns (`Run`, `Sweep` and `Compare` menus above `Runs`, `Sweeps` and
`Compare` tabs). One row carries both jobs instead:

```
 Runs  [Sweeps]  Compare  System                                    2 marked    gpu0 97%  gpu1 12%
╭─ 1 swe┌──────────────────────────┐-depth · status (min val/loss) ───────────────────────────────╮
│ NAME  │ Go to Sweeps          F2 │       n_layer=4     n_layer=6     n_layer=8                  │
│▶lr-dep│ ──────────────────────── │-4     ● 1.41        ● 1.38        ◐ 1.52 ↓                   │
│ warmup│ New sweep…            N  │-4     ● 1.29 ★      ◐ 1.40 ↓      ○ queued                   │
│ seeds │ Pause dispatch        p  │-3     ✗ failed      ○ queued      ○ queued                   │
│ lr-fin│ Extend grid…          +  │-3     ○ queued      ○ queued      ○ queued                   │
│       │ Retry failed runs     R  │                                                              │
│       │ Cancel sweep…            │e  ◐ running  ○ queued  ✗ failed · cell: min val/loss · ★ best│
│       │ ──────────────────────── │──────────────────────────────────────────────────────────────╯
│       │ Compare this sweep    C  │e ────────────────────────────────────────────────────────────╮
│       └──────────────────────────┘ lr-depth-05  lr=3e-4 L6  pretrain  step 4100/5000  eta 6m    │
│                           ││ gpu1  lr-depth-03  lr=1e-4 L8  pretrain  step 1200/5000  eta 21m   │
│                           ││ next  lr-depth-06  lr=3e-4 L8                                      │
│                           │╰────────────────────────────────────────────────────────────────────╯
│                           │╭─ spec ─────────────────────────────────────────────────────────────╮
│                           ││ image  shakespeare-char@3f2a      base  baseline-0921              │
│                           ││ axes   lr (log) 1e-4…3e-3 ×4 · n_layer 4,6,8 → 12 runs             │
│                           ││ pool   gpu0, gpu1 · parallel 2   left  ~1h40 (8 runs)              │
╰───────────────────────────╯╰────────────────────────────────────────────────────────────────────╯
           tab pane  enter open  space mark  p pause  R retry  C compare  : goto  ? help  F10 menu
```

* **The two rows have separate jobs.** The top row says where you are
  and shows state that matters everywhere: the workspace titles, the tray
  and the GPU meters. The bottom row says what you can do here: the keys
  for the focused pane, plus `: goto`, `? help` and `F10 menu`. Nothing
  appears in both.
* **Each title is a place.** `F1`–`F4` switch to a workspace, and the
  active title is highlighted. There is no separate tab strip. Each
  workspace remembers its state.
* **Each title is also a menu** of the commands for *the selected item of
  that kind*. `F10` drops the menu of the workspace you are in; a click
  drops the menu of the clicked title. Pressing the F-key of the workspace
  you are already in also opens its menu, so `F2` `F2` means "go to Sweeps
  and show me what I can do there". Once a menu is open, `←`/`→` move to
  the neighbouring menus without switching workspace, `enter` runs the
  item, and `escape` or `F10` closes it. Each menu's first item is "Go to
  …" with the workspace's F-key, and every item shows its shortcut, which
  is how the keys are learned.
* **Menus can have submenus** (`Run ▸`, `Plot ▸`), opened with `→` or
  `enter` and closed with `←` or `escape`, so a menu stays short enough to
  fit the screen. Textual has no menu widget, so the bar, menus and
  submenus are built here: each menu is a modal screen holding an
  `OptionList`, positioned under its title, and a submenu is a second one
  pushed beside its item. The existing `^p` palette stays as a fuzzy
  search over every command, however deep it sits in a menu.
* **A menu drops down over the workspace**, under its title, like any DOS
  menu. The workspace does not move or resize, keeps refreshing underneath,
  and is not dimmed, so the live state you may be acting on stays visible
  around the menu.
* **Commands act on the selection, wherever it is.** Opening the `Runs`
  menu while in Compare restarts or opens the run under the Compare
  cursor; items that don't apply are greyed. Opening a menu never switches
  workspace, and only the "Go to" items do. So a run's commands are
  reachable from every workspace, not only from Runs.
* **Workspaces:**
  * **Runs:** a sweep ▸ run ▸ phase tree on the left, and fixed plot and
    log panes for the selected phase on the right.
  * **Sweeps:** a list of sweeps, and for the selected one the status and
    heatmap matrix, the queue, and the spec. Creation opens a form derived
    from the config schema: each field gets a "sweep this" toggle, and a
    live grid-size preview is shown.
  * **Compare:** the tray or a sweep, shown through the curves, table,
    response, heatmap and diff lenses. The table sits under the curves,
    and the cursor row is highlighted in the plot.
  * **System:** compute and images as full views. `i` and `c` stay as the
    quick modal glances they are today.
* **The run tray** is global: `space` marks a run in any list of runs. The
  top row shows `n marked` only while something is marked, and clicking it
  opens Compare on those runs.
* **The `:` goto line** accepts CLI addresses (`7/2/pretrain`), sweep
  addresses (`@lr-depth`) and resource kinds (`:images`). This connects the
  TUI and CLI vocabularies and makes cross-workspace jumps a single
  command.
* **The footer** is reduced to the 4–6 keys that matter for the focused
  pane, plus `: goto`, `? help` and `F10 menu`. The menus list everything
  else.

#### Focus and scrolling (all workspaces)

Lists, grids and tables can outgrow their pane, so the keyboard has to
know which pane it is talking to. Every workspace follows the rules the
current screen already uses:

* **One pane has keyboard focus**, shown by its accent border. Arrow keys,
  `PgUp`/`PgDn` and `Home`/`End` go to it.
* **`tab`/`shift+tab` move focus** between panes, and number keys jump to
  one. A pane that takes focus carries its number in its title (`1 runs`,
  `1 sweeps`, `2 lr-depth · status`), as the runs pane does today.
* **The mouse wheel scrolls the pane under the pointer** without moving
  focus (Textual's default), so the mouse needs no focus step.
* **In a list, grid or table, the cursor does the scrolling.** Moving the
  cursor past the edge scrolls the pane, so a two-dimensional grid needs
  no scroll keys of its own.
* **Few panes take focus.** A pane that follows another (the Compare plot
  follows the table's cursor) or is bounded by design (the Sweeps queue
  and spec, the System data store) is not focusable, so `tab` has at most
  two or three stops.

#### Runs

Today's screen has a list that switches between runs and a run's phases
when you drill in, and a content pane whose view (Config, Plots or Logs)
you pick and which is remembered per phase. That makes it the odd one out:
every other workspace shows a list on the left and fixed panes for the
selection on the right. Runs follows the same shape:

```
 [Runs]  Sweeps  Compare  System                                    2 marked    gpu0 97%  gpu1 12%
╭─ 1 runs ───────────────────────────────╮╭─ 2 plots · lr-depth-04/pretrain · lr=3e-4 L4 ─────────╮
│  NAME            POINT      STATUS     ││ val/loss vs step · latest 1.29                        │
│  ▾ lr-depth      ▰▰▰▰▱▱▱▱   4/12       ││ 3.2┤⠑⢄                                                │
│ ◆  ▸ lr-depth-01 lr=1e-4 L4 done       ││    │  ⠈⠢⡀                                             │
│    ▸ lr-depth-02 lr=1e-4 L6 done       ││    │     ⠈⠑⠢⢄⡀                                        │
│    ▸ lr-depth-03 lr=1e-4 L8 running    ││    │          ⠈⠉⠒⠢⠤⣀⣀                                 │
│ ◆  ▾ lr-depth-04 lr=3e-4 L4 done       ││    │                 ⠈⠉⠉⠒⠒⠤⠤⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀             │
│        tokenizer            done       ││ 1.2┤                                                  │
│▶       pretrain             done       ││    └──────────────────────────────────── step 5000    │
│    ▸ lr-depth-05 lr=3e-4 L6 running    ││ train/loss · latest 1.11   (m: pick metrics)          │
│    ▸ lr-depth-07 lr=1e-3 L4 failed     │╰───────────────────────────────────────────────────────╯
│      6 queued                          │╭─ 3 log · lr-depth-04/pretrain ────────────────────────╮
│  ▸ warmup        ▰▰▰▰▰▰▰▰   8/8        ││ step 4800  loss 1.118  val 1.294  lr 3.0e-4           │
│  ▸ baseline-0921            done       ││ step 4900  loss 1.113  val 1.291  lr 3.0e-4           │
│  ▸ live                     running    ││ step 5000  loss 1.109  val 1.290  lr 3.0e-4           │
│  ▸ draft                    configuring││ saved checkpoint data/pretrain/ckpt.pt                │
╰────────────────────────────────────────╯╰───────────────────────────────────────────────────────╯
                    tab pane  enter expand  space mark  e config  z zoom  : goto  ? help  F10 menu
```

* **The list is a tree:** sweep ▸ run ▸ phase. `enter` (or `→`/`←`)
  expands and collapses a row in place, instead of the list switching to
  one run's phases. Runs outside any sweep sit at the top level.
* **The right side is always plots above the log**, for the selected
  phase. On a run row it shows the phase the run is on now, as today; on a
  sweep row, the sweep's status grid. There is no view to pick and no
  per-phase view state.
* **`z` zooms the focused pane** (plots or log) to the whole right side,
  and `z` again restores it, like tmux's zoom. That covers reading a long
  log or a detailed plot.
* **Moving the cursor still updates everything**, so stepping down a
  sweep's runs is still a flip-book of their curves.
* **`POINT`** shows each sweep run's parameter values with a short label
  per axis (`L` for `n_layer`); the sidebar keeps its 42-cell width.
  **`◆`** marks a run for comparison (`space`), and `▶` is the cursor.

Config does not need a permanent place, because it can only be changed
before a run starts: `runs.write_config` refuses any run that is not
`configuring`. So the two cases are handled differently.

**A configuring run edits its config in place.** The config is the
right-hand pane (`2`), shown instead of the plots the run does not have
yet. `e` puts the focus in it, and the fields edit in place as they do
today; `escape` leaves a field and `s` starts the run:

```
 [Runs]  Sweeps  Compare  System                                    2 marked    gpu0 97%  gpu1 12%
╭─ 1 runs ───────────────────────────────╮╭─ 2 config · draft ────────────────────────────────────╮
│  NAME            POINT      STATUS     ││ draft · utrain-fake on cpu · configuring              │
│  ▾ lr-depth      ▰▰▰▰▱▱▱▱   4/12       ││                                                       │
│ ◆  ▸ lr-depth-01 lr=1e-4 L4 done       ││ globals: Model                                        │
│    ▸ lr-depth-02 lr=1e-4 L6 done       ││   Layers                 4                            │
│    ▸ lr-depth-03 lr=1e-4 L8 running    ││   Precision              fp32                         │
│ ◆  ▾ lr-depth-04 lr=3e-4 L4 done       ││                                                       │
│        tokenizer            done       ││ phase: pretrain                                       │
│        pretrain             done       ││   Learning rate          0.001                        │
│    ▸ lr-depth-05 lr=3e-4 L6 running    ││   Resume                 false                        │
│    ▸ lr-depth-07 lr=1e-3 L4 failed     ││                                                       │
│      6 queued                          ││ e edit · s start · N new sweep from this run          │
│  ▸ warmup        ▰▰▰▰▰▰▰▰   8/8        ││                                                       │
│  ▸ baseline-0921            done       ││                                                       │
│  ▸ live                     running    ││                                                       │
│▶ ▸ draft                    configuring││                                                       │
╰────────────────────────────────────────╯╰───────────────────────────────────────────────────────╯
                            tab pane  e edit config  s start  space mark  : goto  ? help  F10 menu
```

**A started run's config opens in a read-only popup.** Once a run has
started, its config can only be looked at ("what learning rate did this
one use?"), so it is a glance rather than a view: `e` opens it over the
screen, the plots and log keep updating behind it, and `escape` closes it,
like the images and compute pop-ups. Swept values are flagged, and `N`
offers the usual next step, a new sweep around this run. The same key
serves both cases: the Run menu calls it "Edit config" on a configuring run
and "Show config" on any other. Comparing configs across runs is
Compare's config diff.

```
 [Runs]  Sweeps  Compare  System                                    2 marked    gpu0 97%  gpu1 12%
╭─ 1 runs ───────────────────────────────╮╭─ 2 plots · lr-depth-04/pretrain · lr=3e-4 L4 ─────────╮
│  NAME            POINT      STATUS     ││ val/loss vs step · latest 1.29                        │
│  ▾ lr-depth      ▰▰▰▰▱▱▱┌─ config · lr-depth-04 · read-only ───────────────┐                    │
│ ◆  ▸ lr-depth-01 lr=1e-4│ shakespeare-char@3f2a on gpu0 · done · 1 attempt │                    │
│    ▸ lr-depth-02 lr=1e-4│ sweep lr-depth, point lr=3e-4 L4                 │                    │
│    ▸ lr-depth-03 lr=1e-4│                                                  │                    │
│ ◆  ▾ lr-depth-04 lr=3e-4│ globals: Model                                   │⣀⣀⣀⣀⣀⣀⣀             │
│        tokenizer        │   Layers                 4        ◂ swept        │                    │
│▶       pretrain         │   Precision              bf16                    │────── step 5000    │
│    ▸ lr-depth-05 lr=3e-4│                                                  │k metrics)          │
│    ▸ lr-depth-07 lr=1e-3│ phase: pretrain                                  │────────────────────╯
│      6 queued           │   Learning rate          3e-4     ◂ swept        │────────────────────╮
│  ▸ warmup        ▰▰▰▰▰▰▰│   Batch size             64                      │lr 3.0e-4           │
│  ▸ baseline-0921        │   Max steps              5000                    │lr 3.0e-4           │
│  ▸ live                 │                                                  │lr 3.0e-4           │
│  ▸ draft                │ N new sweep from this run · escape close         │t.pt                │
╰─────────────────────────└──────────────────────────────────────────────────┘────────────────────╯
                                      escape close  N new sweep from run  : goto  ? help  F10 menu
```

The Runs menu (`F1` again, or `F10`), with its `Plot` submenu open. The
run lifecycle (`e s S R d t`) is under `Run ▸`, and the plot settings
that used to crowd the footer are under `Plot ▸`:

```
 [Runs]  Sweeps  Compare  System                                    2 marked    gpu0 97%  gpu1 12%
╭─┌──────────────────────────┐───────────╮╭─ 2 plots · lr-depth-04/pretrain · lr=3e-4 L4 ─────────╮
│ │ Go to Runs            F1 │STATUS     ││ val/loss vs step · latest 1.29                        │
│ │ ──────────────────────── │4/12       ││ 3.2┤⠑⢄                                                │
│ │ New run…               n │done       ││    │  ⠈⠢⡀                                             │
│ │ New sweep from run…    N │done       ││    │     ⠈⠑⠢⢄⡀                                        │
│ │ ──────────────────────── │running    ││    │          ⠈⠉⠒⠢⠤⣀⣀                                 │
│ │ Run                    ▸ │┌──────────────────────┐           ⠈⠉⠉⠒⠒⠤⠤⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀             │
│ │▶Plot                   ▸ ││ Metrics…           m │                                            │
│▶│ ──────────────────────── ││ Cycle x axis       x │────────────────────────────── step 5000    │
│ │ Mark for compare   space ││ Log y axis         l │ · latest 1.11   (m: pick metrics)          │
│ │ Show in Sweeps           ││ Braille / blocks   b │────────────────────────────────────────────╯
│ │ Expand / collapse  enter ││ Export plot…       E │r-depth-04/pretrain ────────────────────────╮
│ │ Zoom pane              z │└──────────────────────┘ loss 1.118  val 1.294  lr 3.0e-4           │
│ └──────────────────────────┘done       ││ step 4900  loss 1.113  val 1.291  lr 3.0e-4           │
│  ▸ live                     running    ││ step 5000  loss 1.109  val 1.290  lr 3.0e-4           │
│  ▸ draft                    configuring││ saved checkpoint data/pretrain/ckpt.pt                │
╰────────────────────────────────────────╯╰───────────────────────────────────────────────────────╯
                    tab pane  enter expand  space mark  e config  z zoom  : goto  ? help  F10 menu
```

#### Sweeps

```
 Runs  [Sweeps]  Compare  System                                    2 marked    gpu0 97%  gpu1 12%
╭─ 1 sweeps ────────────────╮╭─ 2 lr-depth · status (min val/loss) ───────────────────────────────╮
│ NAME       STATUS    DONE ││             n_layer=4     n_layer=6     n_layer=8                  │
│▶lr-depth   running   4/12 ││ lr=1e-4     ● 1.41        ● 1.38        ◐ 1.52 ↓                   │
│ warmup     done      8/8  ││ lr=3e-4     ● 1.29 ★      ◐ 1.40 ↓      ○ queued                   │
│ seeds      paused    3/10 ││ lr=1e-3     ✗ failed      ○ queued      ○ queued                   │
│ lr-fine    draft     0/6  ││ lr=3e-3     ○ queued      ○ queued      ○ queued                   │
│                           ││                                                                    │
│                           ││ ● done  ◐ running  ○ queued  ✗ failed · cell: min val/loss · ★ best│
│                           │╰────────────────────────────────────────────────────────────────────╯
│                           │╭─ queue ────────────────────────────────────────────────────────────╮
│                           ││ gpu0  lr-depth-05  lr=3e-4 L6  pretrain  step 4100/5000  eta 6m    │
│                           ││ gpu1  lr-depth-03  lr=1e-4 L8  pretrain  step 1200/5000  eta 21m   │
│                           ││ next  lr-depth-06  lr=3e-4 L8                                      │
│                           │╰────────────────────────────────────────────────────────────────────╯
│                           │╭─ spec ─────────────────────────────────────────────────────────────╮
│                           ││ image  shakespeare-char@3f2a      base  baseline-0921              │
│                           ││ axes   lr (log) 1e-4…3e-3 ×4 · n_layer 4,6,8 → 12 runs             │
│                           ││ pool   gpu0, gpu1 · parallel 2   left  ~1h40 (8 runs)              │
╰───────────────────────────╯╰────────────────────────────────────────────────────────────────────╯
           tab pane  enter open  space mark  p pause  R retry  C compare  : goto  ? help  F10 menu
```

* **Left, the sweeps;** right, three panes for the one under the cursor.
* **Status grid:** one cell per point, for the first two axes. A finished
  cell shows the reduced metric, a running one its current value with a
  trend arrow, and `★` marks the best. When the sweep is done the grid is
  its heatmap, so monitoring and the first result are one view. A sweep
  with more than two axes gets a picker for which two are shown; the rest
  are summarised (best, or mean over replicates).
* **Queue:** what is running on each compute, its phase and progress, and
  the next run to start. It is capped at one line per compute plus that
  "next" line, so it never scrolls.
* **Spec:** image, base run, axes, pool, and an estimate of the time left
  from the durations of finished points.
* `enter` on a cell opens that run in Runs, `space` marks it, and `C`
  opens the sweep in Compare.
* **Creating a sweep** needs an image first, because the form is built
  from the image's config schema. There are two ways in:
  * **Runs ▸ New sweep from run… (`N`)** takes the image and the starting
    config from the selected run and goes straight to the form. This is
    the common case: sweeping around a run you already like.
  * **Sweeps ▸ New sweep… (`N`)** first asks, in a dialog shaped like
    today's new-run dialog, for a name, the image, the base config (the
    image's defaults or one of its runs), the compute pool and how many
    runs at once.

  Both then open the same form: the config fields, each with a "sweep
  this" toggle that turns its value into a list or range, and a live grid
  size (`4 × 3 = 12 runs`).

The Sweeps menu is the one shown open at the top of this section.

#### Compare

```
 Runs  Sweeps  [Compare]  System                                    2 marked    gpu0 97%  gpu1 12%
╭─ compare · lr-depth · phase pretrain · metric val/loss · reduce min ────────────────────────────╮
│ [curves + table]   response   heatmap   diff              set lr-depth (12) ▾   tray 2 ▸        │
│                                                                                                 │
│ val/loss vs step · pretrain · colour: lr · ▶ highlighted                                        │
│ 3.2┤⠑⢄⠑⢄                                                                   ── lr=1e-4           │
│    │ ⠈⠢⡀⠈⠢⡀⠑⢄                                                              ── lr=3e-4           │
│    │    ⠈⠑⠢⢄⡀⠈⠑⠢⢄⣀                                                         ── lr=1e-3           │
│    │         ⠈⠉⠒⠒⠤⠤⣀⣀⠉⠒⠤⣀⣀                                                 ── ▶ lr-depth-04     │
│    │                 ⠈⠉⠉⠒⠒⠤⠤⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀                                                      │
│ 1.2┤                           ⠉⠉⠉⠉⠉⠒⠒⠒⠒⠤⠤⠤⠤⣀⣀⣀⣀⣀⣀⣀⣀⣀                                           │
│    └──────────────────────────────────────────────────────────────── step 5000                  │
│                                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│   RUN           lr     n_layer  STATUS    val/loss ▲  train/loss  tok/s  DURATION               │
│ ▶ lr-depth-04   3e-4   4        done      1.29        1.11        41k    42m                    │
│   lr-depth-02   1e-4   6        done      1.38        1.24        33k    58m                    │
│   lr-depth-05   3e-4   6        running   1.40 ↓      1.30        33k    31m                    │
│   lr-depth-01   1e-4   4        done      1.41        1.30        41k    44m                    │
│   lr-depth-03   1e-4   8        running   1.52 ↓      1.47        27k    12m                    │
│   lr-depth-07   1e-3   4        failed    –           –           –      3m                     │
│   6 queued runs not shown                                                                       │
╰─────────────────────────────────────────────────────────────────────────────────────────────────╯
 best lr-depth-04 · 1.29  enter open run  space mark  [ ] lens  < > sort  : goto  ? help  F10 menu
```

* **The set** comes from a sweep (`set lr-depth ▾` picks which) or from
  the tray (`tray 2 ▸`). The title line holds the phase, metric and
  reducer, each changeable from the Compare menu.
* **Lenses** are switched with `[`/`]`: curves and table together
  (shown), response, heatmap, and config diff. (`tab` is kept for moving
  between panes, as everywhere else.)
* **Curves** are coloured by one axis, with the cursor's run drawn in the
  accent colour. **The table** below has one row per run; `<`/`>` change
  the sort column. Moving the cursor in the table moves the highlight in
  the plot.
* `enter` opens the run in Runs on this phase; `space` adds it to the
  tray, which is how the best of a sweep is set against a baseline.

The Compare menu (`F3` again, or `F10`):

```
 Runs  Sweeps  [Compare]  System                                    2 marked    gpu0 97%  gpu1 12%
╭─ compare · lr┌────────────────────────────┐ic val/loss · reduce min ────────────────────────────╮
│ [curves + tab│ Go to Compare           F3 │f              set lr-depth (12) ▾   tray 2 ▸        │
│              │ ────────────────────────── │                                                     │
│ val/loss vs s│ Compare the tray (2)       │▶ highlighted                                        │
│ 3.2┤⠑⢄⠑⢄     │ Compare a sweep…           │                                ── lr=1e-4           │
│    │ ⠈⠢⡀⠈⠢⡀⠑⢄│ Clear the tray             │                                ── lr=3e-4           │
│    │    ⠈⠑⠢⢄⡀│ ────────────────────────── │                                ── lr=1e-3           │
│    │         │ Phase…            pretrain │                                ── ▶ lr-depth-04     │
│    │         │ Metric…           val/loss │                                                     │
│ 1.2┤         │ Reduce by…             min │⠤⣀⣀⣀⣀⣀⣀⣀⣀⣀                                           │
│    └─────────│ Colour by…              lr │───────────────────────── step 5000                  │
│              │ Next lens                ] │                                                     │
├──────────────│ Sort by next column      > │─────────────────────────────────────────────────────┤
│   RUN        │ ────────────────────────── │al/loss ▲  train/loss  tok/s  DURATION               │
│ ▶ lr-depth-04│ Open run in Runs     enter │.29        1.11        41k    42m                    │
│   lr-depth-02│ Mark / unmark        space │.38        1.24        33k    58m                    │
│   lr-depth-05│ Export plot…             E │.40 ↓      1.30        33k    31m                    │
│   lr-depth-01└────────────────────────────┘.41        1.30        41k    44m                    │
│   lr-depth-03   1e-4   8        running   1.52 ↓      1.47        27k    12m                    │
│   lr-depth-07   1e-3   4        failed    –           –           –      3m                     │
│   6 queued runs not shown                                                                       │
╰─────────────────────────────────────────────────────────────────────────────────────────────────╯
 best lr-depth-04 · 1.29  enter open run  space mark  [ ] lens  < > sort  : goto  ? help  F10 menu
```

* The first group picks the set, the second how it is shown. For a
  setting (phase, metric, reducer, colour), the right-hand column shows
  its current value instead of a key; picking the item opens a list of
  the alternatives.

#### System

```
 Runs  Sweeps  Compare  [System]                                    2 marked    gpu0 97%  gpu1 12%
╭─ 1 compute ─────────────────────────────────────────────────────────────────────────────────────╮
│   ID    KIND  NAME            UTIL                MEMORY        POWER   RUNNING                 │
│ ▶ cpu   cpu   EPYC 7443 ×24   ▰▰▱▱▱▱▱▱▱▱  18%     41/256 GB     –       live                    │
│   gpu0  gpu   RTX A5000       ▰▰▰▰▰▰▰▰▰▱  97%     22.1/24 GB    214 W   lr-depth-05             │
│   gpu1  gpu   RTX A5000       ▰▱▱▱▱▱▱▱▱▱  12%     3.4/24 GB     61 W    lr-depth-03             │
│                                                                                                 │
│   queue: 6 runs of lr-depth waiting for gpu0 or gpu1                                            │
╰─────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ 2 images ──────────────────────────────────────────────────────────────────────────────────────╮
│   NAME                    ID     SIZE     RUNS  SWEEPS  ADDED                                   │
│   shakespeare-char:1.2    3f2a   2.1 GB   14    2       3d ago                                  │
│   utrain-fake:latest      9c1e   180 MB   3     0       20d ago                                 │
│   nanochat:0.4            77b0   6.8 GB   0     0       now     pulling ▰▰▰▰▰▰▱▱▱▱ 61%          │
╰─────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ data store ────────────────────────────────────────────────────────────────────────────────────╮
│   2 340 files · 18.4 GB · 312 MB orphaned (reclaim with G)                                      │
╰─────────────────────────────────────────────────────────────────────────────────────────────────╯
                             tab pane  a add image  d delete  G gc store  : goto  ? help  F10 menu
```

* **Compute:** each CPU and GPU with utilisation, memory, power, and
  which run is on it, plus how many sweep runs are waiting for a slot.
  This is what the top row's GPU meters summarise.
* **Images:** each image with its size, and how many runs and sweeps use
  it. A pull in progress shows its progress in place.
* **Data store:** the content-addressed store's size and what `gc` would
  reclaim, so `utrain store check`/`gc` have a place in the TUI.
* `tab` moves between compute (`1`) and images (`2`); the one-line data
  store pane does not take focus. `i` and `c` still open the images
  and compute pop-ups from any workspace for a quick look.

The System menu (`F4` again, or `F10`):

```
 Runs  Sweeps  Compare  [System]                                    2 marked    gpu0 97%  gpu1 12%
╭─ 1 compute ───────────┌────────────────────────────┐────────────────────────────────────────────╮
│   ID    KIND  NAME    │ Go to System            F4 │MORY        POWER   RUNNING                 │
│ ▶ cpu   cpu   EPYC 744│ ────────────────────────── │/256 GB     –       live                    │
│   gpu0  gpu   RTX A500│ Add image…               a │.1/24 GB    214 W   lr-depth-05             │
│   gpu1  gpu   RTX A500│ Delete image…            d │4/24 GB     61 W    lr-depth-03             │
│                       │ ────────────────────────── │                                            │
│   queue: 6 runs of lr-│ Check data store           │                                            │
╰───────────────────────│ Clean up store…          G │────────────────────────────────────────────╯
╭─ 2 images ────────────│ ────────────────────────── │────────────────────────────────────────────╮
│   NAME                │ Settings…                  │PS  ADDED                                   │
│   shakespeare-char:1.2│ Quit                     q │    3d ago                                  │
│   utrain-fake:latest  └────────────────────────────┘    20d ago                                 │
│   nanochat:0.4            77b0   6.8 GB   0     0       now     pulling ▰▰▰▰▰▰▱▱▱▱ 61%          │
╰─────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ data store ────────────────────────────────────────────────────────────────────────────────────╮
│   2 340 files · 18.4 GB · 312 MB orphaned (reclaim with G)                                      │
╰─────────────────────────────────────────────────────────────────────────────────────────────────╯
                             tab pane  a add image  d delete  G gc store  : goto  ? help  F10 menu
```

* `Quit` lives here, as `Exit` lived in a DOS program's `File` menu: the
  menu for the app itself is the natural place for it. `q` still quits
  from anywhere.
* `Settings…` edits `utrain.yaml`, where settings such as the plot
  character set (`tui_charset`) can already be written by hand or given
  as `UTRAIN_*` environment variables.

How the reference tasks play out:

| Task | Today | Proposed |
|---|---|---|
| T1 live run loss | select, `enter`, `3` | `F1`, select the run: its plots and log are already there |
| T2 launch sweep | *not possible* | `F2`, `N`, then form (or Sweeps ▸ New sweep from run… from any workspace) |
| T3 monitor sweep and GPUs | *not possible*, and GPUs are modal | `F2`: matrix, queue, GPU meters |
| T4 best run, then its log | *not possible* | `F3`, sort the metric, `enter` (jumps to `F1` on that run's phase, log under the plots) |
| T5 baseline vs best | *not possible* | `space` on the baseline in `F1`, `space` on the best in `F3`, `C` |
| T6 retry failed | `R` on each failed run | `F2`, `R` (Sweeps ▸ Retry failed runs) |

### 5.2 Prerequisite: a command registry

Whatever shape is chosen, first replace the three parallel lists
(bindings, `_HELP`, and the gating sets) with **one registry**:

```python
@dataclasses.dataclass(frozen=True)
class Command:
    id: str                       # "sweep.retry"
    label: str                    # "Retry failed runs"
    menu: str                     # "Sweep"
    key: str | None               # "R"
    scope: str                    # "sweeps" workspace / "runs.list" pane / "global"
    enabled: Callable[[Context], bool]   # replaces the check_action branches
    run: Callable[[Context], None]
    typing_safe: bool = False     # False → off while an editor has focus (_EDIT_MODE_OFF)
```

The menu bar, footer, help screen, `^p` palette (Textual's
`CommandProvider`), key bindings and `check_action` are all generated from
it. Adding a command then takes one line, not four edits, and the menu
greys out exactly what the key would refuse. This step is useful on its own
and has no visible effect other than a shorter footer.

### 5.3 Phasing

Each step can be shipped on its own:

1. **Command registry.** A refactor with no behaviour change. The footer
   and help screen are generated from it.
2. **Sweep backend and CLI:** the tables, the `queued` status, the
   dispatcher, `utrain sweep …`, and `run_metric_summary`. Testable with
   cram against the fake containers, as run lifecycle is today.
3. **Shell:** the top row, with its menus and submenus, and workspaces
   via `App.MODES`. `Runs` starts as today's `MainScreen`, and `System`
   holds compute and images.
4. **Runs as a tree:** the sweep ▸ run ▸ phase tree, fixed plot and log
   panes with `z` zoom, and config as an editor.
5. **Sweeps workspace:** the list, matrix, queue and create form.
6. **Tray and Compare workspace:** table and curves first, then heatmap,
   response and diff.
7. **Goto line:** addresses, `@sweep`, and resource kinds, with
   completion.
8. *(later)* Named comparisons, filter-defined sets, random and zip sweep
   modes, and a saved tiling "board".

---

## 6. Details worth deciding early

* **Refresh budget.** Only the visible workspace polls, and the status
  line gets a cheap global poll (tray count, sweep progress, GPU meters)
  every few seconds. The Compare snapshot polls only while the set
  contains a running run.
* **Colour.** Status colours already use Rich styles. The curve palette
  should avoid the colours that mean `failed`, `running` and `done`, so
  that a red curve does not read as a failed run.
* **Names.** Sweep runs are named `<sweep>-<nn>`. Their parameters appear
  in dedicated columns (the sidebar gains a `POINT` column inside a
  sweep), not in the name, so they stay sortable and are never truncated
  in the middle.
* **Deleting.** `sweep delete` deletes its runs, after a confirm that
  shows how many. Deleting one run of a sweep leaves a hole in the grid
  that `sweep extend` or `retry` can fill.
* **Chat.** It stays run-scoped (`t`) and opens in the current workspace's
  stack.

---

## 7. Open questions

1. **Grid only for v1,** or also `zip` (paired lists) and `random`
   sampling?
2. **Assigning compute at dispatch time** relaxes the rule that a run's
   compute is fixed at creation, for queued runs only. Is that acceptable,
   or should a sweep pre-assign compute round-robin and accept idle slots?
3. **Cache interaction.** Deduplication into the store happens only when a
   whole run completes, so parallel sweep siblings cannot share a
   cacheable upstream phase until one of them finishes. Should
   deduplication (or at least store insertion) happen per phase?
4. **Should comparisons be saved** (a `comparisons` table), or are the
   tray plus sweeps enough?
5. **One global tray,** or several named sets?
6. **Terminal constraints.** Do you use utrain over ssh, tmux, or terminals
   where `F10`, `F1`–`F4` or `alt` are intercepted? That decides which
   accelerators are primary and which are aliases.
7. **Appetite for the more radical shells.** If E (Miller) or F (tiling)
   appeals more than the recommendation, both can be prototyped as one
   extra workspace without committing the whole app to them.
