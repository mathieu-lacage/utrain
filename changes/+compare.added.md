The TUI has a Compare workspace (`F3`): the runs marked with `space`, a sweep's
runs (`C` in Sweeps), or a comparison saved under a name, seen through one
phase, one metric and one reduction of each curve (its minimum for a loss, its
last value otherwise). `[` and `]` step through the lenses -- every run's curve
over a table of the runs, the reduced metric against an axis, the sweep's grid
as a heatmap, and the config fields that differ -- and `<` `>` sort the table,
whose best run is starred. The comparison on screen is kept, and the app opens
on the workspace, run and sweep it was quit on. The Sweeps grid shows each
run's reduced metric too.
