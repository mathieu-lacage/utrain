`utrain sweep`: grid searches. `utrain sweep create --image IMG --axis
phases.pretrain.learning_rate=log:1e-4:1e-2:5 --axis globals.model.n_layer=4,6,8
--compute gpu0,gpu1` (or a YAML spec file) creates one ordinary run per point of
the grid, with the new `queued` status and a compute fixed at creation, spread
across the listed computes. `sweep start` hands the sweep to a detached
dispatcher that starts each compute's next queued run whenever that compute is
free; `pause`, `resume`, `cancel`, `retry`, `extend`, `show`, `list` and
`delete` manage it from there.
