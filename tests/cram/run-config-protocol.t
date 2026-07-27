The utrain config protocol: `run create` writes config.yaml with keys nested
under group names (globals.<group>.<key>, phases.<phase>.<group>.<key>). A
container must read, for the phase it runs, the merge of top-level scalars +
every globals group + every group of that phase, as a flat namespace. The fake
container echoes the effective `num_layers` (a global) and `batch_size` (a
pretrain-phase value) so we can assert edits actually reach the container.

  $ . "$TESTDIR/setup.sh"
  $ utrain image add "$UTRAIN_TEST_IMAGE_URL" >/dev/null 2>&1

Defaults: an unedited config runs with the schema defaults.

  $ DEF=$(utrain run create --name defaults --image utrain-fake --compute cpu --print-id)
  $ grep -E 'num_layers|batch_size' "$UTRAIN_DATA_DIR/runs/$DEF/config.yaml"
      num_layers: 12
      batch_size: 32
  $ utrain run start "$DEF" >/dev/null
  $ utrain run show "$DEF" --wait >/dev/null
  $ utrain run logs "$DEF" --phase pretrain | grep '^config:'
  config: num_layers=12 batch_size=32

Edited: shrinking a nested global and a nested phase value both take effect.

  $ RID=$(utrain run create --name edited --image utrain-fake --compute cpu --print-id)
  $ sed -i 's/num_layers: 12/num_layers: 3/; s/batch_size: 32/batch_size: 7/' "$UTRAIN_DATA_DIR/runs/$RID/config.yaml"
  $ utrain run start "$RID" >/dev/null
  $ utrain run show "$RID" --wait >/dev/null
  $ utrain run logs "$RID" --phase pretrain | grep '^config:'
  config: num_layers=3 batch_size=7
