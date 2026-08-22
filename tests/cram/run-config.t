  $ . "$TESTDIR/setup.sh"
  $ utrain image add "$UTRAIN_TEST_IMAGE_URL" >/dev/null 2>&1
  $ RID=$(utrain run create --name hello --image utrain-fake --compute cpu --print-id)
  $ ls "$UTRAIN_DATA_DIR/runs/$RID"
  config.yaml
  $ head -2 "$UTRAIN_DATA_DIR/runs/$RID/config.yaml"
  run_id: [0-9a-f]{32} (re)
  compute: cpu
  $ grep 'num_layers' "$UTRAIN_DATA_DIR/runs/$RID/config.yaml"
      num_layers: 12
  $ utrain run start "$RID" >/dev/null
  $ ls -l "$UTRAIN_DATA_DIR/runs/$RID/config.yaml" | awk '{print $1}'
  -r--r--r--* (glob)
  $ ls "$UTRAIN_DATA_DIR/runs/$RID/attempt/1"
  logs
  mnt
  orchestrator.log
  wandb

`mnt` is the container's entire view of the host, bind-mounted read-only at
/utrain. It holds only the two files a phase may read; `data` and `wandb` are
empty stubs that exist so the writable mounts have somewhere to land.

  $ ls "$UTRAIN_DATA_DIR/runs/$RID/attempt/1/mnt"
  config.yaml
  control.json
  data
  serve
  wandb
  $ ls -l "$UTRAIN_DATA_DIR/runs/$RID/attempt/1/mnt/config.yaml" | awk '{print $1}'
  -r--r--r--* (glob)
