  $ . "$TESTDIR/setup.sh"
  $ RID=$(utrain run create --name hello --image utrain-fake --gpu none --print-id)
  $ ls "$UTRAIN_DATA_DIR/runs/$RID"
  config.yaml
  $ head -3 "$UTRAIN_DATA_DIR/runs/$RID/config.yaml"
  run_id: [0-9a-f]{32} (re)
  output_dir: /run
  gpu: none
  $ grep 'num_layers' "$UTRAIN_DATA_DIR/runs/$RID/config.yaml"
      num_layers: 12
  $ utrain run start "$RID" >/dev/null
  $ ls -l "$UTRAIN_DATA_DIR/runs/$RID/config.yaml" | awk '{print $1}'
  -r--r--r--* (glob)
  $ ls "$UTRAIN_DATA_DIR/runs/$RID/attempt/1"
  config.yaml
  control.json
  logs
  orchestrator.log
