  $ . "$TESTDIR/setup.sh"
  $ utrain image add "$UTRAIN_TEST_IMAGE_URL" >/dev/null 2>&1
  $ RID=$(utrain run create --name hello --image utrain-fake --compute cpu --print-id)
Hold the tokenizer phase open so the mid-run snapshots below don't race the
phase's own duration: with `gate: true` the fake container reports the phase
started and then blocks until the gate file appears in the attempt dir.
  $ echo "gate: true" >> "$UTRAIN_DATA_DIR/runs/$RID/config.yaml"
  $ utrain run start "$RID"
  ID\s+NAME\s+IMAGE\s+COMPUTE\s+STATUS\s+ATTEMPT\s+PHASE\s+CREATED (re)
  [0-9a-f]+\s+hello  utrain-fake  cpu      running  1        tokenizer  [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} (re)
  $ utrain phase list "$RID"
  PHASE\s+ORDER\s+STATUS\s+STARTED\s+ENDED (re)
  [0-9a-f]+/1/tokenizer\s+0\s+running\s+[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}\s+-- (re)
  [0-9a-f]+/1/pretrain\s+1\s+pending\s+--\s+-- (re)
  $ utrain phase list "$RID" -q
  [0-9a-f]{32}/1/tokenizer (re)
  [0-9a-f]{32}/1/pretrain (re)
  $ utrain attempt list "$RID" -q
  [0-9a-f]{32}/1 (re)
  $ touch "$UTRAIN_DATA_DIR/runs/$RID/attempt/1/mnt/gate"
  $ utrain run show "$RID" --wait
  id:       [0-9a-f]{32} (re)
  name:     hello
  image:    utrain-fake \([0-9a-f]{12}\) (re)
  compute:  cpu
  status:   done
  attempts: 1 (latest: done)
  created:  [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} (re)
  
  PHASE\s+ORDER\s+STATUS\s+STARTED\s+ENDED (re)
  [0-9a-f]+/1/tokenizer\s+0\s+done\s+[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}\s+[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} (re)
  [0-9a-f]+/1/pretrain\s+1\s+done\s+[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}\s+[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} (re)
  
  config: .* (re)
  logs:   .* (re)
  $ utrain run list
  ID\s+NAME\s+IMAGE\s+COMPUTE\s+STATUS\s+ATTEMPT\s+PHASE\s+CREATED (re)
  [0-9a-f]+\s+hello  utrain-fake  cpu      done    1        --     [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} (re)
  $ utrain run list -q
  [0-9a-f]{32} (re)
