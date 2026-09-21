  $ . "$TESTDIR/setup.sh"
  $ utrain image add "$UTRAIN_TEST_IMAGE_URL" >/dev/null 2>&1
  $ RID=$(utrain run create --name hello --image utrain-fake --compute cpu --print-id)
  $ utrain run start "$RID" >/dev/null
  $ utrain run show "$RID" --wait >/dev/null
  $ utrain phase restart "$RID/pretrain"
  ID\s+NAME\s+IMAGE\s+COMPUTE\s+STATUS\s+ATTEMPT\s+PHASE\s+CREATED (re)
  [0-9a-f]+\s+hello  utrain-fake  cpu      running  2        pretrain  [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} (re)
  $ utrain attempt list "$RID"
  ATTEMPT\s+FROM_PHASE\s+STATUS\s+STARTED\s+ENDED (re)
  [0-9a-f]+/2\s+pretrain\s+running\s+[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}\s+-- (re)
  [0-9a-f]+/1\s+--\s+done\s+[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}\s+[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} (re)
  $ utrain phase list "$RID"
  PHASE\s+ORDER\s+STATUS\s+STARTED\s+ENDED (re)
  [0-9a-f]+/1/tokenizer\s+0\s+done\s+[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}\s+[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} (re)
  [0-9a-f]+/2/pretrain\s+1\s+running\s+[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}\s+-- (re)
  $ utrain run show "$RID" --wait >/dev/null
  $ utrain phase show "$RID/2/pretrain" | head -3
  phase:   pretrain (Pre-Training)
  attempt: 2
  status:  done
  $ utrain phase restart "$RID/1/tokenizer"
  ID\s+NAME\s+IMAGE\s+COMPUTE\s+STATUS\s+ATTEMPT\s+PHASE\s+CREATED (re)
  [0-9a-f]+\s+hello  utrain-fake  cpu      running  3        tokenizer  [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} (re)
  $ utrain run show "$RID" --wait >/dev/null
  $ utrain attempt list "$RID" -q | head -1
  [0-9a-f]+/3 (re)
  $ utrain phase restart "$RID"
  abort: phase address must include a phase name
  [1]
  $ ls "$UTRAIN_DATA_DIR/runs/$RID/attempt"
  1
  2
  3
