  $ . "$TESTDIR/setup.sh"
  $ utrain image add "$UTRAIN_TEST_IMAGE_URL" >/dev/null 2>&1
  $ RID=$(utrain run create --name hello --image utrain-fake --compute cpu --print-id)
  $ utrain run start "$RID" >/dev/null
  $ utrain run show "$RID" --wait >/dev/null
  $ utrain attempt show "$RID/1"
  run:        [0-9a-f]{32} \(hello\) (re)
  attempt:    1
  from_phase: --
  status:     done
  started:    [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} (re)
  ended:      [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} (re)
  
  PHASE      ORDER  STATUS  STARTED           ENDED
  tokenizer  0      done    [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}  [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} (re)
  pretrain   1      done    [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}  [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} (re)
  
  logs: .* (re)
  data: .* (re)
  $ utrain run restart "$RID" --from-phase pretrain >/dev/null
  $ utrain run show "$RID" --wait >/dev/null
  $ utrain attempt show "$RID/2"
  run:        [0-9a-f]{32} \(hello\) (re)
  attempt:    2
  from_phase: pretrain
  status:     done
  started:    [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} (re)
  ended:      [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} (re)
  
  PHASE\s+ORDER\s+STATUS\s+STARTED\s+ENDED (re)
  pretrain\s+1\s+done\s+[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}\s+[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} (re)
  
  logs: .* (re)
  data: .* (re)
  $ utrain attempt show "$RID"
  abort: expected <RUN_ID>/<N>, got '[0-9a-f]{32}' (re)
  [2]
