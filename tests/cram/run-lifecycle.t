  $ . "$TESTDIR/setup.sh"
  $ RID=$(utrain run create --name hello --image utrain-fake --compute cpu --print-id)
  $ utrain run start "$RID"
  ID\s+NAME\s+IMAGE\s+COMPUTE\s+STATUS\s+ATTEMPT\s+PHASE\s+CREATED (re)
  [0-9a-f]+\s+hello  utrain-fake  cpu      running  1        tokenizer  [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} (re)
  $ utrain phase list "$RID"
  PHASE      ORDER  STATUS   STARTED           ENDED
  tokenizer  0      running  [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}  -- (re)
  pretrain   1      pending  --                --
  $ utrain phase list "$RID" -q
  [0-9a-f]{32}/1/tokenizer (re)
  [0-9a-f]{32}/1/pretrain (re)
  $ utrain attempt list "$RID" -q
  [0-9a-f]{32}/1 (re)
  $ utrain run show "$RID" --wait
  id:       [0-9a-f]{32} (re)
  name:     hello
  image:    utrain-fake
  compute:  cpu
  status:   done
  attempts: 1 (latest: done)
  created:  [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} (re)
  
  PHASE      ORDER  STATUS  STARTED           ENDED
  tokenizer  0      done    [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}  [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} (re)
  pretrain   1      done    [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}  [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} (re)
  
  config: .* (re)
  logs:   .* (re)
  $ utrain run list
  ID\s+NAME\s+IMAGE\s+COMPUTE\s+STATUS\s+ATTEMPT\s+PHASE\s+CREATED (re)
  [0-9a-f]+\s+hello  utrain-fake  cpu      done    1        --     [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} (re)
  $ utrain run list -q
  [0-9a-f]{32} (re)
