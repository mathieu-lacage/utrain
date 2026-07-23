  $ . "$TESTDIR/setup.sh"
  $ RID=$(utrain run create --name hello --image utrain-fake --gpu none --print-id)
  $ utrain run start "$RID"
  ID           NAME   IMAGE        GPU   STATUS   ATTEMPT  PHASE      CREATED
  [0-9a-f]{8}\.\.\.  hello  utrain-fake  none  running  1        tokenizer  [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} (re)
  $ utrain phase list "$RID"
  PHASE      ORDER  STATUS   STARTED           ENDED
  tokenizer  0      running  [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}  -- (re)
  pretrain   1      pending  --                --
  $ utrain run show "$RID" --wait
  id:       [0-9a-f]{32} (re)
  name:     hello
  image:    utrain-fake
  gpu:      none
  status:   done
  attempts: 1 (latest: done)
  created:  [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} (re)
  
  PHASE      ORDER  STATUS  STARTED           ENDED
  tokenizer  0      done    [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}  [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} (re)
  pretrain   1      done    [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}  [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} (re)
  
  config: .* (re)
  logs:   .* (re)
  $ utrain run list
  ID           NAME   IMAGE        GPU   STATUS  ATTEMPT  PHASE  CREATED
  [0-9a-f]{8}\.\.\.  hello  utrain-fake  none  done    1        --     [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} (re)
