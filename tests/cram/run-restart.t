  $ . "$TESTDIR/setup.sh"
  $ RID=$(utrain run create --name hello --image utrain-fake --gpu none --print-id)
  $ utrain run start "$RID" >/dev/null
  $ utrain run show "$RID" --wait >/dev/null
  $ utrain run restart "$RID" --from-phase pretrain
  ID\s+NAME\s+IMAGE\s+GPU\s+STATUS\s+ATTEMPT\s+PHASE\s+CREATED (re)
  [0-9a-f]+\s+hello  utrain-fake  none  running  2        pretrain  [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} (re)
  $ utrain attempt list "$RID"
  ATTEMPT  FROM_PHASE  STATUS   STARTED           ENDED
  2        pretrain    running  [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}  -- (re)
  1        --          done     [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}  [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} (re)
  $ utrain phase list "$RID"
  PHASE      ORDER  STATUS      STARTED           ENDED
  tokenizer  0      -- (att.1)  --                --
  pretrain   1      running     [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}  -- (re)
  $ utrain run show "$RID" --wait >/dev/null
  $ utrain phase read "$RID/2/pretrain" | head -3
  phase:   pretrain (Pre-Training)
  attempt: 2
  status:  done
  $ ls "$UTRAIN_DATA_DIR/runs/$RID/attempt"
  1
  2
