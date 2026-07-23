  $ . "$TESTDIR/setup.sh"
  $ RID=$(utrain run create --name hello --image utrain-fake --gpu none --print-id)
  $ utrain run start "$RID" >/dev/null
  $ sleep 1
  $ utrain run stop "$RID"
  ID           NAME   IMAGE        GPU   STATUS   ATTEMPT  PHASE  CREATED
  [0-9a-f]{8}\.\.\.  hello  utrain-fake  none  stopped  1        --     [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} (re)
  $ utrain run show "$RID" | head -7
  id:       [0-9a-f]{32} (re)
  name:     hello
  image:    utrain-fake
  gpu:      none
  status:   stopped
  attempts: 1 (latest: stopped)
  created:  [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} (re)
