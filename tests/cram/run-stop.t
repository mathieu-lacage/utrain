  $ . "$TESTDIR/setup.sh"
  $ utrain image add "$UTRAIN_TEST_IMAGE_URL" >/dev/null 2>&1
  $ RID=$(utrain run create --name hello --image utrain-fake --compute cpu --print-id)
  $ utrain run start "$RID" >/dev/null
  $ sleep 1
  $ utrain run stop "$RID"
  ID\s+NAME\s+IMAGE\s+COMPUTE\s+STATUS\s+ATTEMPT\s+PHASE\s+CREATED (re)
  [0-9a-f]+\s+hello  utrain-fake  cpu      stopped  1        --     [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} (re)
  $ utrain run show "$RID" | head -7
  id:       [0-9a-f]{32} (re)
  name:     hello
  image:    utrain-fake
  compute:  cpu
  status:   stopped
  attempts: 1 (latest: stopped)
  created:  [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} (re)
