  $ . "$TESTDIR/setup.sh"
  $ utrain run create --name hello --image utrain-fake --gpu none --print-id
  [0-9a-f]{32} (re)
  $ utrain run list
  ID           NAME   IMAGE        GPU   STATUS       ATTEMPT  PHASE  CREATED
  [0-9a-f]{8}\.\.\.  hello  utrain-fake  none  configuring  --       --     [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} (re)
  $ utrain run list --json
  .* (re)
