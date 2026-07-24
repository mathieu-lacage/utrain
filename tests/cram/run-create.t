  $ . "$TESTDIR/setup.sh"
  $ utrain run create --name hello --image utrain-fake --compute cpu --print-id
  [0-9a-f]{32} (re)
  $ utrain run list
  ID\s+NAME\s+IMAGE\s+COMPUTE\s+STATUS\s+ATTEMPT\s+PHASE\s+CREATED (re)
  [0-9a-f]+\s+hello  utrain-fake  cpu      configuring  --       --     [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} (re)
  $ utrain run list --json
  .* (re)
