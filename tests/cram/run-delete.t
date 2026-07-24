  $ . "$TESTDIR/setup.sh"
  $ RID=$(utrain run create --name hello --image utrain-fake --compute cpu --print-id)
  $ utrain run delete "$RID"
  removed run [0-9a-f]{32} (re)
  $ test -d "$UTRAIN_DATA_DIR/runs/$RID" && echo "dir still exists" || echo "dir gone"
  dir gone
  $ utrain run show "$RID"
  abort: run '*' not found (glob)
  [1]

Deleting multiple runs at once
  $ A=$(utrain run create --name a --image utrain-fake --compute cpu --print-id)
  $ B=$(utrain run create --name b --image utrain-fake --compute cpu --print-id)
  $ utrain run delete "$A" "$B"
  removed run [0-9a-f]{32} (re)
  removed run [0-9a-f]{32} (re)
  $ utrain run show "$A"
  abort: run '*' not found (glob)
  [1]
  $ utrain run show "$B"
  abort: run '*' not found (glob)
  [1]

Deleting multiple runs where one id does not exist reports the failure but still deletes the rest
  $ C=$(utrain run create --name c --image utrain-fake --compute cpu --print-id)
  $ utrain run delete "$C" doesnotexist
  removed run [0-9a-f]{32} (re)
  abort: run 'doesnotexist' not found
  [1]
  $ utrain run show "$C"
  abort: run '*' not found (glob)
  [1]
