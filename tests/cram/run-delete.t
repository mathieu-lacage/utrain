  $ . "$TESTDIR/setup.sh"
  $ RID=$(utrain run create --name hello --image utrain-fake --gpu none --print-id)
  $ utrain run delete "$RID"
  removed run [0-9a-f]{32} (re)
  $ test -d "$UTRAIN_DATA_DIR/runs/$RID" && echo "dir still exists" || echo "dir gone"
  dir gone
  $ utrain run show "$RID"
  abort: run '*' not found (no-eol) (glob)
  [1]
