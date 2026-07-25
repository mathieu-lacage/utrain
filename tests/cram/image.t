  $ . "$TESTDIR/setup.sh"
  $ utrain image list | grep -E 'NAME|fake' | grep -v gpu
  NAME * SIZE  RUNS (glob)
  utrain-fake .* (re)
  $ utrain image remove utrain-fake
  $ utrain image list | grep -E 'NAME|fake' | grep -v gpu
  NAME * SIZE  RUNS (glob)
  $ utrain image list -q | grep -E 'utrain-fake$'
  [1]
