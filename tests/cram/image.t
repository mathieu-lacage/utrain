  $ . "$TESTDIR/setup.sh"
  $ utrain image list
  NAME * SIZE  RUNS (glob)
  utrain-fake .* (re)
  utrain-shakespeare-char .* (re)
  $ utrain image remove utrain-fake && utrain image list
  NAME * SIZE  RUNS (glob)
  utrain-shakespeare-char .* (re)
