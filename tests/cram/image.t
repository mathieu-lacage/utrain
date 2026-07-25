  $ . "$TESTDIR/setup.sh"
  $ utrain image add "$UTRAIN_TEST_IMAGE_URL" >/dev/null 2>&1
  $ utrain image list | grep -E 'NAME|fake' | grep -v gpu
  NAME * SIZE  RUNS (glob)
  utrain-fake .* (re)
  $ utrain image remove utrain-fake
  $ utrain image list | grep -E 'NAME|fake' | grep -v gpu
  NAME * SIZE  RUNS (glob)
  $ utrain image list -q | grep -E 'utrain-fake$'
  [1]
