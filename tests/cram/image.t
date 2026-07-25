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

Removing several images at once, scriptable via `image list -q`
  $ utrain image add "$UTRAIN_TEST_IMAGE_URL" >/dev/null 2>&1
  $ utrain image add "$UTRAIN_TEST_IMAGE_URL2" >/dev/null 2>&1
  $ utrain image list -q | grep -E 'utrain-fake'
  utrain-fake
  utrain-fake-gpu
  $ utrain image remove $(utrain image list -q | grep utrain-fake)
  $ utrain image list -q | grep -E 'utrain-fake'
  [1]

One valid and one invalid id: the valid image is removed, the invalid one aborts
  $ utrain image add "$UTRAIN_TEST_IMAGE_URL" >/dev/null 2>&1
  $ utrain image remove utrain-fake doesnotexist
  abort: image 'doesnotexist' not found
  [1]
  $ utrain image list -q | grep -E 'utrain-fake$'
  [1]
