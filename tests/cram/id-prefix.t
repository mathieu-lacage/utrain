  $ . "$TESTDIR/setup.sh"
  $ time -p utrain image add "$UTRAIN_TEST_IMAGE_URL"
  $ time -p utrain run create --name c --image utrain-fake --compute cpu --print-id
  $ time -p utrain run create --name d --image utrain-fake --compute cpu --print-id
  $ A=$(utrain run create --name a --image utrain-fake --compute cpu --print-id)
  $ B=$(utrain run create --name b --image utrain-fake --compute cpu --print-id)
  $ time -p utrain run show "${A:0:8}"
  id:       [0-9a-f]{32} (re)
  name:     a
  image:    utrain-fake
  compute:  cpu
  status:   configuring
  attempts: 0
  created:  [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} (re)
  $ time -p utrain run show ""
  abort: id prefix '' is ambiguous .* (re)
  [1]
