  $ . "$TESTDIR/setup.sh"
  $ utrain image add "$UTRAIN_TEST_IMAGE_URL" >/dev/null 2>&1
  $ RID=$(utrain run create --name hello --image utrain-fake --compute cpu --print-id)
  $ utrain run start "$RID" >/dev/null
  $ utrain run show "$RID" --wait >/dev/null
  $ utrain phase show "$RID/pretrain" | head -8
  phase:   pretrain (Pre-Training)
  attempt: 1
  status:  done
  run:     * (hello) (glob)
  started: [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} (re)
  ended:   [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} (re)
  
  METRIC * (glob)
  $ utrain phase show "$RID/pretrain" | grep "^---"
  --- logs/pretrain_stdout.log (tail 20) ---
  $ utrain phase show "$RID/tokenizer" --metric vocab_coverage | head -2
  STEP  VALUE
  0     0.5
