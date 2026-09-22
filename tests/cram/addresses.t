Fully-qualified ids everywhere.

Every address a table's first column prints is something any address-taking
command accepts, in the one grammar ``RUN[/N][/PHASE]``: RUN is a run-id
prefix, N an attempt, and a phase given without its attempt belongs to the
run's latest attempt.

  $ . "$TESTDIR/setup.sh"
  $ utrain image add "$UTRAIN_TEST_IMAGE_URL" >/dev/null 2>&1
  $ RID=$(utrain run create --name hello --image utrain-fake --compute cpu --print-id)
  $ utrain run start "$RID" >/dev/null
  $ utrain run show "$RID" --wait >/dev/null

`run logs` over every form of the address: the run's orchestrator log, one
attempt's, and a phase's with and without its attempt spelled out.

  $ utrain run logs "$RID" | grep "starting phase 'tokenizer'"
  orchestrator: starting phase 'tokenizer'
  $ utrain run logs "$RID/1" | grep "all phases done"
  orchestrator: all phases done
  $ utrain run logs "$RID/pretrain" | grep stderr
  pretrain: note on stderr
  $ utrain run logs "$RID/1/pretrain" | grep stderr
  pretrain: note on stderr

`attempt list` narrows to one attempt with RUN/N, and `phase list` takes the
same form.

  $ utrain attempt list "$RID/1" -q
  [0-9a-f]{32}/1 (re)
  $ utrain phase list "$RID/1" -q
  [0-9a-f]{32}/1/tokenizer (re)
  [0-9a-f]{32}/1/pretrain (re)

A restart takes the phase in the address, exactly what `phase restart` does.

  $ utrain run restart "$RID/pretrain" >/dev/null
  $ utrain attempt list "$RID" -q
  [0-9a-f]{32}/2 (re)
  [0-9a-f]{32}/1 (re)

Commands that only make sense for a whole run refuse longer addresses rather
than silently using the part before the slash.

  $ utrain run show "$RID/1"
  abort: expected a run id, got '.*'.* (re)
  [1]
  $ utrain run restart "$RID/1"
  abort: restart takes RUN or RUN/PHASE, got '.*' (re)
  [1]
  $ utrain run delete "$RID/1"
  abort: expected a run id, got '.*'.* (re)
  [1]

Four components is nobody's address.

  $ utrain run logs "$RID/1/pretrain/extra"
  abort: invalid address '.*' (re)
  [1]

`attempt show` still insists on RUN/N.

  $ utrain attempt show "$RID"
  abort: expected <RUN_ID>/<N>, got '.*' (re)
  [2]

And the old per-part spellings are gone: the address replaces them.

  $ utrain run logs "$RID" --phase pretrain 2>&1 | grep -c "unrecognized arguments"
  1
  $ utrain run restart "$RID" --from-phase pretrain 2>&1 | grep -c "unrecognized arguments"
  1

`store check` keeps its stricter rule: a phase there always carries its
attempt, because the check looks in that attempt's data dir.

  $ utrain store check "$RID/pretrain"
  abort: expected attempt number, got 'pretrain'
  [1]
