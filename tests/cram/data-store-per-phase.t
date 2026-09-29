Each phase is consolidated into the store as it succeeds, so a run stopped in
a later phase still has its finished phases deduplicated.

  $ . "$TESTDIR/setup.sh"
  $ utrain image add "$UTRAIN_TEST_IMAGE_URL" >/dev/null 2>&1
  $ RID=$(utrain run create --name stopped --image utrain-fake --compute cpu --print-id)

Hold only the pretrain phase at the gate: the tokenizer runs to completion,
then pretrain starts and waits for a gate that never opens.

  $ sed -i 's/^  pretrain:$/  pretrain:\n    gate: true/' "$UTRAIN_DATA_DIR/runs/$RID/config.yaml"
  $ utrain run start "$RID" >/dev/null
  $ until utrain phase list "$RID" | grep -Eq '/pretrain +1 +running'; do sleep 0.2; done
  $ utrain run stop "$RID" | tail -1
  [0-9a-f]+\s+stopped\s+utrain-fake\s+cpu\s+stopped\s+1\s+--\s+.* (re)

The tokenizer finished before the stop, so its data is in the store.

  $ utrain store check "$RID/1/tokenizer"
  checked 2 data file(s) in 1 attempt(s) against 2 store file(s): ok
  $ utrain store check
  checked 2 data file(s) in 1 attempt(s) against 2 store file(s): ok

The stopped phase was not consolidated, and asking about it says so.

  $ utrain store check "$RID/1/pretrain"
  abort: phase 'pretrain' of attempt 1 of run '[0-9a-f]{32}' is not done; its data is not in the store yet (re)
  [1]
