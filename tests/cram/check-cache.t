The fake container marks its tokenizer phase cacheable and its manifest
declares content-hashes that don't depend on the run: a second run's
tokenizer phase should be served straight from the content-addressed store
instead of launching a container.

  $ . "$TESTDIR/setup.sh"
  $ utrain image add "$UTRAIN_TEST_IMAGE_URL" >/dev/null 2>&1
  $ RID1=$(utrain run create --name first --image utrain-fake --compute cpu --print-id)
  $ utrain run start "$RID1" >/dev/null
  $ utrain run show "$RID1" --wait >/dev/null

First run has no cache to hit, so its tokenizer output populates the store:

  $ D1="$UTRAIN_DATA_DIR/runs/$RID1/attempt/1/data/tokenizer"
  $ ls "$D1"
  common.txt
  tokenizer.txt
  $ grep -c "served from cache" "$UTRAIN_DATA_DIR/runs/$RID1/attempt/1/orchestrator.log" || true
  0

Second run's tokenizer phase is a cache hit: its output matches the store
without the container ever running.

  $ RID2=$(utrain run create --name second --image utrain-fake --compute cpu --print-id)
  $ utrain run start "$RID2" >/dev/null
  $ utrain run show "$RID2" --wait >/dev/null
  $ utrain run show "$RID2" | grep -E '^status:'
  status:   done
  $ grep "served from cache" "$UTRAIN_DATA_DIR/runs/$RID2/attempt/1/orchestrator.log"
  orchestrator: phase 'tokenizer' served from cache

The cached phase's data is hardlinked from the store, same content as run 1:

  $ D2="$UTRAIN_DATA_DIR/runs/$RID2/attempt/1/data/tokenizer"
  $ ls "$D2"
  common.txt
  tokenizer.txt
  $ [ "$(stat -c %i "$D1/tokenizer.txt")" = "$(stat -c %i "$D2/tokenizer.txt")" ] && echo shared
  shared

No wandb events were written for the skipped phase (no container ran), but
utrain still reports the phase as done with sensible timestamps:

  $ ls "$UTRAIN_DATA_DIR/runs/$RID2/attempt/1/wandb/tokenizer" 2>/dev/null | wc -l
  0
  $ utrain phase show "$RID2/tokenizer" | grep -E '^status:'
  status:  done

Non-cacheable phases (pretrain) still run normally in both attempts:

  $ grep -c "served from cache" "$UTRAIN_DATA_DIR/runs/$RID1/attempt/1/orchestrator.log" "$UTRAIN_DATA_DIR/runs/$RID2/attempt/1/orchestrator.log" | awk -F: '{sum+=$2} END {print sum}'
  1
