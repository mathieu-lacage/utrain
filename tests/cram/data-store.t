  $ . "$TESTDIR/setup.sh"
  $ utrain image add "$UTRAIN_TEST_IMAGE_URL" >/dev/null 2>&1
  $ RID=$(utrain run create --name hello --image utrain-fake --compute cpu --print-id)
  $ utrain run start "$RID" >/dev/null
  $ utrain run show "$RID" --wait >/dev/null
  $ D="$UTRAIN_DATA_DIR/runs/$RID/attempt/1/data"

Phase output files are visible on the host after the run completes
  $ ls "$D/tokenizer"
  common.txt
  tokenizer.txt

The next phase inherits the previous phase's files via a hardlinked copy (cp -rl)
  $ ls "$D/pretrain"
  common.txt
  common2.txt
  pretrain.txt
  tokenizer.txt
  $ cat "$D/pretrain/tokenizer.txt"
  tokenizer output
  $ [ "$(stat -c %i "$D/tokenizer/tokenizer.txt")" = "$(stat -c %i "$D/pretrain/tokenizer.txt")" ] && echo inherited
  inherited

Deduplicated content lives in the content-addressed store (3 distinct contents)
  $ ls "$UTRAIN_DATA_DIR/store" | wc -l
  3

Two differently-named files with identical content share one hardlinked store file
  $ [ "$(stat -c %i "$D/pretrain/common.txt")" = "$(stat -c %i "$D/pretrain/common2.txt")" ] && echo linked
  linked

Deleting the run drops the hardlinks; store gc then reclaims the orphaned files
  $ utrain run delete "$RID" >/dev/null
  $ utrain store gc
  removed 3 file(s), * reclaimed (glob)
  $ ls "$UTRAIN_DATA_DIR/store" | wc -l
  0
