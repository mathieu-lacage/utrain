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

`utrain store check` confirms every data file is a hardlink into the store
  $ utrain store check
  checked 6 data file(s) in 1 attempt(s) against 3 store file(s): ok

With -v, each data file is listed beside the store file it shares an inode with
  $ utrain store check -v
  runs/*/attempt/1/data/pretrain/common.txt  store/* (glob)
  runs/*/attempt/1/data/pretrain/common2.txt  store/* (glob)
  runs/*/attempt/1/data/pretrain/pretrain.txt  store/* (glob)
  runs/*/attempt/1/data/pretrain/tokenizer.txt  store/* (glob)
  runs/*/attempt/1/data/tokenizer/common.txt  store/* (glob)
  runs/*/attempt/1/data/tokenizer/tokenizer.txt  store/* (glob)
  checked 6 data file(s) in 1 attempt(s) against 3 store file(s): ok

The check can be scoped to a run, an attempt or a single phase
  $ utrain store check "$RID"
  checked 6 data file(s) in 1 attempt(s) against 3 store file(s): ok
  $ utrain store check "$RID/1"
  checked 6 data file(s) in 1 attempt(s) against 3 store file(s): ok
  $ utrain store check "$RID/1/tokenizer" -v
  runs/*/attempt/1/data/tokenizer/common.txt  store/* (glob)
  runs/*/attempt/1/data/tokenizer/tokenizer.txt  store/* (glob)
  checked 2 data file(s) in 1 attempt(s) against 3 store file(s): ok
  $ utrain store check "$RID/9"
  abort: attempt 9 of run '*' not found (glob)
  [1]

A tampered copy -- right content, wrong inode -- is what check exists to catch
  $ cp "$D/tokenizer/common.txt" "$D/tokenizer/common.copy"
  $ mv "$D/tokenizer/common.copy" "$D/tokenizer/common.txt"
  $ utrain store check
  runs/*/attempt/1/data/tokenizer/common.txt: not a hardlink into the store (glob)
  checked 6 data file(s) in 1 attempt(s) against 3 store file(s): 1 problem(s)
  [1]

Deleting the run drops the hardlinks; the store files become orphans until gc
  $ utrain run delete "$RID" >/dev/null
  $ utrain store check
  note: 3 orphaned store file(s) ('utrain store gc' reclaims them)
  checked 0 data file(s) in 0 attempt(s) against 3 store file(s): ok
  $ utrain store gc
  removed 3 file(s), * reclaimed (glob)
  $ ls "$UTRAIN_DATA_DIR/store" | wc -l
  0
