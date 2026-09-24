A run is pinned to the image *id* it was created against, not the name. The
name is only what the user picked; if it is later re-tagged to different
content, the run must keep seeing -- and running -- the image it was created
with, not whatever the name points at now.

  $ . "$TESTDIR/setup.sh"
  $ utrain image add "$UTRAIN_TEST_IMAGE_URL" >/dev/null 2>&1
  $ RID=$(utrain run create --name hello --image utrain-fake --compute cpu --print-id)
  $ utrain run show "$RID" | sed -n '3p'
  image:    utrain-fake \([0-9a-f]{12}\) (re)

  $ utrain run start "$RID" >/dev/null
  $ utrain run show "$RID" --wait >/dev/null

Now move the preset name to content that is not a utrain image at all: the
bare interpreter the fake was built from, which implements no `describe`. A
run that resolved its image by name would now find nothing; the frozen one
carries on.

  $ podman tag python:3.11-slim "localhost/utrain-fake:$UTRAIN_IMAGE_TAG"

  $ utrain phase list "$RID"
  PHASE\s+ORDER\s+STATUS\s+STARTED\s+ENDED (re)
  [0-9a-f]+/1/tokenizer\s+0\s+done\s+[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}\s+[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} (re)
  [0-9a-f]+/1/pretrain\s+1\s+done\s+[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}\s+[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} (re)

  $ utrain run restart "$RID" >/dev/null
  $ utrain run show "$RID" --wait | grep -E '^status:'
  status:   done

A run created *now* sees the re-tagged content and is refused: it has no
phases to run. Only runs created before the re-tag are frozen.

  $ utrain run create --name fresh --image utrain-fake --compute cpu >/dev/null 2>&1
  [1]
