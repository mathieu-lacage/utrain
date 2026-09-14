The fake container's training script imports the upstream `wandb` package
directly (and the image installs real wandb from pypi). At run time utrain
mounts a `wandb` shim ahead of it on PYTHONPATH, forwarding to naw. This proves
the runtime replacement works: naw-generated `.rtsdb` files appear even though
the container was written against wandb.

The container passes no `dir=` to `wandb.init`, so this also proves the other
half: `WANDB_DIR` alone puts the file in the phase's directory.

  $ . "$TESTDIR/setup.sh"
  $ utrain image add "$UTRAIN_TEST_IMAGE_URL" >/dev/null 2>&1
  $ RID=$(utrain run create --name wb --image utrain-fake --compute cpu --print-id)
  $ utrain run start "$RID" >/dev/null 2>&1
  $ utrain run show "$RID" --wait >/dev/null 2>&1

The run finished successfully:

  $ utrain run show "$RID" | grep -E '^status:'
  status:   done

naw wrote per-phase rtsdb files even though the container imported wandb. The
container calls a bare `wandb.init()`, so the directory under each phase is
wandb's own `default` -- a name utrain neither chose nor reads:

  $ W="$UTRAIN_DATA_DIR/runs/$RID/attempt/1/wandb"
  $ ls "$W/tokenizer/"*/*.rtsdb >/dev/null && echo ok
  ok
  $ ls "$W/pretrain/"*/*.rtsdb >/dev/null && echo ok
  ok
  $ test -s "$W"/tokenizer/*/*.rtsdb && echo nonempty
  nonempty

Each phase saw only its own metrics directory: nothing of the tokenizer phase
is reachable from where pretrain was writing.

  $ ls "$W/pretrain"
  default
