  $ . "$TESTDIR/setup.sh"

This walks docs/building-a-container.md end to end. The container script and
the Containerfile are extracted from the doc itself rather than copied here,
so the tutorial cannot rot without this test failing.
  $ DOC="$TESTDIR/../../docs/building-a-container.md"
  $ awk '/^```python$/{f=1;next} /^```$/{if(f)exit} f' "$DOC" > demo_container.py
  $ awk '/^```dockerfile$/{f=1;next} /^```$/{if(f)exit} f' "$DOC" > Containerfile
  $ test -s demo_container.py && test -s Containerfile

Step 2: run the phase before there is any container. The doc says
`pip install pyyaml wandb`; here an ephemeral uv environment stands in for
the reader's shell, so the test does not depend on utrain's own venv having
wandb.
  $ WANDB_MODE=offline uv run --quiet --no-project --python 3.11 \
  >   --with pyyaml --with wandb python demo_container.py run --phase train >/dev/null 2>&1
  $ cat run/data/model.txt
  trained for 50 steps

Steps 3-4: build it and check `describe` parses.
  $ podman build -t localhost/utrain-demo:utrain -f Containerfile . >/dev/null 2>&1
  $ podman run --rm localhost/utrain-demo:utrain describe | \
  >   python -c 'import json,sys; d=json.load(sys.stdin); print(d["name"], d["phase_order"])'
  demo ['train']

Step 5: register it. `podman://` tags the local image as a preset instead of
pulling.
  $ utrain image add podman://localhost/utrain-demo:utrain >/dev/null
  $ utrain image list -q
  utrain-demo

The one config field from `describe` lands in the run config, next to the
run_id and compute utrain injects.
  $ RID=$(utrain run create --name demo --image utrain-demo --compute cpu --print-id)
  $ cat "$UTRAIN_DATA_DIR/runs/$RID/config.yaml"
  run_id: [0-9a-f]{32} (re)
  compute: cpu
  phases:
    train:
      steps: 50

Step 6: start it and wait for it to finish.
  $ utrain run start "$RID" >/dev/null
  $ utrain run show "$RID" --wait
  id:       [0-9a-f]{32} (re)
  name:     demo
  image:    utrain-demo
  compute:  cpu
  status:   done
  attempts: 1 (latest: done)
  created:  [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} (re)
  
  PHASE  ORDER  STATUS  STARTED           ENDED
  train  0      done    [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}  [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} (re)
  
  config: .* (re)
  logs:   .* (re)

What the phase wrote to <root>/data lands in the phase's data dir on the host.
  $ cat "$UTRAIN_DATA_DIR/runs/$RID/attempt/1/data/train/model.txt"
  trained for 50 steps

Step 7: the metrics utrain captured through the wandb shim, filed under the
wandb project name the container passed to `wandb.init`.
  $ METRICS="$UTRAIN_DATA_DIR/runs/$RID/attempt/1/wandb/train"
  $ naw metrics "$METRICS"/*.rtsdb
  metric      type
  ----------  -------
  _step       INT64
  _timestamp  FLOAT64
  loss        FLOAT64
  $ naw plot -y loss --lines "$METRICS"/*.rtsdb >/dev/null

All 50 steps are there and loss decreases across them, as the doc promises.
  $ naw watch -t 1 "$METRICS"/*.rtsdb | awk '
  >   {for (i = 1; i <= NF; i++) if ($i ~ /^loss=/) {
  >      split($i, a, "="); if (NR == 1) first = a[2]; last = a[2] }}
  >   END {printf "steps=%d decreasing=%s\n", NR, (first > last ? "yes" : "no")}'
  steps=50 decreasing=yes
