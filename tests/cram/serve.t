  $ . "$TESTDIR/setup.sh"
  $ utrain image add "$UTRAIN_TEST_IMAGE_URL" >/dev/null 2>&1
  $ RID=$(utrain run create --name chat --image utrain-fake --compute cpu --print-id)
  $ utrain run start "$RID" >/dev/null
  $ utrain run show "$RID" --wait | grep '^status:'
  status:   done

A chat session against the finished run.

  $ printf 'hello\nagain\n/quit\n' | utrain run chat "$RID"
  serving chat (utrain-fake, phase 'pretrain')
  endpoint: http://127.0.0.1:[0-9]+/v1  \(OpenAI-compatible\) (re)
  model: data_dir=/utrain/data, id=fake, owned_by=utrain, phase=pretrain
  [fake model] turn 1: 'hello'
  [fake model] turn 2: 'again'


/reset drops the history of the client history, so the next turn is turn 1 again.

  $ printf 'one\n/reset\ntwo\n' | utrain run chat "$RID" | grep 'fake model'
  [fake model] turn 1: 'one'
  [fake model] turn 1: 'two'

Ctrl-D (an empty stdin) ends the session as cleanly as /quit

  $ printf '' | utrain run chat "$RID" >/dev/null 2>&1 && echo "clean exit"
  clean exit

Naming a phase talks to that phase's snapshot instead of the newest one, and
the name reaches the container: it reports back the `--phase` it was started
with, which is how it knows which model the mounted dir is meant to hold.

  $ printf 'hello\n/quit\n' | utrain run chat "$RID/pretrain" | head -3 | grep -v endpoint
  serving chat (utrain-fake, phase 'pretrain')
  model: data_dir=/utrain/data, id=fake, owned_by=utrain, phase=pretrain

A phase the image does not serve is refused by name, even though it ran: only
`pretrain` leaves a model behind.

  $ utrain run chat "$RID/tokenizer"
  abort: image 'utrain-fake' does not serve phase 'tokenizer'; it serves: pretrain
  [1]

A run that has not been started cannot be chatted with.

  $ NEW=$(utrain run create --name unstarted --image utrain-fake --compute cpu --print-id)
  $ utrain run chat "$NEW"
  abort: run .* is still configuring; chat is available once the run has finished (re)
  [1]

An image that does not implement serve

  $ utrain image add "$UTRAIN_TEST_IMAGE_URL2" >/dev/null 2>&1
  $ GID=$(utrain run create --name nogpu --image utrain-fake-gpu --compute cpu --print-id)
  $ utrain run chat "$GID"
  abort: image 'utrain-fake-gpu' does not support serve
  [1]

No container outlives a chat session. That includes the one piped to `head`
above, which closes the pipe mid-banner: the BrokenPipeError still has to reach
the shutdown, and the shutdown removes the container by the id podman wrote at
startup rather than only signalling the client. Matched by the label utrain
puts on its containers: a container started by image id -- what runs freeze at
creation -- is reported by `podman ps` under whichever tag of that image
podman picks, not necessarily this test's.
  $ podman ps --filter "label=utrain.image-tag=$UTRAIN_IMAGE_TAG" --format '{{.ID}}' | wc -l
  0

A plain OpenAI-compatible server, curl style.

  $ SERVEDIR="$UTRAIN_DATA_DIR/by-hand"
  $ mkdir -p "$SERVEDIR"
  $ podman run -d --network=host --name "utrain-cram-serve-$$" \
  >   --security-opt=label=disable \
  >   --rm \
  >   -v "$UTRAIN_DATA_DIR/runs/$RID/attempt/1/data/pretrain:/utrain/data:ro" \
  >   -v "$SERVEDIR:/utrain/serve" \
  >   "localhost/utrain-fake:$UTRAIN_IMAGE_TAG" --utrain-root /utrain serve --port 0 >/dev/null

`--phase` is omitted here on purpose: an image with a single servable phase may
default it, so a hand-run serve needs nothing but the mount and a port.

  $ for _ in $(seq 200); do [ -s "$SERVEDIR/port.json" ] && break; sleep 0.1; done
  $ [ -s "$SERVEDIR/port.json" ] || podman logs "utrain-cram-serve-$$"
  $ PORT=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["port"])' < "$SERVEDIR/port.json")
  $ curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
  >   -H 'Content-Type: application/json' \
  >   -d '{"messages":[{"role":"user","content":"hi"}]}' \
  >   | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
  [fake model] turn 1: 'hi'
  $ podman rm -f "utrain-cram-serve-$$" >/dev/null 2>&1
