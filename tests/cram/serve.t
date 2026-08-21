  $ . "$TESTDIR/setup.sh"
  $ utrain image add "$UTRAIN_TEST_IMAGE_URL" >/dev/null 2>&1
  $ RID=$(utrain run create --name chat --image utrain-fake --compute cpu --print-id)
  $ utrain run start "$RID" >/dev/null
  $ utrain run show "$RID" --wait | grep '^status:'
  status:   done

A chat session against the finished run. The port is chosen by the kernel and
published by the container to <root>/serve/port.json, so it differs every time.

  $ printf 'hello\nagain\n/quit\n' | utrain run chat "$RID"
  serving chat (utrain-fake, phase 'pretrain')
  endpoint: http://127.0.0.1:[0-9]+/v1  \(OpenAI-compatible\) (re)
  model: data_dir=/utrain/data, id=fake, owned_by=utrain
  [fake model] turn 1: 'hello'
  [fake model] turn 2: 'again'

Two things are proved above. `data_dir=/utrain/data` is the container reporting
where utrain mounted the model, so the filesystem contract reached `serve`. And
`turn 2` is the *client* having resent the history: the endpoint is stateless and
the fake counts the user messages it was handed, so a server keeping its own
state is not what produced that number.

/reset drops the client's history, so the next turn is turn 1 again.

  $ printf 'one\n/reset\ntwo\n' | utrain run chat "$RID" | grep 'fake model'
  [fake model] turn 1: 'one'
  [fake model] turn 1: 'two'

Ctrl-D (an empty stdin) ends the session as cleanly as /quit, and the server is
shut down with it rather than left listening.

  $ printf '' | utrain run chat "$RID" >/dev/null 2>&1 && echo "clean exit"
  clean exit

A run that has not produced a model yet cannot be chatted with.

  $ NEW=$(utrain run create --name unstarted --image utrain-fake --compute cpu --print-id)
  $ utrain run chat "$NEW"
  abort: run [0-9a-f]{32} has no completed phase, so there is no model to serve (re)

Nor can an image that does not implement serve: fake-gpu leaves can_serve at its
default of false, and utrain refuses before it goes looking for a model.

  $ utrain image add "$UTRAIN_TEST_IMAGE_URL2" >/dev/null 2>&1
  $ GID=$(utrain run create --name nogpu --image utrain-fake-gpu --compute cpu --print-id)
  $ utrain run chat "$GID"
  abort: image 'utrain-fake-gpu' does not support serve

The endpoint is a plain OpenAI-compatible server, so it is equally reachable
without utrain in the loop: mount a writable dir at /utrain/serve, read the port
the container publishes there, and use curl.

  $ SERVEDIR="$UTRAIN_DATA_DIR/by-hand"
  $ mkdir -p "$SERVEDIR"
  $ podman run --rm -d --network=host --name "utrain-cram-serve-$$" \
  >   -v "$UTRAIN_DATA_DIR/runs/$RID/attempt/1/data/pretrain:/utrain/data:ro" \
  >   -v "$SERVEDIR:/utrain/serve" \
  >   "localhost/utrain-fake:$UTRAIN_IMAGE_TAG" --utrain-root /utrain serve --port 0 >/dev/null
  $ for _ in $(seq 200); do [ -s "$SERVEDIR/port.json" ] && break; sleep 0.1; done
  $ PORT=$(python3 -c "import json;print(json.load(open('$SERVEDIR/port.json'))['port'])")
  $ curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
  >   -H 'Content-Type: application/json' \
  >   -d '{"messages":[{"role":"user","content":"hi"}]}' \
  >   | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
  [fake model] turn 1: 'hi'
  $ podman rm -f "utrain-cram-serve-$$" >/dev/null 2>&1
