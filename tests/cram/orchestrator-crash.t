Reconciliation exists for exactly one case: an orchestrator that dies without
getting to record it. Every other transition is a write the orchestrator makes
itself, so this is the only path that proves reconciliation still works.

  $ . "$TESTDIR/setup.sh"
  $ utrain image add "$UTRAIN_TEST_IMAGE_URL" >/dev/null 2>&1
  $ RID=$(utrain run create --name crash --image utrain-fake --compute cpu --print-id)
  $ utrain run start "$RID" >/dev/null
  $ sleep 1

The run is under way and the orchestrator holds the attempt's lock.

  $ utrain run list | tail -1 | grep -c running
  1

Kill

  $ OPID=$(python3 -c 'import sqlite3, sys
  > db = sqlite3.connect(sys.argv[1])
  > print(db.execute("select pid from run_attempts where run_id = ?", sys.argv[2:]).fetchone()[0])
  > ' "$UTRAIN_DATA_DIR/utrain.db" "$RID")
  $ kill -9 -"$OPID"
  $ sleep 1

The kernel has dropped the lock. The next read notices, and finalizes.

  $ utrain run list | tail -1 | grep -c stopped
  1
  $ utrain run show "$RID" | grep '^status:'
  status:   stopped
