The dispatcher records how an attempt ended once its orchestrator's lock is
free, whatever freed it. An orchestrator killed outright, with no chance to
write anything, is the case that proves the lock alone is enough.

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

The kernel has dropped the lock. The dispatcher notices, and finalizes: the
phase that was running when it died is recorded stopped.

  $ utrain run show "$RID" --wait | grep '^status:'
  status:   stopped
  $ utrain run list | tail -1 | grep -c stopped
  1
