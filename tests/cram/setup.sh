# tests/cram/setup.sh — sourced at the top of every .t file
[ -z "$UTRAIN_CRAM_BIN" ] && UTRAIN_CRAM_BIN="utrain"

export UTRAIN_DATA_DIR=$(mktemp -d /tmp/utrain-cram.XXXXXX)
export UTRAIN_COMPUTE_FIXTURE="$TESTDIR/fixtures/compute.json"

# Isolated per-test enroot store so `utrain image add`/`remove` and running
# containers never touch other tests or the developer's global enroot store.
# Tests that need an image add it themselves via `utrain image add
# "$UTRAIN_TEST_IMAGE_URL"` (the URL comes from the pytest fixture). This dir is
# torn down with UTRAIN_DATA_DIR by the trap below.
export ENROOT_DATA_PATH="$UTRAIN_DATA_DIR/enroot/data"
export ENROOT_RUNTIME_PATH="$UTRAIN_DATA_DIR/enroot/runtime"
export ENROOT_CACHE_PATH="$UTRAIN_DATA_DIR/enroot/cache"
export ENROOT_TEMP_PATH="$UTRAIN_DATA_DIR/enroot/temp"
mkdir -p "$ENROOT_DATA_PATH" "$ENROOT_RUNTIME_PATH" "$ENROOT_CACHE_PATH" "$ENROOT_TEMP_PATH"

trap 'rm -rf "$UTRAIN_DATA_DIR"' EXIT
