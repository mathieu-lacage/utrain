# tests/cram/setup.sh — sourced at the top of every .t file
[ -z "$UTRAIN_CRAM_BIN" ] && UTRAIN_CRAM_BIN="utrain"

export UTRAIN_DATA_DIR=$(mktemp -d /tmp/utrain-cram.XXXXXX)
export UTRAIN_COMPUTE_FIXTURE="$TESTDIR/fixtures/compute.json"

# Run state (DB + run dirs) is isolated per test via UTRAIN_DATA_DIR above.
# Images are not: tests share the developer's podman store, because `utrain
# image add` is a zero-copy `podman tag` of the image the pytest fixture built.
# Tests that need an image add it themselves via `utrain image add
# "$UTRAIN_TEST_IMAGE_URL"` and remove it again; the session-scoped
# _clean_preset_tags fixture in tests/conftest.py sweeps up after failures.

trap 'rm -rf "$UTRAIN_DATA_DIR"' EXIT
