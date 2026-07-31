# tests/cram/setup.sh — sourced at the top of every .t file
[ -z "$UTRAIN_CRAM_BIN" ] && UTRAIN_CRAM_BIN="utrain"

export UTRAIN_DATA_DIR=$(mktemp -d /tmp/utrain-cram.XXXXXX)
export UTRAIN_COMPUTE_FIXTURE="$TESTDIR/fixtures/compute.json"

# Run state (DB + run dirs) is isolated per test via UTRAIN_DATA_DIR above.
# Images live in the developer's shared podman store -- `utrain image add` is a
# zero-copy `podman tag` of the image the pytest fixture built -- so they are
# isolated by tag instead: the _image_namespace fixture in tests/conftest.py
# exports a unique UTRAIN_IMAGE_TAG per test and sweeps it afterwards. Tests
# still add the images they need via `utrain image add "$UTRAIN_TEST_IMAGE_URL"`,
# but what they add, list and remove is private to the test, so they can run in
# parallel.

trap 'rm -rf "$UTRAIN_DATA_DIR"' EXIT
