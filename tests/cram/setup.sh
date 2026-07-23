# tests/cram/setup.sh — sourced at the top of every .t file
[ -z "$UTRAIN_CRAM_BIN" ] && UTRAIN_CRAM_BIN="utrain"
enroot list 2>/dev/null | grep -q '^utrain-fake+utrain$' || {
  echo "no utrain-fake image; run: make presets" >&2
  exit 80   # cram treats 80 as "skip the rest of this file"
}
export UTRAIN_DATA_DIR=$(mktemp -d /tmp/utrain-cram.XXXXXX)
export UTRAIN_COMPUTE_FIXTURE="$TESTDIR/fixtures/compute.json"
trap 'rm -rf "$UTRAIN_DATA_DIR"' EXIT
