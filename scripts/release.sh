#!/usr/bin/env bash
#
# Cuts a release: bumps the patch version, builds the changelog, then tags and
# pushes. The tag is what triggers the publish pipeline, so every precondition
# is checked before anything is written.
set -euo pipefail

BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$BRANCH" != main ]; then
    echo "release: must be on main, not $BRANCH" >&2
    exit 1
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "release: working tree is dirty; commit or stash first" >&2
    exit 1
fi

git fetch --quiet origin main
if [ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]; then
    echo "release: main is not in sync with origin/main" >&2
    exit 1
fi

# Resolve the next version without writing it, so the tag check below can run
# while the tree is still pristine.
VERSION=$(uv version --bump patch --dry-run --short)
if git rev-parse -q --verify "refs/tags/v$VERSION" >/dev/null; then
    echo "release: tag v$VERSION already exists" >&2
    exit 1
fi

uv version --bump patch
uv run towncrier build --version "$VERSION" --yes
cp CHANGELOG.md docs/changelog.md
git add pyproject.toml uv.lock \
        CHANGELOG.md docs/changelog.md changes/
git commit -m "release: $VERSION"
git tag "v$VERSION"
git push && git push --tags
