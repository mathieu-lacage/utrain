.PHONY: all presets frontend check cram FORCE

all: presets

frontend:
	cd frontend &&  npm install && npm run build

presets: containers/shakespeare-char

containers/%: FORCE
	podman build -t utrain-$*:utrain -f containers/$*/Containerfile .

cram:
	uv run pytest tests/test_cram.py

check:
	uv run pre-commit run --all-files
