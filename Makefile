.PHONY: containers/fake containers/shakespeare-char frontend check

all: presets

frontend:
	cd frontend &&  npm install && npm run build

presets: containers/shakespeare-char

containers/fake:
	podman build -t utrain-fake containers/fake

containers/shakespeare-char:
	podman build -t utrain-shakespeare-char containers/shakespeare-char

check:
	uv run pre-commit run --all-files
