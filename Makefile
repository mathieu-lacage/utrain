.PHONY: all presets frontend check FORCE

all: presets

frontend:
	cd frontend &&  npm install && npm run build

presets: containers/shakespeare-char containers/fake

containers/%: FORCE
	podman build -t utrain-$*:utrain containers/$*
	echo "y" | enroot remove utrain-$*+utrain 2>/dev/null || true
	rm -f /tmp/utrain-$*+utrain.sqsh
	enroot import -o /tmp/utrain-$*+utrain.sqsh podman://utrain-$*:utrain || true
	enroot create /tmp/utrain-$*+utrain.sqsh
	rm /tmp/utrain-$*+utrain.sqsh

check:
	uv run pre-commit run --all-files
