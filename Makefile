.PHONY: containers/fake

containers/fake:
	podman build -t utrain-fake containers/fake
