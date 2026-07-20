# utrain

UTrain is a python package that can be run on Linux hosts to manage local SLM training runs:
- download and install training containers
- configure your training run
- monitor the status of your training run
- check benchmark results
- manually interact with your model for simple testing


## Install

First, make sure you install [enroot](https://github.com/NVIDIA/enroot/blob/main/doc/installation.md) either from packages or from source.


Then, you can install utrain. We recommend the use of pipx:
```console
$ pipx install utrain
$ utrain start
UTrain UI available on http://localhost:7612/
```



![UTrain Demo](https://raw.githubusercontent.com/mathieu-lacage/utrain/main/docs/demo.gif)


## License

MIT — see [LICENSE](https://github.com/mathieu-lacage/utrain/blob/main/LICENSE).
