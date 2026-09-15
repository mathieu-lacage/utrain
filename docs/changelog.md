# Changelog

<!-- towncrier release notes start -->

## 0.0.8 (2026-09-15)

### Added

- A recorded tour of the TUI in the documentation, driven end to end by
  `scripts/tour.py` against a nanochat-shaped fixture container.

### Changed

- Isolate wandb dir of phase A from all other phases and setup WANDB_DIR so it is found transparently by `naw`. ([#23](https://gitlab.inria.fr/mlacage/utrain/-/issues/23))

### Removed

- The `nanochat` container preset now lives in its own repository,
  [utrain-nanochat](https://gitlab.inria.fr/mlacage/utrain-nanochat), and is
  published from there: `utrain image add
  docker://registry.gitlab.inria.fr/mlacage/utrain-nanochat:latest`. It wraps an
  upstream project on its own release schedule, so it no longer rides on a utrain
  tag. Nothing about how utrain drives it has changed. ([#21](https://gitlab.inria.fr/mlacage/utrain/-/issues/21))


## 0.0.7 (2026-09-08)

### Added

- `utrain tui`, a terminal UI for browsing runs, phases and their metrics. ([#17](https://gitlab.inria.fr/mlacage/utrain/-/issues/17))
- Let a container name the plots it wants opened on for a phase, as `plots: [{x, y}]` in `describe`. The TUI shows those instead of one plot per metric, until the viewer asks for the dashboard back. ([#18](https://gitlab.inria.fr/mlacage/utrain/-/issues/18))
- Nanochat container ([#19](https://gitlab.inria.fr/mlacage/utrain/-/issues/19))
- A `nanochat` container preset wrapping [karpathy/nanochat](https://github.com/karpathy/nanochat): download, tokenizer, pretrain, SFT and RL, with chat over the trained model.

### Changed

- Pass the utrain root path to containers as a `--utrain-root` command-line argument (defaulting to `$CWD/run`) instead of hardcoding `/utrain`, so a container's phases can be run locally without building the image ([#11](https://gitlab.inria.fr/mlacage/utrain/-/issues/11))
- Simplify reconciliation because we have an orchestrator ([#14](https://gitlab.inria.fr/mlacage/utrain/-/issues/14))
- Split the shakespeare-char `tokenizer` phase into `download` and `tokenizer`. Allow caching download output. ([#15](https://gitlab.inria.fr/mlacage/utrain/-/issues/15))
- Chat is now per phase: `utrain run chat <RUN_ID> --phase <name>` talks to that phase's snapshot, and a container declares which phases are worth chatting with via `can_serve` on each phase in `describe`. The top-level `can_serve` is gone -- an image that only sets that one now serves nothing. utrain also passes `serve --phase <name>`, so a container loads the model that phase produced instead of the best one it can find, and says so when that phase saved none. Chat now requires a finished run.

### Fixed

- Add --version to the cli ([#6](https://gitlab.inria.fr/mlacage/utrain/-/issues/6))
- Allow containers to declare phases cacheable if their output data files are already in the utrain data store ([#7](https://gitlab.inria.fr/mlacage/utrain/-/issues/7))
- Tutorial and reference documentation on the container cli contract ([#8](https://gitlab.inria.fr/mlacage/utrain/-/issues/8))
- Normalize all utrain mount points under /utrain for the container and make sure only the files and directories part of the contract are visible ([#9](https://gitlab.inria.fr/mlacage/utrain/-/issues/9))
- Split shakespeare example in multiploe files to improve readability ([#10](https://gitlab.inria.fr/mlacage/utrain/-/issues/10))
- Split cli output formatting from data reading ([#13](https://gitlab.inria.fr/mlacage/utrain/-/issues/13))
- Make tutorial an e2e cram test to check that the tutorial works ([#16](https://gitlab.inria.fr/mlacage/utrain/-/issues/16))
- Stop leaving containers running. `utrain run chat ... | head -1` was killed by SIGPIPE before it could stop its `serve` container, and a `check-cache` that outlived its timeout left one behind too -- in both cases because stopping a container only signalled `podman run`, which does nothing once the client is gone. Both now remove the container by the id podman records at startup.


## 0.0.6 (2026-08-18)

### Fixed

- test all supported python versions via tox and Gitlab CI/CD ([#5](https://gitlab.inria.fr/mlacage/utrain/-/issues/5))


## 0.0.5 (2026-08-18)

### Fixed

- build and publish release packages on pypi ([#1](https://gitlab.inria.fr/mlacage/utrain/-/issues/1))
- check that files in changes/* match one of the types configured in pyproject.toml ([#2](https://gitlab.inria.fr/mlacage/utrain/-/issues/2))
- Add missing pypi metadata to pyproject.toml ([#3](https://gitlab.inria.fr/mlacage/utrain/-/issues/3))
- Enable mike version selector ([#4](https://gitlab.inria.fr/mlacage/utrain/-/issues/4))


## 0.0.4 (2026-08-18)

No significant changes.


## 0.0.3 (2026-08-18)

No significant changes.


## 0.0.2 (2026-08-17)

No significant changes.
