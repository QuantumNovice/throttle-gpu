# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-10-15

### Added
- `ThermalGuard` context manager that watches GPU temperature while a block runs.
- `run_isolated` to execute a function in a child process that is killed on overheat.
- `throttle-gpu` command with `status`, `watch`, and `run` subcommands.
- NVML temperature source with an `nvidia-smi` fallback.
- `ThermalPolicy` with validated limits and a consecutive-sample trip rule.
