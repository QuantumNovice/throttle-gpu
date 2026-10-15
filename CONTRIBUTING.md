# Contributing to throttle-gpu

Thanks for helping. This guide covers setup, tests, and how to send changes.

## Development setup

```bash
git clone https://github.com/QuantumNovice/throttle-gpu
cd throttle-gpu
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Running checks

```bash
ruff check src tests      # lint
ruff format src tests     # format
pytest                    # tests
```

The tests use a scripted temperature source, so they run without a GPU. Please keep it
that way. New features need tests that do not require hardware.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<optional scope>): <short summary in the imperative mood>

<optional body explaining why, wrapped at about 72 columns>
```

Common types: `feat`, `fix`, `docs`, `test`, `ci`, `chore`, `style`, `refactor`.
Mark breaking changes with `!` after the type, or a `BREAKING CHANGE:` footer.

Examples:

```
feat(guard): add warn_temp_c early-warning callback
fix(isolated): kill child process when the reading fails
test(cli): cover run exit code when the command fails
```

Keep each commit focused on one logical change.

## Pull requests

1. Fork the repository and create a branch from `main`.
2. Make your change with tests and docs updates.
3. Make sure `ruff check`, `ruff format --check`, and `pytest` pass.
4. Add an entry under `[Unreleased]` in `CHANGELOG.md`.
5. Open a pull request. Describe what changed and why, and how you tested it.

## Reporting bugs

Open an issue with your GPU model, driver version, OS, Python version, the output of
`throttle-gpu status`, and a minimal reproduction.

## Code style

- Python 3.9+ compatible code. Use `from __future__ import annotations`.
- Public names are exported from `throttle_gpu/__init__.py`.
- Keep the library dependency list small. New dependencies need a good reason.

## License

By contributing, you agree that your contributions are licensed under the MIT License.
