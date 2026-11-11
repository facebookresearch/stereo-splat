# Contributing to StereoSplat

We want to make contributing to this project as easy and transparent as
possible.

## Our Development Process

StereoSplat is research code released alongside the paper. This GitHub
repository is the source of truth: all changes, including those from the
authors, go through pull requests against `main`. We prioritize bug fixes,
reproducibility issues and documentation improvements; large new features may be
out of scope for a research release.

## Pull Requests

We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes (`pytest`).
5. Make sure your code lints (`ruff check .` and `ruff format --check .`).
6. If you haven't already, complete the Contributor License Agreement ("CLA").

## Contributor License Agreement ("CLA")

In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Meta's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue
(environment, command, config, and full error output).

Meta has a [bounty program](https://bugbounty.meta.com/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## Coding Style

* Python 3.10+, formatted with `ruff format` (88 character lines).
* Lint with `ruff check`.
* Type hints on public functions.
* Every source file starts with the copyright header used in this repository.

## License

By contributing to StereoSplat, you agree that your contributions will be
licensed under the LICENSE file in the root directory of this source tree.
