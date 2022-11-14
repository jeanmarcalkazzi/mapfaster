#!/usr/bin/env bash

set -e
set -x

mypy benchmark.py src/*.py src/**/*.py tests/*.py tests/**/*.py
flake8 tests src benchmark.py
black tests src benchmark.py --check --diff
isort tests src benchmark.py --check-only