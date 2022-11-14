#!/usr/bin/env bash

set -e
set -x

black tests src benchmark.py
isort tests src benchmark.py