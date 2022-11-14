#!/usr/bin/env bash

set -e
set -x

cd docs
sphinx-apidoc -o source ..
make html