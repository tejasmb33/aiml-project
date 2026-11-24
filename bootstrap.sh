#!/usr/bin/env bash
# Run from any directory. Downloads packages/ImageNet weights, never volunteer data.
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install 'pip==24.3.1' 'setuptools==75.8.0' 'wheel==0.44.0' 'cmake==3.30.5'
# CMake must be available before building dlib. GNU/Clang C++ compiler is required.
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
python -m pip install --no-build-isolation -r requirements-lock.txt
python -m pytest -q
printf '\nInstalled and tested. Add consented data using README.md, then run: make run\n'
