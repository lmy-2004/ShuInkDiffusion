#!/usr/bin/env bash
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate live2diff
fi
cd "${ROOT}/Live2Diff/demo" || exit
bash start.sh
