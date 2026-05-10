#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PDF="${1:-thesis.pdf}"
exec typst compile --root "$ROOT" --font-path "$ROOT/fonts" template/thesis.typ "$PDF"
