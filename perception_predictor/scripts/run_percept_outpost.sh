#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

exec zsh "$SCRIPT_DIR/start_percept_refactor.zsh" \
  --min-height 0.3 \
  --cluster-tolerance 1.5 \
  --text-prompt "A compact outpost building with a red gabled roof and a small tower-like front section." \
  --ns \
  "$@"
