#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

exec zsh "$SCRIPT_DIR/start_percept_refactor.zsh" \
  --min-height 0.6 \
  --cluster-tolerance 1.0 \
  --text-prompt "A monument-like structure with a tall rectangular tower attached to a low curved base." \
  "$@"
