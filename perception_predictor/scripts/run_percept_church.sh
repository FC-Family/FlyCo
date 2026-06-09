#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

exec zsh "$SCRIPT_DIR/start_percept_refactor.zsh" \
  --min-height 1.0 \
  --cluster-tolerance 1.0 \
  --text-prompt "A stone church with a long rectangular nave, steep roof, and a prominent square bell tower." \
  --ns \
  "$@"
