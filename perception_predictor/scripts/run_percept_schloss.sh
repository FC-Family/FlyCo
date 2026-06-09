#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

exec zsh "$SCRIPT_DIR/start_percept_refactor.zsh" \
  --min-height 1.0 \
  --cluster-tolerance 1.0 \
  --text-prompt "A castle-like mansion with steep gabled roofs, elongated wings, and small tower-like corner structures." \
  --ns \
  "$@"
