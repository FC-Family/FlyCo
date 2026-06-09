#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

exec zsh "$SCRIPT_DIR/start_percept_refactor.zsh" \
  --min-height 1.0 \
  --cluster-tolerance 1.0 \
  --text-prompt "A compact red-and-white windmill with four cross-shaped blades and a narrow tapered tower." \
  --ns \
  "$@"
