#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

exec zsh "$SCRIPT_DIR/start_percept_refactor.zsh" \
  --min-height 1.0 \
  --cluster-tolerance 0.7 \
  --text-prompt "A multi-tiered wooden pagoda with stacked curved eaves and a tapered vertical silhouette." \
  --ns \
  "$@"
