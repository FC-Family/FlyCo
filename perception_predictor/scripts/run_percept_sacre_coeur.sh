#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

exec zsh "$SCRIPT_DIR/start_percept_refactor.zsh" \
  --min-height 0.6 \
  --cluster-tolerance 1.5 \
  --recall-threshold 0.81 \
  --text-prompt "An ornate cathedral with multiple domes, spires, and an irregular sculptural roofline." \
  "$@"
