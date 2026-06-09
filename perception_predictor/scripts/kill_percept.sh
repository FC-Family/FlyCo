#!/usr/bin/env bash
set -euo pipefail

kill_tmux_session() {
  local session="$1"
  if tmux has-session -t "$session" 2>/dev/null; then
    tmux kill-session -t "$session"
    echo "Killed tmux session: $session"
  fi
}

command -v tmux >/dev/null 2>&1 || {
  echo "tmux is not available; nothing to kill through tmux." >&2
  exit 1
}

if [[ $# -gt 0 ]]; then
  for session in "$@"; do
    kill_tmux_session "$session"
  done
else
  kill_tmux_session ros_commands_refactor
  kill_tmux_session perception_stack
fi

echo "FlyCo perception tmux sessions are stopped."
