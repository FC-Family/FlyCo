#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run.sh -s SCENE [extra perception args...]
  scripts/run.sh --scene SCENE [extra perception args...]
  scripts/run.sh --scene=SCENE [extra perception args...]
  scripts/run.sh --SCENE [extra perception args...]

Examples:
  scripts/run.sh -s sant
  scripts/run.sh --scene sant
  scripts/run.sh --scene=windmill
  scripts/run.sh --sant

Starts the matching planner_simulator and perception_predictor scene scripts
together in a tmux session.
EOF
}

SCENE=""

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

case "$1" in
  -s)
    if [[ $# -lt 2 ]]; then
      echo "-s requires a scene name." >&2
      usage >&2
      exit 2
    fi
    SCENE="$2"
    shift 2
    ;;
  --scene)
    if [[ $# -lt 2 ]]; then
      echo "--scene requires a scene name." >&2
      usage >&2
      exit 2
    fi
    SCENE="$2"
    shift 2
    ;;
  --scene=*)
    SCENE="${1#--scene=}"
    shift
    ;;
  --help|-h)
    usage
    exit 0
    ;;
  --*)
    SCENE="${1#--}"
    shift
    ;;
  *)
    echo "Scene must be passed with -s SCENE, --scene SCENE, --scene=SCENE, or --SCENE." >&2
    usage >&2
    exit 2
    ;;
esac

if [[ -z "$SCENE" ]]; then
  echo "Scene name cannot be empty." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PLANNER_SCRIPT="$ROOT_DIR/planner_simulator/script/run_map_plan_${SCENE}.sh"
PERCEPTION_SCRIPT="$ROOT_DIR/perception_predictor/scripts/run_percept_${SCENE}.sh"
SESSION_NAME="flyco_stack_${SCENE}"

if [[ ! -x "$PLANNER_SCRIPT" ]]; then
  echo "Planner script not found or not executable: $PLANNER_SCRIPT" >&2
  exit 1
fi

if [[ ! -x "$PERCEPTION_SCRIPT" ]]; then
  echo "Perception script not found or not executable: $PERCEPTION_SCRIPT" >&2
  exit 1
fi

command -v tmux >/dev/null 2>&1 || {
  echo "tmux is required to start planner and perception together." >&2
  exit 1
}

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME" >&2
  echo "Stop it first with: $SCRIPT_DIR/kill.sh" >&2
  exit 1
fi

PERCEPTION_ARGS=()
for arg in "$@"; do
  PERCEPTION_ARGS+=("$(printf '%q' "$arg")")
done
PERCEPTION_ARGS_STR="${PERCEPTION_ARGS[*]}"
TMP_DIR="${TMPDIR:-/tmp}/flyco_stack"
mkdir -p "$TMP_DIR"
TMUX_WRAPPER="$TMP_DIR/tmux"
cat > "$TMUX_WRAPPER" <<'EOF'
#!/usr/bin/env bash
if [[ "${1:-}" == "attach" || "${1:-}" == "attach-session" ]]; then
  exit 0
fi
exec /usr/bin/tmux "$@"
EOF
chmod +x "$TMUX_WRAPPER"

tmux new-session -d -s "$SESSION_NAME" -n flyco \
  "unset TMUX; export PATH='$TMP_DIR':\$PATH; cd '$ROOT_DIR/planner_simulator/script' && zsh '$PLANNER_SCRIPT' && exec /usr/bin/tmux attach -t map_plan"
tmux split-window -h -t "$SESSION_NAME:0.0" \
  "unset TMUX; cd '$ROOT_DIR/perception_predictor/scripts' && NO_ATTACH=1 bash '$PERCEPTION_SCRIPT' $PERCEPTION_ARGS_STR && exec tmux attach -t ros_commands_refactor"
tmux select-pane -t "$SESSION_NAME:0.0" -T planner
tmux select-pane -t "$SESSION_NAME:0.1" -T perception
tmux select-layout -t "$SESSION_NAME:0" even-horizontal >/dev/null

echo "Started FlyCo stack for scene: $SCENE"
echo "  planner:    $PLANNER_SCRIPT"
echo "  perception: $PERCEPTION_SCRIPT"
echo "Controller tmux session: $SESSION_NAME"

if [[ -t 1 ]]; then
  tmux attach-session -t "$SESSION_NAME"
else
  echo "No interactive TTY detected; attach with: tmux attach-session -t $SESSION_NAME"
fi
