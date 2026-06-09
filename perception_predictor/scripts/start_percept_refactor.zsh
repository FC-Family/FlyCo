#!/usr/bin/env zsh
set -e

SESSION_NAME="${SESSION_NAME:-ros_commands_refactor}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-}"
CONDA_ROOT="${CONDA_ROOT:-}"
ROS_SETUP="${ROS_SETUP:-}"
WORKSPACE_SETUP="${WORKSPACE_SETUP:-}"
MIN_HEIGHT="0.1"
START_RVIZ="false"
LAUNCH_PROMPT_UI="true"
CLUSTER_TOLERANCE=""
CLUSTER_MIN_X=""
CLUSTER_MAX_X=""
CLUSTER_MIN_Y=""
CLUSTER_MAX_Y=""
CLUSTER_MIN_Z=""
CLUSTER_MAX_Z=""
CLUSTER_ALGORITHM=""
TARGET_REFINE_MODE=""
TRACKING_CONFIG_FILE=""
CLUSTER_CONFIG_FILE=""
USE_CONFIG_MIN_HEIGHT="false"
RECALL_THRESHOLD=""
TEXT_PROMPT=""
BAG_PATH=""
BAG_DIR="${BAG_DIR:-${HOME}/bag}"
BAG_START="0"
BAG_RATE="1.0"
USE_CLOCK="true"
NKSR_SCALE_LAUNCH_ARG=""
KEEP_ROS_ENV="false"
DEBUG_DUMP_DIR=""
DEBUG_DUMP_ONCE="false"
RUNTIME_CONFIG_PATH=""
MESH_PRED_CKPT_PATH=""

usage() {
  cat <<USAGE
Usage:
  start_percept_refactor [-l | -s | -d] [--nb | --ns] [options]

Options:
  -d                      Default cluster height preset, min_height=0.1.
  -s                      Small/low preset, min_height=0.2.
  -l                      Large/high preset, min_height=1.5.
  --nb                    Pass nksr_scale:=0.3 to mesh_predict.
  --ns                    Pass nksr_scale:=0.2 to mesh_predict.
  --rviz                  Launch RViz in an extra pane.
  --no-ui                 Disable prompt UI inside full_stack.launch.
  --min-height V          Override min_height directly.
  --cluster-tolerance V   Override cluster_tolerance.
  --cluster-min-x V       Override cluster_min_x.
  --cluster-max-x V       Override cluster_max_x.
  --cluster-min-y V       Override cluster_min_y.
  --cluster-max-y V       Override cluster_max_y.
  --cluster-min-z V       Override cluster_min_z.
  --cluster-max-z V       Override cluster_max_z.
  --cluster-algorithm V   Override cluster algorithm.
  --target-refine-mode V  Override cluster target refine mode.
  --tracking-config-file PATH
                          Override sam_tracking tracking config.
  --cluster-config-file PATH
                          Override sam_tracking cluster config.
  --use-config-min-height Use min_height from config files instead of launch override.
  --recall-threshold V    Override mesh_predict recall_threshold.
  --text-prompt TEXT      Publish text prompt to tracking and mesh_predict.
  --bag PATH|NAME         Play a rosbag in an extra pane. If not a file, resolve from BAG_DIR.
  --bag-dir PATH          Directory used to resolve bag names. Default: \$HOME/bag
  --start SECONDS         rosbag play --start value. Default: 0.
  --rate RATE             rosbag play -r value. Default: 1.0.
  --no-clock              Do not pass --clock to rosbag play.
  --debug-dump-dir PATH   Dump next mesh_predict pointcloud inference inputs/outputs.
  --debug-dump-once       Dump only the first successful pointcloud prediction.
  --runtime-config PATH   Override mesh_predict runtime config YAML.
  --mesh-ckpt PATH        Override mesh_predict checkpoint path.
  --keep-ros-env          Preserve caller ROS_MASTER_URI/ROS_IP/ROS_HOSTNAME.
  --session NAME          Override tmux session name.
  -h | --help             Show this help.

Examples:
  start_percept_refactor -d --ns
  start_percept_refactor -d --ns --rviz
  start_percept_refactor -d --ns --bag tower --start 10 --rate 1.0 --rviz
  start_percept_refactor -d --ns --bag christ --debug-dump-dir /tmp/christ_dump --debug-dump-once
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -d) MIN_HEIGHT="0.1"; shift ;;
    -s) MIN_HEIGHT="0.2"; shift ;;
    -l) MIN_HEIGHT="1.5"; shift ;;
    --nb) NKSR_SCALE_LAUNCH_ARG="nksr_scale:=0.3"; shift ;;
    --ns) NKSR_SCALE_LAUNCH_ARG="nksr_scale:=0.2"; shift ;;
    --rviz) START_RVIZ="true"; shift ;;
    --no-ui) LAUNCH_PROMPT_UI="false"; shift ;;
    --min-height) MIN_HEIGHT="$2"; shift 2 ;;
    --cluster-tolerance) CLUSTER_TOLERANCE="$2"; shift 2 ;;
    --cluster-min-x) CLUSTER_MIN_X="$2"; shift 2 ;;
    --cluster-max-x) CLUSTER_MAX_X="$2"; shift 2 ;;
    --cluster-min-y) CLUSTER_MIN_Y="$2"; shift 2 ;;
    --cluster-max-y) CLUSTER_MAX_Y="$2"; shift 2 ;;
    --cluster-min-z) CLUSTER_MIN_Z="$2"; shift 2 ;;
    --cluster-max-z) CLUSTER_MAX_Z="$2"; shift 2 ;;
    --cluster-algorithm) CLUSTER_ALGORITHM="$2"; shift 2 ;;
    --target-refine-mode) TARGET_REFINE_MODE="$2"; shift 2 ;;
    --tracking-config-file) TRACKING_CONFIG_FILE="$2"; shift 2 ;;
    --cluster-config-file) CLUSTER_CONFIG_FILE="$2"; shift 2 ;;
    --use-config-min-height) USE_CONFIG_MIN_HEIGHT="true"; shift ;;
    --recall-threshold) RECALL_THRESHOLD="$2"; shift 2 ;;
    --text-prompt) TEXT_PROMPT="$2"; shift 2 ;;
    --bag-dir) BAG_DIR="$2"; shift 2 ;;
    --bag) BAG_PATH="$2"; shift 2 ;;
    --start|--bag-start) BAG_START="$2"; shift 2 ;;
    --rate|--bag-rate) BAG_RATE="$2"; shift 2 ;;
    --no-clock) USE_CLOCK="false"; shift ;;
    --debug-dump-dir) DEBUG_DUMP_DIR="$2"; shift 2 ;;
    --debug-dump-once) DEBUG_DUMP_ONCE="true"; shift ;;
    --runtime-config) RUNTIME_CONFIG_PATH="$2"; shift 2 ;;
    --mesh-ckpt) MESH_PRED_CKPT_PATH="$2"; shift 2 ;;
    --keep-ros-env) KEEP_ROS_ENV="true"; shift ;;
    --session) SESSION_NAME="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$ROS_SETUP" ]]; then
  if [[ -f "/opt/ros/noetic/setup.zsh" ]]; then
    ROS_SETUP="/opt/ros/noetic/setup.zsh"
  else
    ROS_SETUP="/opt/ros/noetic/setup.bash"
  fi
fi
if [[ ! -f "$ROS_SETUP" ]]; then
  echo "Missing ROS setup file: $ROS_SETUP" >&2
  exit 1
fi

if [[ -z "$WORKSPACE_SETUP" ]]; then
  if [[ -f "$ROOT_DIR/devel/setup.zsh" ]]; then
    WORKSPACE_SETUP="$ROOT_DIR/devel/setup.zsh"
  else
    WORKSPACE_SETUP="$ROOT_DIR/devel/setup.bash"
  fi
fi
if [[ ! -f "$WORKSPACE_SETUP" ]]; then
  echo "Missing $ROOT_DIR/devel/setup.zsh or setup.bash. Build the workspace first." >&2
  exit 1
fi

if [[ -z "$CONDA_ROOT" ]]; then
  for candidate in "$HOME/anaconda3" "$HOME/miniconda3" "/opt/conda"; do
    if [[ -f "$candidate/etc/profile.d/conda.sh" ]]; then
      CONDA_ROOT="$candidate"
      break
    fi
  done
fi
if [[ -z "$CONDA_ROOT" || ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
  echo "Cannot find conda.sh. Set CONDA_ROOT to your conda installation." >&2
  exit 1
fi

source "$CONDA_ROOT/etc/profile.d/conda.sh"
if [[ -z "$CONDA_ENV" ]]; then
  if conda env list | awk '{print $1}' | grep -qx "flyco"; then
    CONDA_ENV="flyco"
  elif conda env list | awk '{print $1}' | grep -qx "sam_tk"; then
    CONDA_ENV="sam_tk"
  else
    echo "Cannot find conda env flyco or sam_tk. Set CONDA_ENV explicitly." >&2
    exit 1
  fi
fi

yaml_escape_single_quotes() {
  printf "%s" "$1" | sed "s/'/''/g"
}

TEXT_PROMPT_YAML=""
if [[ -n "$TEXT_PROMPT" ]]; then
  TEXT_PROMPT_YAML="$(yaml_escape_single_quotes "$TEXT_PROMPT")"
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "会话 $SESSION_NAME 已存在。请先关闭或使用它。"
  echo "Attach: tmux attach -t $SESSION_NAME"
  echo "Kill:   tmux kill-session -t $SESSION_NAME"
  exit 1
fi

ROS_ENV_CMD=""
if [[ "$KEEP_ROS_ENV" == "true" ]]; then
  if [[ -n "${ROS_MASTER_URI:-}" ]]; then
    ROS_ENV_CMD="$ROS_ENV_CMD && export ROS_MASTER_URI=$(printf '%q' "$ROS_MASTER_URI")"
  else
    ROS_ENV_CMD="$ROS_ENV_CMD && unset ROS_MASTER_URI"
  fi
  if [[ -n "${ROS_IP:-}" ]]; then
    ROS_ENV_CMD="$ROS_ENV_CMD && export ROS_IP=$(printf '%q' "$ROS_IP")"
  else
    ROS_ENV_CMD="$ROS_ENV_CMD && unset ROS_IP"
  fi
  if [[ -n "${ROS_HOSTNAME:-}" ]]; then
    ROS_ENV_CMD="$ROS_ENV_CMD && export ROS_HOSTNAME=$(printf '%q' "$ROS_HOSTNAME")"
  else
    ROS_ENV_CMD="$ROS_ENV_CMD && unset ROS_HOSTNAME"
  fi
else
  ROS_ENV_CMD="$ROS_ENV_CMD && unset ROS_MASTER_URI && unset ROS_IP && unset ROS_HOSTNAME"
fi

BASE_CMD="cd $ROOT_DIR && source $ROS_SETUP && source $CONDA_ROOT/etc/profile.d/conda.sh && conda activate $CONDA_ENV$ROS_ENV_CMD && export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 && source $WORKSPACE_SETUP"
FULL_STACK_CMD="$BASE_CMD && roslaunch sam_tracking full_stack.launch launch_prompt_ui:=$LAUNCH_PROMPT_UI"
if [[ -n "$TEXT_PROMPT" ]]; then
  FULL_STACK_CMD="$FULL_STACK_CMD initial_text_prompt:=$(printf '%q' "$TEXT_PROMPT")"
fi
if [[ "$USE_CONFIG_MIN_HEIGHT" != "true" ]]; then
  FULL_STACK_CMD="$FULL_STACK_CMD min_height:=$MIN_HEIGHT"
fi
if [[ -n "$TRACKING_CONFIG_FILE" ]]; then
  FULL_STACK_CMD="$FULL_STACK_CMD tracking_config_file:=$TRACKING_CONFIG_FILE"
fi
if [[ -n "$CLUSTER_CONFIG_FILE" ]]; then
  FULL_STACK_CMD="$FULL_STACK_CMD cluster_config_file:=$CLUSTER_CONFIG_FILE"
fi
if [[ -n "$CLUSTER_TOLERANCE" ]]; then
  FULL_STACK_CMD="$FULL_STACK_CMD cluster_tolerance:=$CLUSTER_TOLERANCE"
fi
if [[ -n "$CLUSTER_ALGORITHM" ]]; then
  FULL_STACK_CMD="$FULL_STACK_CMD cluster_algorithm:=$CLUSTER_ALGORITHM"
fi
if [[ -n "$TARGET_REFINE_MODE" ]]; then
  FULL_STACK_CMD="$FULL_STACK_CMD target_refine_mode:=$TARGET_REFINE_MODE"
fi
if [[ -n "$CLUSTER_MIN_X" ]]; then
  FULL_STACK_CMD="$FULL_STACK_CMD cluster_min_x:=$CLUSTER_MIN_X"
fi
if [[ -n "$CLUSTER_MAX_X" ]]; then
  FULL_STACK_CMD="$FULL_STACK_CMD cluster_max_x:=$CLUSTER_MAX_X"
fi
if [[ -n "$CLUSTER_MIN_Y" ]]; then
  FULL_STACK_CMD="$FULL_STACK_CMD cluster_min_y:=$CLUSTER_MIN_Y"
fi
if [[ -n "$CLUSTER_MAX_Y" ]]; then
  FULL_STACK_CMD="$FULL_STACK_CMD cluster_max_y:=$CLUSTER_MAX_Y"
fi
if [[ -n "$CLUSTER_MIN_Z" ]]; then
  FULL_STACK_CMD="$FULL_STACK_CMD cluster_min_z:=$CLUSTER_MIN_Z"
fi
if [[ -n "$CLUSTER_MAX_Z" ]]; then
  FULL_STACK_CMD="$FULL_STACK_CMD cluster_max_z:=$CLUSTER_MAX_Z"
fi
MESH_CMD="$BASE_CMD && roslaunch mesh_predict mesh_predict.launch"
if [[ -n "$NKSR_SCALE_LAUNCH_ARG" ]]; then
  MESH_CMD="$MESH_CMD $NKSR_SCALE_LAUNCH_ARG"
fi
if [[ -n "$RECALL_THRESHOLD" ]]; then
  MESH_CMD="$MESH_CMD recall_threshold:=$RECALL_THRESHOLD"
fi
if [[ -n "$RUNTIME_CONFIG_PATH" ]]; then
  MESH_CMD="$MESH_CMD runtime_config_path:=$(printf '%q' "$RUNTIME_CONFIG_PATH")"
fi
if [[ -n "$MESH_PRED_CKPT_PATH" ]]; then
  MESH_CMD="$MESH_CMD mesh_pred_ckpt_path:=$(printf '%q' "$MESH_PRED_CKPT_PATH")"
fi
if [[ -n "$DEBUG_DUMP_DIR" ]]; then
  MESH_CMD="$MESH_CMD debug_dump_dir:=$(printf '%q' "$DEBUG_DUMP_DIR")"
fi
if [[ "$DEBUG_DUMP_ONCE" == "true" ]]; then
  MESH_CMD="$MESH_CMD debug_dump_once:=true"
fi
RVIZ_CMD="$BASE_CMD && roslaunch sam_tracking rviz.launch"
TEXT_PROMPT_PUBLISH_CMD=""
if [[ -n "$TEXT_PROMPT_YAML" ]]; then
  TEXT_PROMPT_PUBLISH_CMD="$BASE_CMD && sleep 5 && rostopic pub -1 /query_text std_msgs/String \"data: '$TEXT_PROMPT_YAML'\" && rostopic pub -1 /text_prompt std_msgs/String \"data: '$TEXT_PROMPT_YAML'\""
fi

EXTRA_CMD=""
if [[ -n "$BAG_PATH" ]]; then
  if [[ ! -f "$BAG_PATH" ]]; then
    if [[ -f "$BAG_DIR/$BAG_PATH" ]]; then
      BAG_PATH="$BAG_DIR/$BAG_PATH"
    elif [[ -f "$BAG_DIR/$BAG_PATH.bag" ]]; then
      BAG_PATH="$BAG_DIR/$BAG_PATH.bag"
    fi
  fi
  if [[ ! -f "$BAG_PATH" ]]; then
    echo "Bag not found: $BAG_PATH" >&2
    exit 1
  fi
  BAG_CMD="$BASE_CMD && sleep 5"
  if [[ -n "$TEXT_PROMPT_PUBLISH_CMD" ]]; then
    BAG_CMD="$BAG_CMD && rostopic pub -1 /query_text std_msgs/String \"data: '$TEXT_PROMPT_YAML'\" && rostopic pub -1 /text_prompt std_msgs/String \"data: '$TEXT_PROMPT_YAML'\" && sleep 1"
  fi
  BAG_CMD="$BAG_CMD && rosbag play $BAG_PATH --start $BAG_START -r $BAG_RATE"
  if [[ "$USE_CLOCK" == "true" ]]; then
    BAG_CMD="$BAG_CMD --clock"
  fi
  EXTRA_CMD="$BAG_CMD"
fi

# Start only the panes that run useful commands.
tmux new-session -d -s "$SESSION_NAME" -n "ROS"
tmux split-window -h -t "$SESSION_NAME"

send_zsh_command() {
  local target="$1"
  local command="$2"
  tmux send-keys -t "$target" "zsh -lc $(printf "%q" "$command")" C-m
}

if [[ -n "$TEXT_PROMPT_PUBLISH_CMD" && -z "$BAG_PATH" ]]; then
  FULL_STACK_CMD="$FULL_STACK_CMD & $TEXT_PROMPT_PUBLISH_CMD"
fi

send_zsh_command "$SESSION_NAME:0.0" "$FULL_STACK_CMD"
sleep 2
send_zsh_command "$SESSION_NAME:0.1" "$MESH_CMD"
NEXT_PANE=2
if [[ "$START_RVIZ" == "true" ]]; then
  tmux split-window -v -t "$SESSION_NAME:0.1"
  sleep 1
  send_zsh_command "$SESSION_NAME:0.$NEXT_PANE" "$RVIZ_CMD"
  NEXT_PANE=$((NEXT_PANE + 1))
fi
if [[ -n "$EXTRA_CMD" ]]; then
  tmux split-window -v -t "$SESSION_NAME:0.0"
  sleep 1
  send_zsh_command "$SESSION_NAME:0.$NEXT_PANE" "$EXTRA_CMD"
fi
tmux select-layout -t "$SESSION_NAME" tiled

if [[ "${NO_ATTACH:-0}" != "1" && -t 1 ]]; then
  tmux attach-session -t "$SESSION_NAME"
else
  echo "No interactive TTY detected; session left running."
  echo "Attach with: tmux attach-session -t $SESSION_NAME"
fi

echo "所有命令已经在 tmux 中启动。"
echo "SESSION_NAME: $SESSION_NAME"
echo "min_height: $MIN_HEIGHT"
if [[ -n "$NKSR_SCALE_LAUNCH_ARG" ]]; then
  echo "NKSR_SCALE 参数: $NKSR_SCALE_LAUNCH_ARG"
fi
if [[ -n "$TEXT_PROMPT" ]]; then
  echo "text_prompt: $TEXT_PROMPT"
fi
if [[ -n "$BAG_PATH" ]]; then
  echo "bag: $BAG_PATH"
fi
if [[ -n "$DEBUG_DUMP_DIR" ]]; then
  echo "debug_dump_dir: $DEBUG_DUMP_DIR"
  echo "debug_dump_once: $DEBUG_DUMP_ONCE"
fi
if [[ -n "$RUNTIME_CONFIG_PATH" ]]; then
  echo "runtime_config_path: $RUNTIME_CONFIG_PATH"
fi
if [[ -n "$MESH_PRED_CKPT_PATH" ]]; then
  echo "mesh_pred_ckpt_path: $MESH_PRED_CKPT_PATH"
fi
