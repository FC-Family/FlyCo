#!/usr/bin/env bash
set -euo pipefail

kill_tmux_session() {
  local session="$1"
  if tmux has-session -t "$session" 2>/dev/null; then
    tmux kill-session -t "$session"
    echo "Killed tmux session: $session"
  fi
}

kill_matching_processes() {
  local label="$1"
  shift
  local patterns=("$@")
  local pids=()
  local pattern

  for pattern in "${patterns[@]}"; do
    while IFS= read -r pid; do
      [[ -n "$pid" ]] && pids+=("$pid")
    done < <(pgrep -f "$pattern" 2>/dev/null || true)
  done

  if [[ ${#pids[@]} -eq 0 ]]; then
    return
  fi

  mapfile -t pids < <(printf '%s\n' "${pids[@]}" | sort -u)
  kill -TERM "${pids[@]}" 2>/dev/null || true
  sleep 1
  kill -KILL "${pids[@]}" 2>/dev/null || true
  echo "Killed residual $label processes: ${pids[*]}"
}

command -v tmux >/dev/null 2>&1 || {
  echo "tmux is not available; nothing to kill through tmux." >&2
  exit 1
}

for session in $(tmux list-sessions -F '#S' 2>/dev/null | awk '/^flyco_stack_/'); do
  kill_tmux_session "$session"
done

kill_tmux_session map_plan
kill_tmux_session perception_stack
kill_tmux_session ros_commands_refactor

kill_matching_processes "FlyCo" \
  "roslaunch .*flyco_planner_manager" \
  "roslaunch .*airsim_ros_pkgs" \
  "roslaunch .*ue4_control" \
  "roslaunch .*sam_tracking" \
  "roslaunch .*mesh_predict" \
  "flyco_planner_manager/flyco_planner" \
  "flyco_planner_manager/traj_server" \
  "rgb_point_mapping_node" \
  "airsim_ros_pkgs/airsim_node" \
  "px4ctrl/px4ctrl_node" \
  "ue4_control/joy_cmd" \
  "joy/joy_node" \
  "sam_tracking/prompt_ui_cv_node.py" \
  "sam_tracking/prompt_ui_node.py" \
  "sam_tracking/tracking_node.py" \
  "sam_tracking/prompt_node.py" \
  "sam_tracking_node" \
  "mesh_predict_node.py" \
  "rosmaster --core -p 11311" \
  "/rosout/rosout"

echo "FlyCo planner/perception tmux sessions are stopped."
