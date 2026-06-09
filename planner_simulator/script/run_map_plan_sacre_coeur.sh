#!/bin/zsh

SCRIPT_DIR="$(cd "$(dirname "${(%):-%N}")" && pwd)"

LOG_DIR="${SCRIPT_DIR}/../run_log"
TIMESTAMP=$(date +'%Y-%m-%d_%H-%M-%S')
LOG_FILE="${LOG_DIR}/flyco_${TIMESTAMP}.log"

sleep 1s

if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
    echo "Log folder ${LOG_DIR} is created." | tee -a "$LOG_FILE"
else
    echo "Log folder ${LOG_DIR} already exists." | tee -a "$LOG_FILE"
fi

tmux new-session -d -s map_plan

# --- Pane 1: Horizontal split ---
tmux split-window -h -p 80 -t map_plan:0.0

# --- Pane 2: Vertical split of Pane 0 ---
tmux select-pane -t map_plan:0.0
tmux split-window -v -p 50 -t map_plan:0.0

# --- Pane 3: Extra split of Pane 0 ---
tmux select-pane -t map_plan:0.0
tmux split-window -v -p 50 -t map_plan:0.0

tmux select-pane -t map_plan:0.0
tmux send-keys 'echo "This is pane 0"' C-m

tmux select-pane -t map_plan:0.1
tmux send-keys 'echo "This is pane 1"' C-m

tmux select-pane -t map_plan:0.2
tmux send-keys 'echo "This is pane 2"' C-m

tmux select-pane -t map_plan:0.3
tmux send-keys 'echo "This is pane 3"' C-m

sleep 1s

tmux send-keys -t map_plan:0.0 "source $SCRIPT_DIR/../devel/setup.zsh && roslaunch flyco_planner_manager rviz.launch; exec zsh" C-m
sleep 1s 

tmux send-keys -t map_plan:0.1 "source $SCRIPT_DIR/../devel/setup.zsh && roslaunch airsim_ros_pkgs airsim_node.launch; exec zsh" C-m
sleep 1s 

tmux send-keys -t map_plan:0.2 "source $SCRIPT_DIR/../devel/setup.zsh && roslaunch ue4_control joy_control.launch; exec zsh" C-m
sleep 1s 

tmux send-keys -t map_plan:0.3 "source $SCRIPT_DIR/../src/3rdparty/embree-3/embree.zsh && source $SCRIPT_DIR/../devel/setup.zsh && stdbuf -oL roslaunch flyco_planner_manager flyco_planner_sacre_coeur.launch 2>&1 | tee -a $LOG_FILE; exec zsh" C-m
sleep 1s 

FCP_PID=$(pgrep -f "flyco_planner_manager/flyco_planner")

if [ -n "$FCP_PID" ]; then
    echo "map_plan PID: $FCP_PID"
    tmux send-keys -t map_plan:0.1 "expect -c \"spawn sudo cpulimit -p $FCP_PID -l 400; expect \\\"password for $USER:\\\"; send \\\"$SUDO_PASSWORD\\r\\\"; interact\"; exec zsh" C-m
else
    echo "PID not found. Failed to get map_plan PID."
fi

tmux attach -t map_plan