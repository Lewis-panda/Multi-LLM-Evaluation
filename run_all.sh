#!/bin/bash

# 定义虚拟环境路径
ENV_PATH="/Users/lewis611036/DataAugmentation/DAenv/bin/activate"
WORK_DIR="/Users/lewis611036/DataAugmentation/Run_Models"

# 移除现有的 mysession tmux 会话
tmux kill-session -t mysession 2>/dev/null

# 创建一个 AppleScript 来打开新的终端窗口并运行 tmux 命令
osascript <<EOF
tell application "Terminal"
    do script "
    cd $WORK_DIR
    tmux new-session -d -s mysession -n run_command
    tmux send-keys -t mysession 'source $ENV_PATH; python3 run_command.py' C-m

    # 分割畫面並運行 run_gemma2.py
    tmux split-window -h -t mysession:0
    tmux send-keys -t mysession 'source $ENV_PATH; python3 run_gemma2.py' C-m

    # 分割畫面並運行 run_llama.py
    tmux split-window -v -t mysession:0.0
    tmux send-keys -t mysession 'source $ENV_PATH; python3 run_llama.py' C-m

    # 分割畫面並運行 run_qwen.py
    tmux split-window -v -t mysession:0.1
    tmux send-keys -t mysession 'source $ENV_PATH; python3 run_qwen.py' C-m

    # 分割畫面並運行 run_yi.py
    tmux split-window -h -t mysession:0.2
    tmux send-keys -t mysession 'source $ENV_PATH; python3 run_yi.py' C-m

    # 分割畫面並運行 run_deepseek.py
    tmux split-window -h -t mysession:0.3
    tmux send-keys -t mysession 'source $ENV_PATH; python3 run_deepseek.py' C-m

    # 確保所有窗格均勻排列
    tmux select-layout -t mysession tiled
    tmux -2 attach-session -t mysession
    "
    # 最大化终端窗口
#    set bounds of front window to {0, 0, 1440, 900}
    # 将终端窗口设为全屏
    tell application "System Events"
        keystroke "f" using {control down, command down}
    end tell
    
end tell
EOF
