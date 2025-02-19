#!/bin/bash
set -e

SESSION_NAME="dsba_tmux"
IMAGE_NAME="nyc326/dsba_pretrain"
TAG="nlp"
CONTAINER_NAME="nlp"

echo "Starting tmux session $SESSION_NAME"
tmux new-session -d -s $SESSION_NAME "docker run --gpus all -dit -v ~/Projects/DSBA_Pretrain:/workspace --name $CONTAINER_NAME $IMAGE_NAME:$TAG /bin/bash"
tmux attach-session -t $SESSION_NAME
