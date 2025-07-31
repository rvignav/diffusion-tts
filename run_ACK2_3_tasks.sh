#!/bin/bash
set -x
source ./setup_testime_env.sh

# Run all tasks in parallel within the container
for task in $(seq 0 5); do
    gpu=$task
    echo "Starting task $task on GPU $gpu"
    python ACK2_3.py --gpu_id $gpu --task_id $task &
done

wait
echo "All tasks completed"
