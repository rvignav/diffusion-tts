#!/bin/bash
set -x
source ./setup_testime_env.sh

# Total tasks: 1 scorer x 6 methods = 6 tasks
# Run across 6 GPUs (one task per GPU)

NUM_GPUS=6
TOTAL_TASKS=6

for task in $(seq 0 $((TOTAL_TASKS-1))); do
    gpu=$task
    echo "Starting task $task on GPU $gpu"
    python D5sF_1.py --gpu_id $gpu --task_id $task &
done

wait
echo "All tasks completed"
