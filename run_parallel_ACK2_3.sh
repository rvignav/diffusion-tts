#!/bin/bash

# Total tasks: 3 scorers x 6 methods = 18 tasks
# Run across 18 GPUs (one task per GPU)

cd /code/testtime_scaling
. ./setup_testime_env.sh

NUM_GPUS=8
TOTAL_TASKS=6

for task in $(seq 0 $((TOTAL_TASKS-1))); do
    gpu=$task
    echo "Starting task $task on GPU $gpu"
    python ACK2_3.py --gpu_id $gpu --task_id $task &
done

wait
echo "All tasks completed" 