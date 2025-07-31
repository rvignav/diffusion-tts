#!/bin/bash

cd /code/testtime_scaling
. ./setup_testime_env.sh

# Total tasks: 1 scorer x 2 methods = 2 tasks
# Run across 2 GPUs (one task per GPU)

NUM_GPUS=8
TOTAL_TASKS=6

for task in $(seq 0 $((TOTAL_TASKS-1))); do
    gpu=$task
    echo "Starting task $task on GPU $gpu"
    python D5sF_2.py --gpu_id $gpu --task_id $task &
done

wait
echo "All tasks completed"