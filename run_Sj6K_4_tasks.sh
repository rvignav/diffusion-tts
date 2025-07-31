#!/bin/bash
set -x
source ./setup_testime_env.sh

# Run tasks distributed across 8 GPUs
# Each GPU handles multiple tasks in sequence
NUM_GPUS=8
TOTAL_TASKS=6

for gpu in $(seq 0 $((NUM_GPUS-1))); do
    for task in $(seq $gpu $NUM_GPUS $((TOTAL_TASKS-1))); do
        echo "Starting task $task on GPU $gpu"
        python Sj6K_4.py --gpu_id $gpu --task_id $task &
    done
done

wait
echo "All tasks completed"
