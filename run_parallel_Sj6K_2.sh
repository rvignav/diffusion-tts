#!/bin/bash

cd /code/testtime_scaling
. ./setup_testime_env.sh

# Total tasks: 1 scorer x 2 methods = 2 tasks
# Run across 8 GPUs

NUM_GPUS=8
TOTAL_TASKS=2

for gpu in $(seq 0 $((NUM_GPUS-1))); do
    for task in $(seq $gpu $NUM_GPUS $((TOTAL_TASKS-1))); do
        echo "Starting task $task on GPU $gpu"
        python Sj6K_2.py --gpu_id $gpu --task_id $task &
    done
done

wait
echo "All tasks completed"