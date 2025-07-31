#!/bin/bash

cd /code/testtime_scaling
. ./setup_testime_env.sh

# Total tasks: 3 scorers x 6 methods = 18 tasks
# Run across 8 GPUs

NUM_GPUS=8
TOTAL_TASKS=18

for gpu in $(seq 0 $((NUM_GPUS-1))); do
    for task in $(seq $gpu $NUM_GPUS $((TOTAL_TASKS-1))); do
        echo "Starting task $task on GPU $gpu"
        python ACK2_4.py --gpu_id $gpu --task_id $task &
    done
done

wait
echo "All tasks completed" 