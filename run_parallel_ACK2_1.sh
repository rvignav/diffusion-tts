#!/bin/bash

# This script should be run INSIDE a SLURM job with multiple GPUs
cd /code/testtime_scaling

# Use the same environment setup as torchrun_script.sh (sh-compatible)
. ./setup_testime_env.sh

echo "Running inside SLURM job $SLURM_JOB_ID"
echo "Available GPUs: $SLURM_GPUS_ON_NODE"

# Run ACK2_1.py in parallel on 6 GPUs
echo "Starting parallel execution on 6 GPUs..."

# Tasks:
# 0: brightness + zero_order
# 1: brightness + eps_greedy  
# 2: compressibility + zero_order
# 3: compressibility + eps_greedy
# 4: imagenet + zero_order
# 5: imagenet + eps_greedy

for task_id in 0 1 2 3 4 5; do
    echo "Starting task $task_id on GPU $task_id"
    python3 ACK2_1.py --gpu_id $task_id --task_id $task_id &
    sleep 5  # Small delay to avoid simultaneous model downloads
done

echo "All tasks started. Waiting for completion..."
wait
echo "All tasks completed!" 