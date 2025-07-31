#!/bin/bash

# This script runs ACK2_2.py in parallel on multiple GPUs
# 6 tasks total: 3 scorers × 2 methods = 6 combinations
# Use the same environment setup as other scripts

cd /code/testtime_scaling
. ./setup_testime_env.sh

echo "Running ACK2_2.py inside SLURM job $SLURM_JOB_ID"
echo "Available GPUs: $SLURM_GPUS_ON_NODE"

echo "Starting parallel execution of ACK2_2.py on 6 GPUs..."

# Tasks for ACK2_2.py:
# 0: brightness + zero_order
# 1: brightness + eps_greedy  
# 2: compressibility + zero_order
# 3: compressibility + eps_greedy
# 4: imagenet + zero_order
# 5: imagenet + eps_greedy

for task_id in 0 1 2 3 4 5; do
    echo "Starting ACK2_2 task $task_id on GPU $task_id"
    python3 ACK2_2.py --gpu_id $task_id --task_id $task_id &
    sleep 5  # Small delay to avoid simultaneous model downloads
done

echo "All ACK2_2 tasks started. Waiting for completion..."
wait
echo "All ACK2_2 tasks completed!" 