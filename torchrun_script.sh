cd /code/testtime_scaling
source setup_testime_env.sh

# Set the environment variables
# Be verbose
export PET_RDZV_JOIN_TIMEOUT=1800  # Increased from 600 to 1800 seconds
export NCCL_DEBUG=INFO  # Enable NCCL debugging
export NCCL_IB_TIMEOUT=23  # Increase InfiniBand timeout
export NCCL_SOCKET_TIMEOUT=1800  # Increase socket timeout
export NCCL_ASYNC_ERROR_HANDLING=1  # Enable async error handling

# WANDB_CACHE_DIR is set by setup_testime_env.sh to Lustre space

cd /code/testtime_scaling;
pwd;

torchrun --nnodes=${NNODES} --nproc_per_node=${SLURM_GPUS_ON_NODE} \
         --rdzv_id=${SLURM_JOB_ID} --rdzv_backend=c10d  \
         --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
         "$@"; # Pass all the options to the python script
