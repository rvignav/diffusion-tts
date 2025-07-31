#!/bin/bash
set -e # Fail on error

# Load our personal variables
source ./personal_variables.sh

# SLURM job parameters
export NNODES="1"
export TIME="04:00:00"
export PARTITION="polar,polar3,polar4"
export SLURM_OUTPUT_FOLDER='./outputs_slurm'
export job_name="${MY_SLURM_ACCOUNT}-Sj6K_3"
export gpus_per_node=8

# Container settings
readonly _container_image="/lustre/fsw/portfolios/nvr/users/mmardani/containers/fcn2-25.01.sqsh"
readonly _container_name="pytorch"
readonly _container_mounts="${MY_CODE_FOLDER}:/code:rw,${MY_OUTPUT_FOLDER}:/output:rw,/lustre:/lustre"

# Make sure output folder exists
if [ ! -d ${SLURM_OUTPUT_FOLDER} ]; then
    mkdir -p ${SLURM_OUTPUT_FOLDER}
fi

# Total tasks: 3 scorers x 2 methods = 6 tasks
TOTAL_TASKS=6

# Create a simple run script to avoid complex escaping
cat > run_Sj6K_3_tasks.sh << 'EOF'
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
        python Sj6K_3.py --gpu_id $gpu --task_id $task &
    done
done

wait
echo "All tasks completed"
EOF

chmod +x run_Sj6K_3_tasks.sh

sbatch <<EOT
#!/bin/sh
#SBATCH --nodes=${NNODES}
#SBATCH --time=${TIME} 
#SBATCH --account=${MY_SLURM_ACCOUNT}  
#SBATCH --job-name=${job_name}
#SBATCH -p ${PARTITION}
#SBATCH --gpus-per-node=${gpus_per_node}
#SBATCH --output=${SLURM_OUTPUT_FOLDER}/%j.out

srun --container-image="${_container_image}" \\
     --container-name="${_container_name}" \\
     --container-mounts="${_container_mounts}" \\
     --container-workdir="/code/testtime_scaling" \\
     ./run_Sj6K_3_tasks.sh
EOT