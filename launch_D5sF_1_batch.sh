#!/bin/bash
set -e # Fail on error

# Load our personal variables
source ./personal_variables.sh

# SLURM job parameters
export NNODES="1"
export TIME="04:00:00"
export PARTITION="polar,polar3,polar4"
export SLURM_OUTPUT_FOLDER='./outputs_slurm'
export job_name="${MY_SLURM_ACCOUNT}-D5sF_1"
export gpus_per_node=8

# Container settings
readonly _container_image="/lustre/fsw/portfolios/nvr/users/mmardani/containers/fcn2-25.01.sqsh"
readonly _container_name="pytorch"
readonly _container_mounts="${MY_CODE_FOLDER}:/code:rw,${MY_OUTPUT_FOLDER}:/output:rw,/lustre:/lustre"

# Make sure output folder exists
if [ ! -d ${SLURM_OUTPUT_FOLDER} ]; then
    mkdir -p ${SLURM_OUTPUT_FOLDER}
fi

# Total tasks: 1 scorer x 6 methods = 6 tasks
TOTAL_TASKS=6

# Create a simple run script to avoid complex escaping
cat > run_D5sF_1_tasks.sh << 'EOF'
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
EOF

chmod +x run_D5sF_1_tasks.sh

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
     ./run_D5sF_1_tasks.sh
EOT