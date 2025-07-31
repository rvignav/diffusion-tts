#!/bin/bash
export TIME="04:00:00"
export PARTITION="interactive"
export NNODES=1

# Debug print
set -x

# Load our personal variables
source ./personal_variables.sh

# Make sure out folder is defined
if [ ! -d  ${MY_OUTPUT_FOLDER} ]; then
    mkdir -p  ${MY_OUTPUT_FOLDER}
fi

export job_name="${MY_SLURM_ACCOUNT}-latentai:testtime_scaling"

gpus_per_node=8  #1
if [[ -n "$SLURM_GPUS" ]]; then
    gpus_per_node="$SLURM_GPUS"
fi

#readonly _container_image="gitlab-master.nvidia.com/dl/dgx/pytorch:23.08.06-py3-base-amd64"
#readonly _container_image="gitlab-master.nvidia.com/tkurth/makani:fcn2-dho-25.01"  #"gitlab-master.nvidia.com/tkurth/makani:fcn2-dho-25.01"    #"gitlab-master.nvidia.com/tkurth/makani:cuda-disco-24.08"
#readonly _container_image="gitlab-master.nvidia.com/tkurth/makani:fcn2-stable-25.02"
readonly _container_image="/lustre/fsw/portfolios/nvr/users/mmardani/containers/fcn2-25.01.sqsh"
readonly _container_name="pytorch"
readonly _container_mounts="${MY_CODE_FOLDER}:/code:rw,${MY_OUTPUT_FOLDER}:/output:rw,/lustre:/lustre,${MY_TMP_FOLDER}:/tmp:rw"

srun -A ${MY_SLURM_ACCOUNT} -p ${PARTITION} -N ${NNODES} -t ${TIME} \
     --job-name="${job_name}" \
     --gpus-per-node=${gpus_per_node} \
     --container-image="${_container_image}" \
     --container-name="${_container_name}" \
     --container-mounts="${_container_mounts}" \
     --container-workdir /workspace \
     --pty /bin/bash -i