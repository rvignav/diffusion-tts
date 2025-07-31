#!/bin/bash

# Test script to verify container environment
echo "=== Container Environment Test ==="
echo "User: $(whoami)"
echo "Working directory: $(pwd)"
echo "Python version: $(python3 --version)"
echo "Pip version: $(pip3 --version)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"

echo ""
echo "=== GPU Test ==="
python3 -c "
import subprocess
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    print('GPU available:', 'NVIDIA' in result.stdout)
    if 'NVIDIA' in result.stdout:
        print('GPU count from nvidia-smi:', result.stdout.count('GeForce') + result.stdout.count('Tesla') + result.stdout.count('RTX') + result.stdout.count('A100'))
except:
    print('nvidia-smi failed')
"

echo ""
echo "=== Installing PyTorch ==="
pip3 install torch==2.2.2+cu118 torchvision==0.17.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html

echo ""
echo "=== Testing PyTorch ==="
python3 -c "
try:
    import torch
    print('PyTorch version:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
    print('CUDA device count:', torch.cuda.device_count())
    if torch.cuda.is_available():
        print('CUDA version:', torch.version.cuda)
        for i in range(torch.cuda.device_count()):
            print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
except Exception as e:
    print('PyTorch test failed:', e)
" 