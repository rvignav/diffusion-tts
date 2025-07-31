#!/bin/bash

# Setup script for testime_env in SLURM jobs
# Single source of truth for all environment setup

# Setup package installation to Lustre space to avoid quota issues
PYTHON_PACKAGES_DIR="/lustre/fs12/portfolios/nvr/projects/nvr_lpr_compgenai/users/mmardani/python_packages"
mkdir -p $PYTHON_PACKAGES_DIR

echo "Setting up Python environment in Lustre space (no quota limits)..."

# Set cache directories to avoid quota issues
export DNNLIB_CACHE_DIR="/lustre/fs12/portfolios/nvr/projects/nvr_lpr_compgenai/users/mmardani/.cache"
export WANDB_CACHE_DIR="/lustre/fs12/portfolios/nvr/projects/nvr_lpr_compgenai/users/mmardani/wandb"
export HF_HOME="/lustre/fs12/portfolios/nvr/projects/nvr_lpr_compgenai/users/mmardani/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p ${DNNLIB_CACHE_DIR}
mkdir -p ${WANDB_CACHE_DIR}
mkdir -p ${HF_HOME}

# Setup Python path to find packages in Lustre space
export PYTHONPATH="$PYTHON_PACKAGES_DIR:$PYTHONPATH"

# Check if packages are already installed (check multiple key packages)
if [ ! -d "$PYTHON_PACKAGES_DIR/transformers" ] || [ ! -d "$PYTHON_PACKAGES_DIR/datasets" ] || [ ! -d "$PYTHON_PACKAGES_DIR/ImageReward" ]; then
    echo "Installing packages to $PYTHON_PACKAGES_DIR..."
    
    # Test network connectivity
    echo "Testing network connectivity..."
    python3 -c "import urllib.request; urllib.request.urlopen('https://pypi.org', timeout=10); print('PyPI connectivity OK')" || echo "PyPI connectivity failed"
    
    # Install dependencies from environment.yml (except PyTorch/PIL/matplotlib - use container's system versions)
    # Base dependencies from environment.yml  
    python3 -m pip install --target $PYTHON_PACKAGES_DIR "numpy>=1.20" "click>=8.0" "scipy>=1.7.1" psutil requests tqdm imageio
    # Skip torch==1.12.1 installation - use container's system PyTorch 2.6.0 to avoid conflicts
    
    # Pip dependencies from environment.yml (skip matplotlib/pillow - use system versions to avoid conflicts)
    python3 -m pip install --target $PYTHON_PACKAGES_DIR "imageio-ffmpeg>=0.4.3" pyspng torchmetrics datasets transformers accelerate image-reward hpsv2
else
    echo "Packages already installed in $PYTHON_PACKAGES_DIR"
fi

# Install CLIP if not already installed
if [ ! -d "$PYTHON_PACKAGES_DIR/clip" ]; then
    echo "Installing CLIP..."
    python3 -m pip install --target $PYTHON_PACKAGES_DIR git+https://github.com/openai/CLIP.git
else
    echo "CLIP already installed"
fi

echo "Package installation complete!"

echo "Environment setup complete!"
echo "Python path: $(which python3)"
echo "Packages location: $PYTHON_PACKAGES_DIR"
python3 -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())" 2>/dev/null || echo "PyTorch test will work after first run"