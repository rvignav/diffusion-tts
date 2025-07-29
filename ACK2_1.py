print("ACK2_1: low K at intermediate steps - EDM")
print("sweeps over (n,m) where we do m<20 local search iterations at first and last n denoising steps, 20 otherwise (intermediate denoising is harder)")
print("================================================")

import os
import sys
import argparse
import importlib.util
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import importlib

# =========================
# EDM Import Helper
# =========================
def import_edm():
    """Dynamically import EDM modules and scorers."""
    edm_dir = Path(__file__).parent / 'edm'
    sys.path.insert(0, str(edm_dir))
    dnnlib = importlib.import_module('dnnlib')
    dnnlib_util = importlib.import_module('dnnlib.util')
    from scorers import BrightnessScorer, CompressibilityScorer, ImageNetScorer
    return dnnlib, dnnlib_util, BrightnessScorer, CompressibilityScorer, ImageNetScorer

# =========================
# SD Import Helper
# =========================
def import_sd():
    """Dynamically import SD pipeline and scorers."""
    sd_dir = Path(__file__).parent / 'sd'
    diffusers_path = sd_dir / 'diffusers' / 'src' / 'diffusers' / '__init__.py'
    spec = importlib.util.spec_from_file_location('diffusers', str(diffusers_path.resolve()))
    sys.modules['diffusers'] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sys.modules['diffusers'])
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    sys.path.insert(0, str(sd_dir))
    from scorers import BrightnessScorer, CompressibilityScorer, CLIPScorer
    return StableDiffusionPipeline, DDIMScheduler, BrightnessScorer, CompressibilityScorer, CLIPScorer

# =========================
# Scorer Factory
# =========================
def get_scorer(backend, scorer_name, BrightnessScorer, CompressibilityScorer, CLIPScorer=None, ImageNetScorer=None):
    """Return the appropriate scorer instance for the backend and scorer name."""
    if scorer_name == 'brightness':
        return BrightnessScorer(dtype=torch.float32)
    elif scorer_name == 'compressibility':
        return CompressibilityScorer(dtype=torch.float32)
    elif scorer_name == 'clip' and backend == 'sd':
        return CLIPScorer(dtype=torch.float32)
    elif scorer_name == 'imagenet' and backend == 'edm':
        return ImageNetScorer(dtype=torch.float32)
    else:
        raise ValueError(f"Unknown or invalid scorer '{scorer_name}' for backend '{backend}'")

# =========================
# Main Logic
# =========================
def main():
    backend = 'edm'
    scorers = ['brightness', 'compressibility', 'imagenet']
    methods = ['zero_order', 'eps_greedy']
    N = 4
    lambda_param = 0.15
    eps = 0.4
    B = 2
    S = 8
    seed = 0
    device = 'cuda'

    scores = {}

    for curr_scorer in scorers:
        for method in methods:
            for n in list(range(0, 9)):
                for m in [0, 1, 2, 3, 5, 7, 10, 13, 17, 19]:
                    dnnlib, dnnlib_util, BrightnessScorer, CompressibilityScorer, ImageNetScorer = import_edm()
                    scorer = get_scorer('edm', curr_scorer, BrightnessScorer, CompressibilityScorer, ImageNetScorer=ImageNetScorer)

                    # EDM defaults
                    model_root = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained'
                    network_pkl = f'{model_root}/edm-imagenet-64x64-cond-adm.pkl'
                    gridw = gridh = 6
                    latents = torch.randn([gridw * gridh, 3, 64, 64])
                    class_labels = torch.eye(1000)[torch.randint(1000, size=[gridw * gridh])]
                    device = torch.device(device)
                    num_steps = 18

                    # EDM method mapping
                    from edm.main import SamplingMethod, generate_image_grid
                    method_map = {
                        'naive': SamplingMethod.NAIVE,
                        'rejection': SamplingMethod.REJECTION_SAMPLING,
                        'beam': SamplingMethod.BEAM_SEARCH,
                        'mcts': SamplingMethod.MCTS,
                        'zero_order': SamplingMethod.ZERO_ORDER,
                        'eps_greedy': SamplingMethod.EPS_GREEDY,
                    }
                    if method not in method_map:
                        raise ValueError(f"Unknown method: {method}")
                    sampling_method = method_map[method]
                    sampling_params = {'scorer': scorer}

                    # Add master params if relevant for method
                    if method in ['rejection', 'zero_order', 'eps_greedy', 'beam', 'mcts']:
                        if N is not None:
                            sampling_params['N'] = N
                        sampling_params['K'] = (n, m)
                        if lambda_param is not None:
                            sampling_params['lambda_param'] = lambda_param
                        if eps is not None:
                            sampling_params['eps'] = eps
                        if B is not None:
                            sampling_params['B'] = B
                        if S is not None:
                            sampling_params['S'] = S

                    outname = f"edm_{method}_{curr_scorer}.png"
                    score = generate_image_grid(
                        network_pkl,
                        outname,
                        latents,
                        class_labels,
                        seed=seed,
                        gridw=gridw,
                        gridh=gridh,
                        device=device,
                        num_steps=num_steps,
                        S_churn=40,
                        S_min=0.05,
                        S_max=50,
                        S_noise=1.003,
                        sampling_method=sampling_method,
                        sampling_params=sampling_params,
                    )
                    scores[f'{curr_scorer}_{method}_n{n}_m{m}'] = score

    current_filename = os.path.basename(__file__).split('.')[0]
    with open(f'{current_filename}.txt', 'a') as f:
        for key, value in scores.items():
            f.write(f'{key}: {value}\n')

if __name__ == '__main__':
    main()