"""
This script allows you to generate images using either the Stable Diffusion (SD) or EDM backend,
with a choice of scorer and sampling method

Usage examples:
    python main.py --backend sd --scorer brightness --method naive --prompt "A beautiful landscape"
    python main.py --backend edm --scorer imagenet --method zero_order

Arguments:
    --backend   : 'sd' or 'edm' (required)
    --scorer    : 'brightness', 'compressibility', 'clip', or 'imagenet' (required)
    --method    : Sampling method (available: 'naive', 'rejection', 'beam', 'mcts', 'zero_order', 'eps_greedy') (default: 'naive')
    --prompt    : Prompt for SD (default: 'A beautiful landscape')
    --output    : Output filename
    --N, --lambda_, --eps, --K, --B, --S : sampling parameters (see code for defaults)
    --seed      : Random seed (default: 0)
    --device    : Device (default: 'cuda')
"""

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
    # -----------
    # CLI Arguments
    # -----------
    parser = argparse.ArgumentParser(
        description='Unified Diffusion Image Generator (EDM/SD)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--backend', type=str, choices=['edm', 'sd'], required=True, help='Backend: edm or sd')
    parser.add_argument('--scorer', type=str, choices=['brightness', 'compressibility', 'clip', 'imagenet'], required=True, help='Scorer name')
    parser.add_argument('--method', type=str, default='naive', help='Sampling method (naive, rejection, beam, mcts, zero_order, eps_greedy)')
    parser.add_argument('--prompt', type=str, default='YOUR PROMPT HERE', help='Prompt for SD')
    parser.add_argument('--output', type=str, default=None, help='Output filename (default: auto)')
    # Master params (with SD defaults)
    parser.add_argument('--N', type=int, default=4, help='Master param N')
    parser.add_argument('--lambda_', type=float, default=0.15, help='Master param lambda')
    parser.add_argument('--eps', type=float, default=0.4, help='Master param eps')
    parser.add_argument('--K', type=int, default=20, help='Master param K')
    parser.add_argument('--B', type=int, default=2, help='Master param B')
    parser.add_argument('--S', type=int, default=8, help='Master param S')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    args = parser.parse_args()

    # -----------
    # Validation
    # -----------
    if args.backend == 'sd' and args.scorer == 'imagenet':
        raise ValueError('imagenet scorer is only available for edm backend')
    if args.backend == 'edm' and args.scorer == 'clip':
        raise ValueError('clip scorer is only available for sd backend')

    # -----------
    # SD Backend
    # -----------
    if args.backend == 'sd':
        StableDiffusionPipeline, DDIMScheduler, BrightnessScorer, CompressibilityScorer, CLIPScorer = import_sd()
        scorer = get_scorer('sd', args.scorer, BrightnessScorer, CompressibilityScorer, CLIPScorer=CLIPScorer)

        model_id = "runwayml/stable-diffusion-v1-5"
        local_scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        local_pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=local_scheduler,
            torch_dtype=torch.float16,
        ).to(args.device)

        method = args.method
        MASTER_PARAMS = {
            'N': args.N,
            'lambda': args.lambda_,
            'eps': args.eps,
            'K': args.K,
            'B': args.B,
            'S': args.S,
        }

        best_result, best_score = None, float('-inf')
        for _ in range(MASTER_PARAMS['N'] if method == "rejection" else 1):
            result, score = local_pipe(
                prompt=args.prompt,
                num_inference_steps=50,
                score_function=scorer,
                method=method,
                params=MASTER_PARAMS,
            )
            if score > best_score:
                best_result, best_score = result, score

        outname = args.output or f"sd_{method}_{args.scorer}.png"
        best_result.images[0].save(outname)
        print(f"\n[SD] Saved: {outname}\nBest score: {best_score}\n")

    # -----------
    # EDM Backend
    # -----------
    elif args.backend == 'edm':
        dnnlib, dnnlib_util, BrightnessScorer, CompressibilityScorer, ImageNetScorer = import_edm()
        scorer = get_scorer('edm', args.scorer, BrightnessScorer, CompressibilityScorer, ImageNetScorer=ImageNetScorer)

        # EDM defaults
        model_root = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained'
        network_pkl = f'{model_root}/edm-imagenet-64x64-cond-adm.pkl'
        num_images = 1
        gridw = gridh = 1
        latents = torch.randn([num_images, 3, 64, 64])
        class_labels = torch.eye(1000)[torch.randint(1000, size=[num_images])]
        device = torch.device(args.device)
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
        if args.method not in method_map:
            raise ValueError(f"Unknown method: {args.method}")
        sampling_method = method_map[args.method]
        sampling_params = {'scorer': scorer}

        # Add master params if relevant for method
        if args.method in ['rejection', 'zero_order', 'eps_greedy', 'beam', 'mcts']:
            if args.N is not None:
                sampling_params['N'] = args.N
            if args.K is not None:
                sampling_params['K'] = args.K
            if args.lambda_ is not None:
                sampling_params['lambda_param'] = args.lambda_
            if args.eps is not None:
                sampling_params['eps'] = args.eps
            if args.B is not None:
                sampling_params['B'] = args.B
            if args.S is not None:
                sampling_params['S'] = args.S

        outname = args.output or f"edm_{args.method}_{args.scorer}.png"
        score = generate_image_grid(
            network_pkl,
            outname,
            latents,
            class_labels,
            seed=args.seed,
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
        current_filename = os.path.basename(__file__).split('.')[0]
        with open(f'{current_filename}.txt', 'a') as f:
            f.write(f'{outname}: {score}\n')

if __name__ == '__main__':
    main()