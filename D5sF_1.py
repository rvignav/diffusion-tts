print("================================================")
print("D5sF_1: SDXL + Image-Reward")
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
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DDIMScheduler
    sys.path.insert(0, str(sd_dir))
    from scorers import BrightnessScorer, CompressibilityScorer, CLIPScorer, ImageRewardScorer
    return StableDiffusionPipeline, StableDiffusionXLPipeline, DDIMScheduler, BrightnessScorer, CompressibilityScorer, CLIPScorer, ImageRewardScorer

# =========================
# Scorer Factory
# =========================
def get_scorer(backend, scorer_name, BrightnessScorer, CompressibilityScorer, CLIPScorer=None, ImageNetScorer=None, ImageRewardScorer=None):
    """Return the appropriate scorer instance for the backend and scorer name."""
    if scorer_name == 'brightness':
        return BrightnessScorer(dtype=torch.float32)
    elif scorer_name == 'compressibility':
        return CompressibilityScorer(dtype=torch.float32)
    elif scorer_name == 'clip' and backend == 'sd':
        return CLIPScorer(dtype=torch.float32)
    elif scorer_name == 'imagenet' and backend == 'edm':
        return ImageNetScorer(dtype=torch.float32)
    elif scorer_name == 'imagereward':
        return ImageRewardScorer()
    else:
        raise ValueError(f"Unknown or invalid scorer '{scorer_name}' for backend '{backend}'")

# =========================
# Main Logic
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID (0-8)')
    parser.add_argument('--task_id', type=int, default=None, help='Task ID for parallel execution')
    args = parser.parse_args()
    
    # Setup output directory
    script_name = os.path.basename(__file__).split('.')[0]
    output_dir = Path('outputs') / script_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    backend = 'sd'
    scorers = ['imagereward']
    methods = ['naive', 'rejection', 'beam', 'mcts', 'zero_order', 'eps_greedy']
    N = 4
    lambda_param = 0.15
    eps = 0.4
    K = 20
    B = 2
    S = 8
    seed = 0
    device = f'cuda:{args.gpu_id}'

    # Create all task combinations
    tasks = [(scorer, method) for scorer in scorers for method in methods]
    
    # If task_id specified, run only that task
    if args.task_id is not None:
        if args.task_id >= len(tasks):
            print(f"Invalid task_id {args.task_id}. Max is {len(tasks)-1}")
            return
        tasks = [tasks[args.task_id]]

    for curr_scorer, method in tasks:
        task_scores = {}  # Separate scores for this task
        if True:
            StableDiffusionPipeline, StableDiffusionXLPipeline, DDIMScheduler, BrightnessScorer, CompressibilityScorer, CLIPScorer, ImageRewardScorer = import_sd()
            scorer = get_scorer('sd', curr_scorer, BrightnessScorer, CompressibilityScorer, CLIPScorer=CLIPScorer, ImageRewardScorer=ImageRewardScorer)

            model_id = "stabilityai/stable-diffusion-xl-base-1.0"
            local_scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
            local_pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                scheduler=local_scheduler,
                torch_dtype=torch.float16,
            ).to(device)

            method = method
            num_inference_steps = 200
            MASTER_PARAMS = {
                'N': N,
                'lambda': lambda_param,
                'eps': eps,
                'K': K,
                'B': B,
                'S': S,
            }
            best_scores = []
            prompts = [
                "a tench",
                "a goldfish",
                "a great white shark",
                "a tiger shark",
                "a hammerhead",
                "an electric ray",
                "a stingray",
                "a cock",
                "a hen",
                "an ostrich",
                "a brambling",
                "a goldfinch",
                "a house finch",
                "a junco",
                "an indigo bunting",
                "a robin",
                "a bulbul",
                "a jay",
                "a magpie",
                "a chickadee",
                "a water ouzel",
                "a kite",
                "a bald eagle",
                "a vulture",
                "a great grey owl",
                "a European fire salamander",
                "a common newt",
                "an eft",
                "a spotted salamander",
                "an axolotl",
                "a bullfrog",
                "a tree frog",
                "a tailed frog",
                "a loggerhead",
                "a leatherback turtle",
                "a mud turtle"
            ]
            for prompt in prompts:
                best_result, best_score = None, float('-inf')
                for _ in range(MASTER_PARAMS['N'] if method == "rejection" else 1):
                    result, score = local_pipe(
                        prompt=prompt,
                        num_inference_steps=num_inference_steps,
                        score_function=scorer,
                        method=method,
                        params=MASTER_PARAMS,
                    )
                    if score > best_score:
                        best_result, best_score = result, score
                # Convert tensor score to CPU float
                if torch.is_tensor(best_score):
                    best_scores.append(best_score.cpu().item())
                else:
                    best_scores.append(float(best_score))

            task_scores[f'{curr_scorer}_{method}'] = np.mean(best_scores)

        # Write separate file for each scorer/method combination
        with open(output_dir / f'{script_name}_{backend}_{curr_scorer}_{method}_steps{num_inference_steps}.txt', 'w') as f:
            for key, value in task_scores.items():
                f.write(f'{key}: {value}\n')

if __name__ == '__main__':
    main()