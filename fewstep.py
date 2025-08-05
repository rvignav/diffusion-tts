print("================================================")
print("fewstep")
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
    backend = 'sd'
    scorers = ['clip']
    methods = ['zero_order', 'eps_greedy', 'beam']
    N = 4
    lambda_param = 0.15
    eps = 0.4
    K = 20
    B = 2
    S = 8
    seed = 0
    device = f'cuda'

    for curr_scorer, method in zip(scorers, methods):
        task_scores = {}  # Separate scores for this task
        StableDiffusionPipeline, DDIMScheduler, BrightnessScorer, CompressibilityScorer, CLIPScorer = import_sd()
        scorer = get_scorer('sd', curr_scorer, BrightnessScorer, CompressibilityScorer, CLIPScorer=CLIPScorer)

        model_id = "runwayml/stable-diffusion-v1-5"
        local_scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        local_pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=local_scheduler,
            torch_dtype=torch.float16,
        ).to(device)

        method = method
        num_inference_steps = 18
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
                    # "a goldfish",
                    # "a great white shark",
                    # "a tiger shark",
                    # "a hammerhead shark",
                    # "an electric ray",
                    # "a stingray",
                    # "a cock",
                    # "a hen",
                    # "an ostrich",
                    # "a brambling",
                    # "a goldfinch",
                    # "a house finch",
                    # "a junco",
                    # "an indigo bunting",
                    # "an American robin",
                    # "a bulbul",
                    # "a jay",
                    # "a magpie",
                    # "a chickadee",
                    # "an American dipper",
                    # "a kite",
                    # "a bald eagle",
                    # "a vulture",
                    # "a great grey owl",
                    # "a fire salamander",
                    # "a smooth newt",
                    # "a newt",
                    # "a spotted salamander",
                    # "an axolotl",
                    # "an American bullfrog",
                    # "a tree frog",
                    # "a tailed frog",
                    # "a loggerhead sea turtle",
                    # "a leatherback sea turtle",
                    # "a mud turtle",
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
                    config=None,
                    few_step_model="XCLiu/instaflow_0_9B_from_sd_1_5"
                )
                if score > best_score:
                    best_result, best_score = result, score
            best_scores.append(best_score)

        task_scores[f'{curr_scorer}_{method}'] = np.mean(best_scores)
        print(f'{curr_scorer}_{method}: {task_scores[f"{curr_scorer}_{method}"]}')

if __name__ == '__main__':
    main()