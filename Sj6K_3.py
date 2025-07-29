print("================================================")
print("Sj6K_3: SD eval on 100 images")
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
    scorers = ['brightness', 'compressibility', 'clip']
    methods = ['naive', 'rejection', 'beam', 'mcts', 'zero_order', 'eps_greedy']
    N = 4
    lambda_param = 0.15
    eps = 0.4
    K = 20
    B = 2
    S = 8
    seed = 0
    device = 'cuda'

    scores = {}

    for curr_scorer in scorers:
        for method in methods:
            if True:
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
                    "a hammerhead shark",
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
                    "an American robin",
                    "a bulbul",
                    "a jay",
                    "a magpie",
                    "a chickadee",
                    "an American dipper",
                    "a kite",
                    "a bald eagle",
                    "a vulture",
                    "a great grey owl",
                    "a fire salamander",
                    "a smooth newt",
                    "a newt",
                    "a spotted salamander",
                    "an axolotl",
                    "an American bullfrog",
                    "a tree frog",
                    "a tailed frog",
                    "a loggerhead sea turtle",
                    "a leatherback sea turtle",
                    "a mud turtle",
                    "a terrapin",
                    "a box turtle",
                    "a banded gecko",
                    "a green iguana",
                    "a Carolina anole",
                    "a desert grassland whiptail lizard",
                    "an agama",
                    "a frilled‑necked lizard",
                    "an alligator lizard",
                    "a Gila monster",
                    "a European green lizard",
                    "a chameleon",
                    "a Komodo dragon",
                    "a Nile crocodile",
                    "an American alligator",
                    "a triceratops",
                    "a worm snake",
                    "a ring‑necked snake",
                    "an eastern hog‑nosed snake",
                    "a smooth green snake",
                    "a kingsnake",
                    "a garter snake",
                    "a water snake",
                    "a vine snake",
                    "a night snake",
                    "a boa constrictor",
                    "an African rock python",
                    "an Indian cobra",
                    "a green mamba",
                    "a sea snake",
                    "a Saharan horned viper",
                    "an eastern diamondback rattlesnake",
                    "a sidewinder",
                    "a trilobite",
                    "a harvestman",
                    "a scorpion",
                    "a yellow garden spider",
                    "a barn spider",
                    "a European garden spider",
                    "a southern black widow",
                    "a tarantula",
                    "a wolf spider",
                    "a tick",
                    "a centipede",
                    "a black grouse",
                    "a ptarmigan",
                    "a ruffed grouse",
                    "a prairie grouse",
                    "a peacock",
                    "a quail",
                    "a partridge",
                    "a grey parrot",
                    "a macaw",
                    "a sulphur‑crested cockatoo",
                    "a lorikeet",
                    "a coucal",
                    "a bee eater",
                    "a hornbill",
                    "a hummingbird",
                    "a jacamar",
                    "a toucan",
                    "a duck",
                    "a red‑breasted merganser",
                    "a goose"
                ]
                for prompt in prompts:
                    best_result, best_score = None, float('-inf')
                    for _ in range(MASTER_PARAMS['N'] if method == "rejection" else 1):
                        result, score = local_pipe(
                            prompt=prompt,
                            num_inference_steps=50,
                            score_function=scorer,
                            method=method,
                            params=MASTER_PARAMS,
                        )
                        if score > best_score:
                            best_result, best_score = result, score
                    best_scores.append(best_score)

                scores[f'{curr_scorer}_{method}'] = np.mean(best_scores)

    current_filename = os.path.basename(__file__).split('.')[0]
    with open(f'{current_filename}.txt', 'a') as f:
        for key, value in scores.items():
            f.write(f'{key}: {value}\n')

if __name__ == '__main__':
    main()