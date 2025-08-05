print("================================================")
print("counting: SD counting reward")
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
    from scorers import BrightnessScorer, CompressibilityScorer, CLIPScorer, CountingScorer
    return StableDiffusionPipeline, DDIMScheduler, BrightnessScorer, CompressibilityScorer, CLIPScorer, CountingScorer

# =========================
# Scorer Factory
# =========================
def get_scorer(backend, scorer_name, BrightnessScorer, CompressibilityScorer, CLIPScorer=None, ImageNetScorer=None, CountingScorer=None):
    """Return the appropriate scorer instance for the backend and scorer name."""
    if scorer_name == 'brightness':
        return BrightnessScorer(dtype=torch.float32)
    elif scorer_name == 'compressibility':
        return CompressibilityScorer(dtype=torch.float32)
    elif scorer_name == 'clip' and backend == 'sd':
        return CLIPScorer(dtype=torch.float32)
    elif scorer_name == 'imagenet' and backend == 'edm':
        return ImageNetScorer(dtype=torch.float32)
    elif scorer_name == 'counting':
        return CountingScorer(dtype=torch.float32)
    else:
        raise ValueError(f"Unknown or invalid scorer '{scorer_name}' for backend '{backend}'")

# =========================
# Main Logic
# =========================
word2num = {
    "One": 1,  "Two": 2,  "Three": 3,  "Four": 4,  "Five": 5,
    "Six": 6,  "Seven": 7,  "Eight": 8,  "Nine": 9,  "Ten": 10
}

def parse_prompt(prompt: str) -> dict:
    """
    Given a prompt like "Five horses, three cars, one train, five airplanes",
    returns a dict with
      - class_names: "horses, cars, train, airplanes"
      - class_gt_counts: "5, 3, 1, 5"
    """
    # split on commas and also handle ' and '
    parts = [p.strip() for p in prompt.replace(" and ", ", ").split(",") if p.strip()]
    names, counts = [], []
    for part in parts:
        tokens = part.split()
        # first token is the spelled-out number
        num_word = tokens[0].capitalize()
        count = word2num.get(num_word)
        if count is None:
            raise ValueError(f"Unknown number word: {num_word}")
        # the rest is the class name
        name = " ".join(tokens[1:])
        counts.append(count)
        names.append(name)
    return {
        "class_names":  ", ".join(names),
        "class_gt_counts": ", ".join(str(c) for c in counts)
    }

def main():
    backend = 'sd'
    scorers = ['counting']
    methods = ['naive', 'rejection', 'beam', 'mcts', 'zero_order', 'eps_greedy']
    N = 4
    lambda_param = 0.15
    eps = 0.4
    K = 20
    B = 2
    S = 8
    seed = 0
    device = f'cuda'
    
    # Create all task combinations
    tasks = [(scorer, method) for scorer in scorers for method in methods]
    
    for curr_scorer, method in tasks:
        task_scores = {}  # Separate scores for this task
        StableDiffusionPipeline, DDIMScheduler, BrightnessScorer, CompressibilityScorer, CLIPScorer, CountingScorer = import_sd()
        scorer = get_scorer('sd', curr_scorer, BrightnessScorer, CompressibilityScorer, CLIPScorer=CLIPScorer, CountingScorer=CountingScorer)

        model_id = "runwayml/stable-diffusion-v1-5"
        local_scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        local_pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=local_scheduler,
            torch_dtype=torch.float16,
        ).to(device)

        method = method
        num_inference_steps = 50
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
            "Five horses, three cars, one train, five airplanes",
            # "Two cups, three paintings, four lamps, and four bananas",
            # "Four balloons, one cup, four desks, two dogs and four microwaves",
            # "Four candles, two balloons, one dog, two tomatoes and three helicopters",
            # "Eight chairs",
            # "Three apples, two oranges, and one banana",
            # "Six pencils, four erasers, and two notebooks",
            # "One globe, three backpacks, two rulers, and five markers",
            # "Four socks, one hat, and two scarves",
            # "Seven balloons, three kites, and one paper airplane",
            # "Two swords and three shields",
            # "Five pillows, two blankets, and one teddy bear",
            # "Nine bottles, one glass, and two mugs",
            # "Eight wheels, four pedals, and one seat",
            # "Two cameras, three tripods, and four lenses",
            # "One laptop, two mice, and three keyboards",
            # "Six tomatoes, four cucumbers, and two peppers",
            # "Three birds, two cats, and one dog",
            # "Seven flowers, one vase, and two candles",
            # "Five coins, three bills, and one check",
            # "Two shirts, four pants, and one jacket",
            # "Eight stamps, one envelope, and three letters",
            # "One violin, two guitars, and three drums",
            # "Four chairs and two tables",
            # "Six plants and one watering can",
            # "Three cookies, two cakes, and one pie",
            # "Nine bricks, four stones, and one wood plank",
            # "Two buses, one train, and three bicycles",
            # "Seven socks, three shoes, and one hat",
            # "Six eggs and two cartons",
            # "Five keys, one lock, and three doors",
            # "Four spoons, two forks, and one knife",
            # "Three rings, two necklaces, and one bracelet",
            # "Eight lamps and two fans",
            # "One television, three speakers, and two remotes",
            # "Seven books and one bookmark",
        ]
        configs = [parse_prompt(p) for p in prompts]
        for prompt, config in zip(prompts, configs):
            best_result, best_score = None, float('-inf')
            for _ in range(MASTER_PARAMS['N'] if method == "rejection" else 1):
                result, score = local_pipe(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    score_function=scorer,
                    method=method,
                    params=MASTER_PARAMS,
                    config=config,
                )
                if score > best_score:
                    best_result, best_score = result, score
            best_scores.append(best_score)

        task_scores[f'{curr_scorer}_{method}'] = np.mean(best_scores)
        print(f'{curr_scorer}_{method}: {task_scores[f"{curr_scorer}_{method}"]}')

if __name__ == '__main__':
    main()