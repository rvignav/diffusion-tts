import os
import sys
import importlib.util
import torch
import torch.multiprocessing as mp
import math
from functools import partial
from pathlib import Path
import numpy as np

def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, os.path.abspath(relative_path))
    sys.modules[name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sys.modules[name])

load_module("diffusers", "diffusers/src/diffusers/__init__.py")
from diffusers import StableDiffusionPipeline, DDIMScheduler
from scorers import BrightnessScorer, CompressibilityScorer, CLIPScorer
from tqdm import tqdm
from PIL import Image

model_id = "runwayml/stable-diffusion-v1-5"
local_scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
local_pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    scheduler=local_scheduler,
    torch_dtype=torch.float16,
).to(f'cuda')

method = "naive" # can be either "naive", "rejection", "beam", "mcts", "zero_order", or "eps_greedy"

MASTER_PARAMS = {
    'N': 4,
    'lambda': 0.15,
    'eps': 0.4,
    'K': 20,
    'B': 2,
    'S': 8,
}

prompt = "YOUR PROMPT HERE"
method = "naive" # alternatives: "rejection", "beam", "mcts", "zero_order", "eps_greedy"
    
for name, scorer in zip(["brightness", "compressibility", "clip"], [BrightnessScorer(), CompressibilityScorer(), CLIPScorer()]):
    best_result, best_score = None, 0
    for _ in range(MASTER_PARAMS['N'] if method == "rejection" else 1):
        result, score = local_pipe(
            prompt=prompt, 
            num_inference_steps=18,
            score_function=scorer,
            method=method,
            params=MASTER_PARAMS,
        )
        if score > best_score:
            best_result, best_score = result, score

    result = best_result
    best_result.images[0].save(f'{method}_{name}.png')
            
    print(best_score)