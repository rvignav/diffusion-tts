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
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID (0-5)')
    parser.add_argument('--task_id', type=int, default=None, help='Task ID for parallel execution')
    args = parser.parse_args()
    
    # Setup output directory
    script_name = os.path.basename(__file__).split('.')[0]
    output_dir = Path('outputs') / script_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    backend = 'sd'
    scorers = ['brightness', 'compressibility', 'clip']
    methods = ['beam', 'mcts', 'zero_order', 'eps_greedy']
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
    tasks.append(('clip', 'rejection'))
    
    # If task_id specified, run only that task
    if args.task_id is not None:
        if args.task_id >= len(tasks):
            print(f"Invalid task_id {args.task_id}. Max is {len(tasks)-1}")
            return
        tasks = [tasks[args.task_id]]

    for curr_scorer, method in tasks:
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
        num_inference_steps = 100
        MASTER_PARAMS = {
            'N': N,
            'lambda': lambda_param,
            'eps': eps,
            'K': K,
            'B': B,
            'S': S,
        }
        best_scores = []
        # prompts = [
        #             "a tench",
        #             "a goldfish",
        #             "a great white shark",
        #             "a tiger shark",
        #             "a hammerhead shark",
        #             "an electric ray",
        #             "a stingray",
        #             "a cock",
        #             "a hen",
        #             "an ostrich",
        #             "a brambling",
        #             "a goldfinch",
        #             "a house finch",
        #             "a junco",
        #             "an indigo bunting",
        #             "an American robin",
        #             "a bulbul",
        #             "a jay",
        #             "a magpie",
        #             "a chickadee",
        #             "an American dipper",
        #             "a kite",
        #             "a bald eagle",
        #             "a vulture",
        #             "a great grey owl",
        #             "a fire salamander",
        #             "a smooth newt",
        #             "a newt",
        #             "a spotted salamander",
        #             "an axolotl",
        #             "an American bullfrog",
        #             "a tree frog",
        #             "a tailed frog",
        #             "a loggerhead sea turtle",
        #             "a leatherback sea turtle",
        #             "a mud turtle",
        #             "a terrapin",
        #             "a box turtle",
        #             "a banded gecko",
        #             "a green iguana",
        #             "a Carolina anole",
        #             "a desert grassland whiptail lizard",
        #             "an agama",
        #             "a frilled‑necked lizard",
        #             "an alligator lizard",
        #             "a Gila monster",
        #             "a European green lizard",
        #             "a chameleon",
        #             "a Komodo dragon",
        #             "a Nile crocodile",
        #             "an American alligator",
        #             "a triceratops",
        #             "a worm snake",
        #             "a ring‑necked snake",
        #             "an eastern hog‑nosed snake",
        #             "a smooth green snake",
        #             "a kingsnake",
        #             "a garter snake",
        #             "a water snake",
        #             "a vine snake",
        #             "a night snake",
        #             "a boa constrictor",
        #             "an African rock python",
        #             "an Indian cobra",
        #             "a green mamba",
        #             "a sea snake",
        #             "a Saharan horned viper",
        #             "an eastern diamondback rattlesnake",
        #             "a sidewinder",
        #             "a trilobite",
        #             "a harvestman",
        #             "a scorpion",
        #             "a yellow garden spider",
        #             "a barn spider",
        #             "a European garden spider",
        #             "a southern black widow",
        #             "a tarantula",
        #             "a wolf spider",
        #             "a tick",
        #             "a centipede",
        #             "a black grouse",
        #             "a ptarmigan",
        #             "a ruffed grouse",
        #             "a prairie grouse",
        #             "a peacock",
        #             "a quail",
        #             "a partridge",
        #             "a grey parrot",
        #             "a macaw",
        #             "a sulphur‑crested cockatoo",
        #             "a lorikeet",
        #             "a coucal",
        #             "a bee eater",
        #             "a hornbill",
        #             "a hummingbird",
        #             "a jacamar",
        #             "a toucan",
        #             "a duck",
        #             "a red‑breasted merganser",
        #             "a goose"
        #         ]
        
        
        prompts = [
                    "a tench", "a goldfish", "a great white shark", "a tiger shark", "a hammerhead",
                    "an electric ray", "a stingray", "a cock", "a hen", "an ostrich",
                    "a brambling", "a goldfinch", "a house finch", "a junco", "an indigo bunting",
                    "a robin", "a bulbul", "a jay", "a magpie", "a chickadee",
                    "a water ouzel", "a kite", "a bald eagle", "a vulture", "a great grey owl",
                    "a european fire salamander", "a common newt", "an eft", "a spotted salamander", "an axolotl",
                    "a bullfrog", "a tree frog", "a tailed frog", "a loggerhead", "a leatherback turtle",
                    "a mud turtle", "a terrapin", "a box turtle", "a banded gecko", "a common iguana",
                    "an american chameleon", "a whiptail", "an agama", "a frilled lizard", "an alligator lizard",
                    "a gila monster", "a green lizard", "an african chameleon", "a komodo dragon", "an african crocodile",
                    "an american alligator", "a triceratops", "a thunder snake", "a ringneck snake", "a hognose snake",
                    "a green snake", "a king snake", "a garter snake", "a water snake", "a vine snake",
                    "a night snake", "a boa constrictor", "a rock python", "an indian cobra", "a green mamba",
                    "a sea snake", "a horned viper", "a diamondback", "a sidewinder", "a trilobite",
                    "a harvestman", "a scorpion", "a black and gold garden spider", "a barn spider", "a garden spider",
                    "a black widow", "a tarantula", "a wolf spider", "a tick", "a centipede",
                    "a black grouse", "a ptarmigan", "a ruffed grouse", "a prairie chicken", "a peacock",
                    "a quail", "a partridge", "an african grey", "a macaw", "a sulphur-crested cockatoo",
                    "a lorikeet", "a coucal", "a bee eater", "a hornbill", "a hummingbird",
                    "a jacamar", "a toucan", "a drake", "a red-breasted merganser", "a goose"
                    ] + [
                    "a black swan", "a tusker", "an echidna", "a platypus", "a wallaby",
                    "a koala", "a wombat", "a jellyfish", "a sea anemone", "a brain coral",
                    "a flatworm", "a nematode", "a conch", "a snail", "a slug",
                    "a sea slug", "a chiton", "a chambered nautilus", "a dungeness crab", "a rock crab",
                    "a fiddler crab", "a king crab", "an american lobster", "a spiny lobster", "a crayfish",
                    "a hermit crab", "an isopod", "a white stork", "a black stork", "a spoonbill",
                    "a flamingo", "a little blue heron", "an american egret", "a bittern", "a crane",
                    "a limpkin", "a european gallinule", "an american coot", "a bustard", "a ruddy turnstone",
                    "a red-backed sandpiper", "a redshank", "a dowitcher", "an oystercatcher", "a pelican",
                    "a king penguin", "an albatross", "a grey whale", "a killer whale", "a dugong",
                    "a sea lion", "a chihuahua", "a japanese spaniel", "a maltese dog", "a pekinese",
                    "a shih-tzu", "a blenheim spaniel", "a papillon", "a toy terrier", "a rhodesian ridgeback",
                    "an afghan hound", "a basset", "a beagle", "a bloodhound", "a bluetick",
                    "a black-and-tan coonhound", "a walker hound", "an english foxhound", "a redbone", "a borzoi",
                    "an irish wolfhound", "an italian greyhound", "a whippet", "an ibizan hound", "a norwegian elkhound",
                    "an otterhound", "a saluki", "a scottish deerhound", "a weimaraner", "a staffordshire bullterrier",
                    "an american staffordshire terrier", "a bedlington terrier", "a border terrier", "a kerry blue terrier", "an irish terrier",
                    "a norfolk terrier", "a norwich terrier", "a yorkshire terrier", "a wire-haired fox terrier", "a lakeland terrier",
                    "a sealyham terrier", "an airedale", "a cairn", "an australian terrier", "a dandie dinmont",
                    "a boston bull", "a miniature schnauzer", "a giant schnauzer", "a standard schnauzer", "a scotch terrier"
                    ] + [
                    "a tibetan terrier", "a silky terrier", "a soft-coated wheaten terrier", "a west highland white terrier", "a lhasa",
                    "a flat-coated retriever", "a curly-coated retriever", "a golden retriever", "a labrador retriever", "a chesapeake bay retriever",
                    "a german short-haired pointer", "a vizsla", "an english setter", "an irish setter", "a gordon setter",
                    "a brittany spaniel", "a clumber", "an english springer", "a welsh springer spaniel", "a cocker spaniel",
                    "a sussex spaniel", "an irish water spaniel", "a kuvasz", "a schipperke", "a groenendael",
                    "a malinois", "a briard", "a kelpie", "a komondor", "an old english sheepdog",
                    "a shetland sheepdog", "a collie", "a border collie", "a bouvier des flandres", "a rottweiler",
                    "a german shepherd", "a doberman", "a miniature pinscher", "a greater swiss mountain dog", "a bernese mountain dog",
                    "an appenzeller", "an entlebucher", "a boxer", "a bull mastiff", "a tibetan mastiff",
                    "a french bulldog", "a great dane", "a saint bernard", "an eskimo dog", "a malamute",
                    "a siberian husky", "a dalmatian", "an affenpinscher", "a basenji", "a pug",
                    "a leonberg", "a newfoundland", "a great pyrenees", "a samoyed", "a pomeranian",
                    "a chow", "a keeshond", "a brabancon griffon", "a pembroke", "a cardigan",
                    "a toy poodle", "a miniature poodle", "a standard poodle", "a mexican hairless", "a timber wolf",
                    "a white wolf", "a red wolf", "a coyote", "a dingo", "a dhole",
                    "an african hunting dog", "a hyena", "a red fox", "a kit fox", "an arctic fox",
                    "a grey fox", "a tabby", "a tiger cat", "a persian cat", "a siamese cat",
                    "an egyptian cat", "a cougar", "a lynx", "a leopard", "a snow leopard",
                    "a jaguar", "a lion", "a tiger", "a cheetah", "a brown bear",
                    "an american black bear", "an ice bear", "a sloth bear", "a mongoose", "a meerkat"
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
            best_scores.append(best_score)

        task_scores[f'{curr_scorer}_{method}'] = np.mean(best_scores)

        # Write separate file for each scorer/method combination
        with open(output_dir / f'{script_name}_{backend}_{curr_scorer}_{method}_steps{num_inference_steps}.txt', 'w') as f:
            for key, value in task_scores.items():
                f.write(f'{key}: {value}\n')

if __name__ == '__main__':
    main()