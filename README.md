# Test-Time Scaling of Diffusion Models via Noise Trajectory Search

## Installation

```
conda env create -f environment.yml -n edm -y
conda activate edm
pip install -r requirements.txt
```

## Generating Images

### Class-Conditional Generation (EDM)

`cd` into `edm`, then edit `main.py` to use the desired method (one of the following):
```
class SamplingMethod(Enum):
    MCTS = auto()
    BEAM_SEARCH = auto()
    ZERO_ORDER = auto()
    NAIVE = auto()
    REJECTION_SAMPLING = auto()
    EPS_GREEDY = auto()
```
To use any parameters different from those in the paper, edit the params object corresponding to your method (e.g. `MCTSParams`, `NaiveParams`, etc.). Then simply run `python main.py`.

### Text-to-Image Generation (Stable Diffusion)

`cd` into `sd`, then edit `main.py` to use the desired method (one of `naive`, `rejection`, `beam`, `mcts`, `zero_order`, or `eps_greedy`) and any params different from those in `MASTER_PARAMS` (which were used to generate the results in the paper). Then simply run `python main.py`.