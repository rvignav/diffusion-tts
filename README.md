# Test-Time Scaling of Diffusion Models via Noise Trajectory Search

## Installation

```
cd edm/ && conda env create -f environment.yml -n edm -y && cd ..
conda activate edm
pip install -r requirements.txt
```

## Generating Images

### Class-Conditional Generation (EDM)

`cd` into `edm`, then edit `main.py` to use the desired method (one of `SamplingMethod.NAIVE`, `SamplingMethod.REJECTION_SAMPLING`, `SamplingMethod.BEAM_SEARCH`, `SamplingMethod.MCTS`, `SamplingMethod.ZERO_ORDER`, `SamplingMethod.EPS_GREEDY`).

To use any parameters different from those in the paper, edit the params object corresponding to your method (e.g. `MCTSParams`, `NaiveParams`, etc.). Then simply run `python main.py`.

### Text-to-Image Generation (Stable Diffusion)

`cd` into `sd`, then edit `main.py` to use the desired method (one of `naive`, `rejection`, `beam`, `mcts`, `zero_order`, or `eps_greedy`) and any params different from those in `MASTER_PARAMS` (which were used to generate the results in the paper). Also rewrite the `prompt` variable as desired. Then simply run `python main.py`.