# Matrix Completion and Effective-Rank Experiments

Research code for studying effective rank in deep matrix factorization and in
fully connected or convolutional networks trained on CIFAR-10/CIFAR-100.

## Repository layout

```text
experiments/
  matrix_completion.py       Deep linear matrix-completion experiment
  cifar_fcn/                 Fully connected CIFAR-10 experiments
  cifar_cnn/                 ResNet/VGG CIFAR-10 and CIFAR-100 experiments
    legacy/                  Preserved, non-default subset-selection helper
notebooks/                   Analysis notebooks (stored without outputs)
```

Datasets, checkpoints, W&B run data, logs, figures, and result tables are
generated locally and excluded from Git.

## Setup

Python 3.9 or newer is recommended. Install a PyTorch build appropriate for
your CUDA version, then install the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The original, fully pinned Conda environment is retained at
`experiments/cifar_cnn/environment.yml` for reproducibility.

## Matrix completion

The former depth-specific scripts are consolidated into one command:

```bash
python experiments/matrix_completion.py \
  --depth 3 \
  --device cuda:0 \
  --seed-start 80 \
  --seed-end 100
```

Results are written under `outputs/matrix_completion/`.

## CIFAR fully connected networks

Run one configuration:

```bash
python experiments/cifar_fcn/train.py \
  --depth 3 \
  --activation relu \
  --optimizer sgd \
  --out-json outputs/cifar_fcn/run.json \
  --out-csv outputs/cifar_fcn/summary.csv
```

Run the predefined sweep, assigning one or more comma-separated CUDA devices:

```bash
GPUS=0,1 bash experiments/cifar_fcn/run_sweep.sh
```

Track the average effective rank of a standard MLP:

```bash
python experiments/cifar_fcn/track_effective_rank.py \
  --device cuda:0 \
  --rank-log outputs/cifar_fcn/effective_rank.json \
  --rank-plot outputs/cifar_fcn/effective_rank.png
```

## CIFAR convolutional networks

The CNN experiments log metrics to Weights & Biases. Authenticate with
`wandb login` before running them:

```bash
python experiments/cifar_cnn/train.py \
  --dataset cifar10 \
  --arch resnet18 \
  --gpu 0 \
  --save_path outputs/cifar_cnn
```

Use `experiments/cifar_cnn/train_lop.py` for the LoP variant. Analysis code is
in `notebooks/plot_cnn_wandb.ipynb`.
