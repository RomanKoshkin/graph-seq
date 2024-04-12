# Unsupervised Detection of Cell Assemblies with Graph Neural Networks

![](videos/output.gif)
- [PAPER](https://openreview.net/pdf?id=Tbzv_BbjjO8)

# Abstract

Cell assemblies, putative units of neural computation, manifest themselves as repeating and temporally coordinated activity of neurons. However, understanding of their role in brain function is hampered by a lack of scalable methods for their unsupervised detection. We propose using a graph neural network for embedding spike data into a sequence of fixed size vectors and clustering them based on their
self-similarity across time. We validate our method on synthetic data and real neural recordings.

# Quickstart

## Install dependencies

We recommend starting with a fresh Conda environment and installing all the dependencies in `requirements.txt` (although not all of them are needed for the code in this repo to work).

## Generate a synthetic dataset

```bash
cd demo
python make_dataset.py \
    --p_drop 0.2 \
    --gap_ts 800 \
    --seqlen 100 \
    --jitter_std 15 \
    --winsize 100 \
    --step_sz 4 \
    --tau 25
```

## Run sequence detection

```bash
cd demo
python detect.py \
    --snapshot_interval 5 \
    --z_dim 6 \
    --K 6 \
    --epochs 400
```

## Visualize the optimization process

```bash
cd scripts
sh make_vids2.sh
```
You video will be assembled from frames (stored in `data`) and saved as an mp4 and a gif file.

## Tensorboard

For your dataset, you might want to monitor the losses to tweak the hyperparameters. After at least one optimization run is started

```bash
cd runs
tensorboard --logdir=.
```
and navigate to `http://localhost:6006` to monitor the losses.
