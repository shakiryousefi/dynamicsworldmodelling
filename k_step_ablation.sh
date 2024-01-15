#!/bin/bash

# Seeds
seeds=(0 1 2)

# Games (tasks)
tasks=("atari_pong")

# Alphas
ks=(3 5 10)

# Loop through each configuration
for seed in "${seeds[@]}"; do
    for task in "${tasks[@]}"; do
        for k in "${ks[@]}"; do
            echo "Running with seed: $seed, task: $task, k: $k"
            sbatch --gpus=a100:1 --mem-per-cpu=32GB --time=4:00:00 \
                --wrap="python dreamerv3/train.py \
                --logdir logdir=~/logdir/${task}_seed${seed}_k${k}_$(date '+%Y%m%d-%H%M%S') \
                --configs atari100k --seed $seed --task $task --dynamics.alpha_strength 5 --dynamics.k_steps $k"
        done
    done
done
