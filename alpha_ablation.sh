#!/bin/bash

# Seeds
seeds=(0 1 2)

# Games (tasks)
tasks=("atari_pong")  # Add more tasks as needed

# Alphas
alphas=(0 2 5 10)

# Loop through each configuration
for seed in "${seeds[@]}"; do
    for task in "${tasks[@]}"; do
        for alpha in "${alphas[@]}"; do
            # Format the log directory with the task, seed, and alpha
            logdir=~/logdir/${task}_seed${seed}_alpha${alpha}_$(date '+%Y%m%d-%H%M%S')

            echo "Running with seed: $seed, task: $task, alpha: $alpha"
            sbatch --gpus=a100:1 --mem-per-cpu=32GB --time=4:00:00 \
                --wrap="python dreamerv3/train.py \
                --logdir $logdir \
                --configs atari100k --seed $seed --task $task --dynamics.alpha_strength $alpha"
        done
    done
done

