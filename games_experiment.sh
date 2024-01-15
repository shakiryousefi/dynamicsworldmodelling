#!/bin/bash

# Seeds
seeds=(0 1 2)

# Games (tasks)
tasks=("atari_pong" "atari_ms_pacman" "atari_boxing")  # Add more tasks as needed

#K steps


# Loop through each configuration
for seed in "${seeds[@]}"; do
    for task in "${tasks[@]}"; do
        # Format the log directory with the seed and taska
        #logdir=/logdir/${task}_seed${seed}_$(date '+%Y%m%d-%H%M%S')

        echo "Running with seed: $seed, task: $task"
        sbatch --gpus=a100:1 --mem-per-cpu=32GB --time=3:10:00 \
            --wrap="python dreamerv3/train.py \
            --logdir logdir=~/logdir/${task}_seed${seed}_$(date '+%Y%m%d-%H%M%S') \
            --configs atari100k --seed $seed --task $task --dynamics.alpha_strength 5 --dynamics.k_steps 3"
    done
done


