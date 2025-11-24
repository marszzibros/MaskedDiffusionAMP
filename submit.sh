#!/bin/bash

# Define the list of arguments
# fusions=("sigmoid_4" "attention_seq_only_4" "crossattention_4" "without_norm_cross" "without_structure_4")
fusions=("sigmoid" "crossattention" "without_structure")
# Submit a job for each fusion argument
for categor in 10 20 30; do
    for fusion in "${fusions[@]}"; do
        sbatch --export=ALL,FUSION_ARG="$fusion",CAT_ARG="$categor" train.sh
    done
done
