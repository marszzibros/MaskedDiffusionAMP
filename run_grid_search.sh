#!/bin/bash

# Configuration parameters
CHECKPOINT_PATH="./output/20251224-123317/model-epoch_800.ckpt"
NUM_SAMPLES=500
GROUPS="0 1"

# Create output directory in the same folder as the checkpoint
CHECKPOINT_DIR=$(dirname "$CHECKPOINT_PATH")
OUTPUT_DIR="${CHECKPOINT_DIR}/grid_search_scales"
mkdir -p "$OUTPUT_DIR"

SUMMARY_FILE="${OUTPUT_DIR}/grid_search_summary.csv"

echo "Starting grid search..."
echo "Results will be saved to: $OUTPUT_DIR"
echo "Summary file: $SUMMARY_FILE"
echo "------------------------------------------------------"

for species in $(seq 0.0 1.0 7.0); do
    for groups in $(seq 0.0 1.0 7.0); do
        for mic in $(seq 0.0 1.0 7.0); do
            echo "Testing scales - Species: $species | Groups: $groups | MIC: $mic"
            
            # Format the output file name with the current scale values
            OUTPUT_FILE="${OUTPUT_DIR}/sp_${species}_gr_${groups}_mic_${mic}.csv"
            
            # Run the python script
            python experiment_scales.py \
                --checkpoint_path "$CHECKPOINT_PATH" \
                --num_samples $NUM_SAMPLES \
                --species_scale $species \
                --groups_scale $groups \
                --mic_scale $mic \
                --output_file "$OUTPUT_FILE" \
                --summary_file "$SUMMARY_FILE" \
                --groups 0 1 \
                --species 0
        done
    done
done

echo "------------------------------------------------------"
echo "Grid search completed successfully!"
echo "Summary saved to: $SUMMARY_FILE"
