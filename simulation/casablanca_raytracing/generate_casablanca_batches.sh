#!/bin/bash

# Navigate to the casablanca raytracing directory
cd "$(dirname "$0")" || exit

NUM_BATCHES=10

# Parameters for a "small" batch to save resources
SIM_TIME=30
NUM_UES=5
NUM_DRONES=2

echo "Starting generation of $NUM_BATCHES small batches for Casablanca Raytracing..."

# Create a directory to store the batches
mkdir -p batches

for i in $(seq 1 $NUM_BATCHES); do
    echo "----------------------------------------"
    echo "Running batch $i/$NUM_BATCHES (Seed: $i)..."
    
    /home/wassimmchichi/miniconda3/envs/sionna-env/bin/python run_hybrid_pipeline_sionna.py \
        --sim-time $SIM_TIME \
        --num-ues $NUM_UES \
        --num-drones $NUM_DRONES \
        --seed $i \
        --scene casablanca_scene.xml \
        --enable-rt \
        --output "batches/batch_${i}_hybrid_handover_dataset_sionna.csv" \
        --drone-output "batches/batch_${i}_hybrid_drone_positions_sionna.csv" \
        --summary "batches/batch_${i}_hybrid_handover_summary_sionna.json"
        
    echo "Batch $i completed."
done

echo "----------------------------------------"
echo "All $NUM_BATCHES batches generated successfully in the 'batches' directory!"
echo "Combining datasets into a single CSV..."

FINAL_DATASET="combined_hybrid_handover_dataset_sionna.csv"
FINAL_DRONE_DATASET="combined_hybrid_drone_positions_sionna.csv"

# Extract header from the first batch and write to the final files
head -n 1 "batches/batch_1_hybrid_handover_dataset_sionna.csv" > "$FINAL_DATASET"
head -n 1 "batches/batch_1_hybrid_drone_positions_sionna.csv" > "$FINAL_DRONE_DATASET"

# Append the contents (without headers) from all batches
for i in $(seq 1 $NUM_BATCHES); do
    if [ -f "batches/batch_${i}_hybrid_handover_dataset_sionna.csv" ]; then
        tail -n +2 "batches/batch_${i}_hybrid_handover_dataset_sionna.csv" >> "$FINAL_DATASET"
        tail -n +2 "batches/batch_${i}_hybrid_drone_positions_sionna.csv" >> "$FINAL_DRONE_DATASET"
    else
        echo "Warning: Batch $i missing. Skipping in combined CSV."
    fi
done

echo "Combined handover dataset saved to: $FINAL_DATASET"
echo "Combined drone positions saved to: $FINAL_DRONE_DATASET"
