#!/bin/bash

echo "Searching for an available GPU..."

while true; do
    # List all GPUs and their free memory
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits

    # Pick the GPU with the most free memory above the threshold
    GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -nrk2 | awk '$2 > 1000 {print $1; exit}')

    if [[ -n "$GPU" ]]; then
        echo "Starting training on GPU $GPU..."
        export CUDA_VISIBLE_DEVICES=$GPU

        # Start the training process
        python train.py -s /data1/alex/datasets/tanks_templates/tanksandtemples/truck --lambda_segment 0.00

        # If training exits successfully, break the loop
        if [[ $? -eq 0 ]]; then
            echo "Training completed successfully."
            break
        else
            echo "Training crashed or failed. Retrying..."
        fi
    else
        echo "No available GPU found. Retrying in 10 seconds..."
        sleep 10
    fi
done
