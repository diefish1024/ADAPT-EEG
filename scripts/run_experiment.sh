#!/bin/bash

set -e

echo "Starting ADAPT-EEG Experiment..."

CONFIG_PATH="configs/experiment_seed.yaml"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Configuratoin file not found at $CONFIG_PATH"
    exit 1
fi

echo "Using configuration: $CONFIG_PATH"

python -m src.main --config "$CONFIG_PATH"

echo "ADAPT-EEG Experiment Finished."