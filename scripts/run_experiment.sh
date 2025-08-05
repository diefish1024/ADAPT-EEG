#!/bin/bash
# scripts/run_experiment.sh

set -e

DEFAULT_CONFIG="configs/seed_emt_binary_tent.yaml"
CONFIG_PATH=""

# Check if the first argument exists and is a file path (not a flag)
if [ -n "$1" ] && [ "$(echo "$1" | cut -c1)" != "-" ]; then
    CONFIG_PATH=$1
    shift 
else
    CONFIG_PATH=$DEFAULT_CONFIG
fi

echo "================================================="
echo "Starting ADAPT-EEG Experiment"
echo "Configuration: $CONFIG_PATH"
if [ "$#" -gt 0 ]; then
    echo "Arguments:     $@"
fi
echo "================================================="

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Configuration file not found at $CONFIG_PATH"
    exit 1
fi

python -m src.main --config "$CONFIG_PATH" "$@"

echo "-------------------------------------------------"
echo "Experiment finished successfully."
echo "-------------------------------------------------"