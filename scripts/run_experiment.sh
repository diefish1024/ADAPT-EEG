#!/bin/bash
# scripts/run_experiment.sh

set -e

# --- Configuration ---
DEFAULT_CONFIG="configs/seed_emt_binary_tent.yaml"
CONFIG_PATH=""
OTHER_ARGS=()

# --- Argument Parsing ---
# Loop through all provided arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      # If --config is found, take the next argument as its value
      CONFIG_PATH="$2"
      shift 2 # Move past --config and its value
      ;;
    *)
      # Store any other argument (e.g., --resume)
      OTHER_ARGS+=("$1")
      shift # Move past the current argument
      ;;
  esac
done

# If no --config was provided, use the default
if [ -z "$CONFIG_PATH" ]; then
    CONFIG_PATH=$DEFAULT_CONFIG
fi

echo "================================================="
echo "Starting ADAPT-EEG Experiment"
echo "Configuration: $CONFIG_PATH"
if [ ${#OTHER_ARGS[@]} -gt 0 ]; then
    echo "Arguments:     ${OTHER_ARGS[@]}"
fi
echo "================================================="

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Configuration file not found at $CONFIG_PATH"
    exit 1
fi

python -m src.main --config "$CONFIG_PATH" "${OTHER_ARGS[@]}"

echo "-------------------------------------------------"
echo "Experiment finished successfully."
echo "-------------------------------------------------"