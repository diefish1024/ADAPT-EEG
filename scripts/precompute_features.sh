#!/bin/bash
# scripts/precompute_features.sh
#
# If no config path is provided, it uses a default.
#
# Usage:
#   # Run with default config
#   bash scripts/precompute_features.sh
#
#   # Run with a specific config
#   bash scripts/precompute_features.sh configs/experiment/YOUR_OTHER_EXP.yaml

set -e

# --- Configuration ---
DEFAULT_CONFIG="configs/experiment/seed_emt_binary_tent.yaml"
CONFIG_PATH="${1:-$DEFAULT_CONFIG}" # Use first argument or default

echo "================================================="
echo "Starting Feature Pre-computation"
echo "Using configuration from: $CONFIG_PATH"
echo "================================================="

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Configuration file not found at $CONFIG_PATH"
    exit 1
fi

# Run the pre-computation script
python -m tools.precompute_features --config "$CONFIG_PATH"

echo "================================================="
echo "Feature Pre-computation Finished."
echo "================================================="
