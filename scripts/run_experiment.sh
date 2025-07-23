#!/bin/bash

set -e

echo "Starting ADAPT-EEG Experiment..."

CONFIG_PATH="configs/experiment_seed.yaml"
TO_EMAIL="huangjiehang@sjtu.edu.cn"
FROM_EMAIL="server@$(hostname)" # Using hostname for sender address

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Configuration file not found at $CONFIG_PATH"
    
    # Send email notification for error
    SUBJECT="[ADAPT-EEG Notification] Experiment Failed - Configuration Missing"
    MESSAGE="ADAPT-EEG experiment failed to start.\n\nError: Configuration file not found at $CONFIG_PATH\n\nTime: $(date)\nServer: $(hostname)"
    echo -e "$MESSAGE" | mail -s "$SUBJECT" -a "From: $FROM_EMAIL" "$TO_EMAIL"

    exit 1
fi

echo "Using configuration: $CONFIG_PATH"

# Run the main experiment
python -m src.main --config "$CONFIG_PATH"

# Capture the exit status of the previous command (python -m src.main)
EXIT_STATUS=$?

echo "ADAPT-EEG Experiment Finished."

# Send email notification based on exit status
if [ $EXIT_STATUS -eq 0 ]; then
    STATUS="Success"
    SUBJECT="[ADAPT-EEG Notification] Experiment Finished - Status: Success"
    MESSAGE="ADAPT-EEG experiment completed successfully.\n\nConfiguration: $CONFIG_PATH\n\nTime: $(date)\nServer: $(hostname)"
else
    STATUS="Failure"
    SUBJECT="[ADAPT-EEG Notification] Experiment Finished - Status: Failure"
    MESSAGE="ADAPT-EEG experiment encountered an error and terminated.\n\nConfiguration: $CONFIG_PATH\n\nTime: $(date)\nServer: $(hostname)\n\nPlease check logs for details."
fi

echo -e "$MESSAGE" | mail -s "$SUBJECT" -a "From: $FROM_EMAIL" "$TO_EMAIL"

# Exit with the original exit status of the main command
exit $EXIT_STATUS
