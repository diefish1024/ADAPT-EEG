# src/utils/config_parser.py

import yaml
from pathlib import Path

def load_config(config_path: str) -> dict:
    """
    Load and parse YAML configuration files.

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        dict: The parsed configuration dictionary.
    """
    file_path = Path(config_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config: dict, config_path: str):
    """
    Save the configuration dictionary to a YAML file.

    Args:
        config (dict): The configuration dictionary to save.
        config_path (str): The path to save the configuration files.
    """
    file_path = Path(config_path)
    file_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, default_flow_style=False)

