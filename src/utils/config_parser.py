# src/utils/config_parser.py

import yaml
from pathlib import Path

def load_config(config_path: str) -> dict:
    """
    加载并解析 YAML 配置文件。

    Args:
        config_path (str): 配置文件的路径。

    Returns:
        dict: 解析后的配置字典。
    """
    file_path = Path(config_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config: dict, config_path: str):
    """
    保存配置字典到 YAML 文件。

    Args:
        config (dict): 要保存的配置字典。
        config_path (str): 保存配置文件的路径。
    """
    file_path = Path(config_path)
    file_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, default_flow_style=False)

