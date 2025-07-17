# src/utils/config_parser.py

import yaml
from pathlib import Path
import os

class ConfigParser:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found at: {config_path}")
        initial_config = self._load_single_file(self.config_path)
        self.config = self._recursively_process_dict(initial_config, self.config_path.parent)

    def _deep_merge_dicts(self, d1: dict, d2: dict) -> dict:
        """Recursively merge the content of d2 into d1."""
        for key, value in d2.items():
            if key in d1 and isinstance(d1[key], dict) and isinstance(value, dict):
                d1[key] = self._deep_merge_dicts(d1[key], value)
            else:
                d1[key] = value
        return d1
    
    def _load_single_file(self, file_path: Path) -> dict:
        """Loads one YAML file only, without handling '_include' directives."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
        
    def _recursively_process_dict(self, current_dict: dict, base_path: Path) -> dict:
        """Recursively handles '_include' directives."""
        processed_dict = {}
        included_content_to_merge = {}
        keys_to_del = []

        for key, value in current_dict.items():
            if isinstance(key, str) and key.startswith('_include'):
                include_file_relative_path = value
                include_path = base_path / include_file_relative_path

                if not include_path.exists():
                    raise FileNotFoundError(f"Included config file not found: {include_path} (from {base_path})")
                
                # Recutsively load the content of the file
                included_raw_content = self._load_single_file(include_path)
                processed_included_content = self._recursively_process_dict(included_raw_content, include_path.parent)

                # Merge the content to included_content_to_merge
                self._deep_merge_dicts(included_content_to_merge, processed_included_content)
                keys_to_del.append(key)

            elif isinstance(value, dict):
                processed_dict[key] = self._recursively_process_dict(value, base_path)
            else:
                processed_dict[key] = value

        self._deep_merge_dicts(processed_dict, included_content_to_merge)
        
        for key in keys_to_del:
            if key in processed_dict:
                del processed_dict[key]

        return processed_dict

    def save_config(self, save_path: str):
        """Save the loaded configuration to a new YAML file."""
        save_file_path = Path(save_path)
        save_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_file_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(self.config, f, default_flow_style=False, sort_keys=False) # sort_keys=False to preserve order

