# src/datasets/__init__.py
from .SEED_dataset import SEED

def get_dataset(dataset_name, **kwargs):
    if dataset_name == 'SEED':
        return SEED(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

