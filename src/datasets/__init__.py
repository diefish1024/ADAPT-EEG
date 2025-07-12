# src/datasets/__init__.py
from .seed_dataset import SEEDDataset

def get_dataset(dataset_name, **kwargs):
    if dataset_name == 'SEED':
        return SEEDDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

