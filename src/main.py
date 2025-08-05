# src/main.py

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import datetime
from pathlib import Path

from src.utils.config_parser import ConfigParser
from src.utils import set_seed
from src.utils.logger import get_logger, configure_app_logger
from src.datasets import get_eeg_dataset
from src.models import get_eeg_model
from src.trainers.source_trainer import SourceTrainer
from src.trainers.tta_trainer import TTATrainer
from src.utils.data_utils import custom_collate_fn

def main(config_path: str, resume: str = None):
    # 1. Load Configuration
    config_parser = ConfigParser(config_path)
    config = config_parser.config

    # Output directory setup
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = config['experiment']['name']
    results_base_dir = config['paths']['results_dir']
    exp_results_dir = Path(results_base_dir) / f"{exp_name}_{timestamp}"
    exp_results_dir.mkdir(parents=True, exist_ok=True)

    log_dir = exp_results_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    configure_app_logger(log_dir)
    main_logger = get_logger(__name__)
    main_logger.info(f"Experiment results will be saved to: {exp_results_dir}")
    config_parser.save_config(exp_results_dir / "config.yaml")

    exp_checkpoint_dir = exp_results_dir / "checkpoints"
    exp_checkpoint_dir.mkdir(exist_ok=True)

    # 2. Set Device and Seed
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    set_seed(config['training']['seed'])
    main_logger.info(f"Using device: {device}")
    main_logger.info(f"Random seed set to: {config['training']['seed']}")

    # 3. Data Loading
    main_logger.info("Setting up data loaders...")
    dataset_name = config['dataset']['name']
    data_dir = config['dataset']['data_dir']
    preprocess_config = config['dataset']['preprocess']
    task_config = config['task']

    # Source Domain Data
    source_train_dataset = get_eeg_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        subject_ids=config['subject_setup']['source_train_subjects'],
        session_ids=config['subject_setup']['source_train_sessions'],
        task_config=task_config,
        preprocess_config=preprocess_config
    )
    source_val_dataset = get_eeg_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        subject_ids=config['subject_setup']['source_val_subjects'],
        session_ids=config['subject_setup']['source_val_sessions'],
        task_config=task_config,
        preprocess_config=preprocess_config
    )
    source_train_loader = DataLoader(
        dataset=source_train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        collate_fn=custom_collate_fn,
        drop_last=True
    )
    source_val_loader = DataLoader(
        dataset=source_val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False, num_workers=config['training']['num_workers'],
        collate_fn=custom_collate_fn
    )
    
    # Target Domain Data
    target_test_dataset = get_eeg_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        subject_ids=config['subject_setup']['target_test_subjects'],
        session_ids=config['subject_setup']['target_test_sessions'],
        task_config=task_config,
        preprocess_config=preprocess_config
    )
    target_test_loader = DataLoader(
        dataset=target_test_dataset,
        batch_size=config['tta'].get('test_batch_size', 32),
        shuffle=False, num_workers=config['training']['num_workers'],
        collate_fn=custom_collate_fn
    )

    # 4. Model Building
    main_logger.info("Initializing model...")
    model = get_eeg_model(config, preprocess_config).to(device)

    # 5. Source Domain Pre-training or Checkpoint Loading
    # The --resume CLI argument takes precedence over the config file path.
    checkpoint_to_load = resume or config['training'].get('load_pretrained_checkpoint_path')
    train_source_model = False

    if checkpoint_to_load:
        if os.path.exists(checkpoint_to_load):
            main_logger.info(f"Loading pre-trained model from: {checkpoint_to_load}")
            try:
                model.load_state_dict(torch.load(checkpoint_to_load, map_location=device))
                main_logger.info("Model state dictionary loaded successfully.")
            except Exception as e:
                main_logger.error(f"Failed to load checkpoint '{checkpoint_to_load}': {e}. Proceeding with source training.")
                train_source_model = True
        else:
            main_logger.warning(f"Checkpoint path not found: '{checkpoint_to_load}'. Proceeding with source training.")
            train_source_model = True
    else:
        main_logger.info("No pre-trained checkpoint specified. Starting source domain pre-training.")
        train_source_model = True

    if train_source_model:
        source_trainer = SourceTrainer(model, source_train_loader, source_val_loader, config, exp_checkpoint_dir, device)
        source_trainer.train()
        main_logger.info("Source domain pre-training complete. Best model loaded for TTA.")

    # 6. Test-Time Adaptation (TTA) and Evaluation
    if config['tta']['enable']:
        main_logger.info("Starting Test-Time Adaptation (TTA) and evaluation phase.")
        tta_trainer = TTATrainer(model, target_test_loader, config, exp_results_dir, device)
        tta_results = tta_trainer.adapt_and_evaluate()
        main_logger.info(f"Final TTA results for target subject(s) {config['subject_setup']['target_test_subjects']}: {tta_results}")
    else:
        main_logger.info("TTA is disabled. Skipping adaptation phase.")
    
    main_logger.info("Experiment finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ADAPT-EEG Main Experiment Script")
    parser.add_argument("--config", type=str, required=True, help="Path to the experiment configuration YAML file.")
    parser.add_argument('--resume', type=str, default=None, help="Optional path to a checkpoint file to resume training from.")
    args = parser.parse_args()
    main(args.config, args.resume)