# src/main.py

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import datetime

from src.utils.config_parser import ConfigParser
from src.utils import set_seed
from src.utils.logger import get_logger, configure_app_logger
from src.datasets import get_eeg_dataset
from src.models import get_eeg_model
from src.trainers.source_trainer import SourceTrainer
from src.trainers.tta_trainer import TTATrainer
from src.utils.data_utils import custom_collate_fn

def main(config_path: str):
    # 1. Load Configuration
    config_parser = ConfigParser(config_path)
    config = config_parser.config

    # Output directory setup
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = config['experiment']['name']
    results_base_dir = config['paths']['results_dir']
    exp_results_dir = os.path.join(results_base_dir, f"{exp_name}_{timestamp}")
    os.makedirs(exp_results_dir, exist_ok=True)

    log_dir = os.path.join(exp_results_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    configure_app_logger(log_dir)
    main_logger = get_logger(__name__)
    main_logger.info(f"Experiment results will be saved to: {exp_results_dir}")
    config_parser.save_config(os.path.join(exp_results_dir, "config.yaml")) # Save active config

    # Checkpoint directory
    exp_checkpoint_dir = os.path.join(exp_results_dir, "checkpoints")
    os.makedirs(exp_checkpoint_dir, exist_ok=True)

    # 2. Set Device and Seed
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    set_seed(config['training']['seed'])
    main_logger.info(f"Using device: {device}")
    main_logger.info(f"Random seed set to: {config['training']['seed']}")

    # 3. Data Loading
    main_logger.info("Setting up source domain data loaders for pre-training.")
    
    # Dataset specific parameters from config
    dataset_name = config['dataset']['name']
    data_dir = config['dataset']['data_dir']
    preprocess_config = config['dataset']['preprocess']
    task_config = config['task']

    # Source Domain (Pre-training) Data
    source_train_dataset = get_eeg_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        subject_ids=config['dataset'].get('source_train_subjects'),
        session_ids=config['dataset'].get('source_train_sessions'),
        task_config=task_config,
        preprocess_config=preprocess_config
    )
    source_val_dataset = get_eeg_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        subject_ids=config['dataset'].get('source_val_subjects'),
        session_ids=config['dataset'].get('source_val_sessions'),
        task_config=task_config,
        preprocess_config=preprocess_config
    )

    source_train_loader = DataLoader(source_train_dataset, 
                                     batch_size=config['training']['batch_size'], 
                                     shuffle=True, num_workers=config['training'].get('num_workers', 0),
                                     drop_last=True,
                                     collate_fn=custom_collate_fn)
    source_val_loader = DataLoader(source_val_dataset, 
                                   batch_size=config['training']['batch_size'], 
                                   shuffle=False, num_workers=config['training'].get('num_workers', 0),
                                   drop_last=False,
                                   collate_fn=custom_collate_fn)
    
    # Target Domain (Test/Adaptation) Data - For simplified LOSO-like single run
    main_logger.info("Setting up target domain data loader for TTA and evaluation.")
    target_test_dataset = get_eeg_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        subject_ids=config['dataset'].get('target_test_subjects'),
        session_ids=config['dataset'].get('target_test_sessions'),
        task_config=task_config,
        preprocess_config=preprocess_config
    )
    target_test_loader = DataLoader(target_test_dataset,
                                    batch_size=config['tta'].get('test_batch_size', 32),
                                    shuffle=False, num_workers=config['training'].get('num_workers', 0),
                                    drop_last=False,
                                    collate_fn=custom_collate_fn)


    # 4. Model Building
    main_logger.info("Initializing model...")
    model = get_eeg_model(config['model'], preprocess_config).to(device)
    # main_logger.info(f"Model architecture:\n{model}")

    # 5. Source Domain Pre-training or Checkpoint Loading
    checkpoint_to_load = config['training'].get('load_pretrained_checkpoint_path', None)
    train_source_model = False

    if checkpoint_to_load:
        if os.path.exists(checkpoint_to_load):
            main_logger.info(f"Loading pre-trained model from: {checkpoint_to_load}")
            try:
                checkpoint = torch.load(checkpoint_to_load, map_location=device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.eval()
                    main_logger.info("Model state dictionary loaded successfully. Model set to eval mode.")
                elif isinstance(checkpoint, nn.Module):
                    model = checkpoint.to(device)
                    model.eval()
                    main_logger.info("Entire model object loaded successfully. Model set to eval mode.")
                else:
                    main_logger.warning("Checkpoint structure not recognized. Expected 'model_state_dict' or a full model. Proceeding with source domain pre-training.")
                    train_source_model = True
            except Exception as e:
                main_logger.error(f"Failed to load checkpoint '{checkpoint_to_load}': {e}. Proceeding with source domain pre-training.")
                train_source_model = True
        else:
            main_logger.warning(f"Pre-trained checkpoint path not found: '{checkpoint_to_load}'. Proceeding with source domain pre-training.")
            train_source_model = True
    else:
        main_logger.info("No pre-trained checkpoint specified. Starting source domain pre-training phase.")
        train_source_model = True

    if train_source_model:
        source_trainer = SourceTrainer(
            model=model,
            train_loader=source_train_loader,
            val_loader=source_val_loader,
            config=config,
            checkpoint_dir=exp_checkpoint_dir,
            device=device,
        )
        # The source_trainer will save the best model and load it back into `model` internally
        source_trainer.train()
        main_logger.info("Source domain pre-training complete. Best model loaded.")

    # 6. Test-Time Adaptation (TTA) and Evaluation
    main_logger.info("Starting Test-Time Adaptation (TTA) and evaluation phase.")
    tta_trainer = TTATrainer(
        model=model,
        test_loader=target_test_loader,
        config=config,
        results_dir=results_base_dir,
        device=device
    )
    
    # For a single run, target_test_loader represents one target domain (e.g., Subject 15)
    tta_results = tta_trainer.adapt_and_evaluate()
    
    main_logger.info(f"Final TTA results for target subject(s) {config['dataset']['target_test_subjects']}: {tta_results}")

    # Optionally save results to a CSV
    # if not os.path.exists(os.path.join(exp_results_dir, 'metrics.csv')):
    #    with open(os.path.join(exp_results_dir, 'metrics.csv'), 'w') as f:
    #        f.write(','.join(tta_results.keys()) + '\n')
    # with open(os.path.join(exp_results_dir, 'metrics.csv'), 'a') as f:
    #     f.write(','.join(map(str, tta_results.values())) + '\n')
    
    main_logger.info("Experiment finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ADAPT-EEG Main Experiment Script")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the experiment configuration YAML file")
    args = parser.parse_args()

    main(args.config)
