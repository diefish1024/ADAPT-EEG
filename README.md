# ADAPT-EEG: Adaptive Decoding of Affective States from EEG

## Project Overview

`ADAPT-EEG` enhances BCI emotion recognition by addressing **EEG signal variability** (inter-subject, inter-session) using **Test-Time Adaptation (TTA)**. It aims to improve model generalization and robustness across classification and dimensional regression tasks.

## Core Technical Approach

### Base Model Architecture
Uses **ResNet-18** as feature extractor with **pluggable heads** (`src/models/heads/`) for classification (discrete), standard regression (continuous), and **uncertainty regression** (mean & variance outputs).

### Test-Time Adaptation (TTA) Paradigms
`ADAPT-EEG` develops TTA strategies to mitigate domain shift in EEG signals.

#### A. K-Classification Task on SEED Dataset
Evaluates **mainstream TTA algorithms** (e.g., **BN Adaptation**, **Tent**, **CoTTA**, **SHOT**, **MEMO**) using unlabeled target domain data.

#### B. Dimensional Emotion Regression Task on DEAP Dataset
Focuses on **innovative TTA** combining uncertainty estimation and latent space domain invariant learning.

1.  **Uncertainty-Aware TTA**: Uses `uncertainty_regression_head` for mean and variance prediction. Trained with **Negative Log-Likelihood (NLL) loss**. TTA applies **uncertainty-weighted consistency regularization**, **uncertainty minimization**, and **uncertainty-weighted pseudo-label regression** on unlabeled data.
2.  **Latent Space Invariance**: Explores **Domain Adversarial Neural Networks (DANN)** or **distance-based methods (MMD/CORAL)** for domain alignment, and **contrastive learning (InfoNCE Loss)** for robust feature representation during TTA.

## Expected Project Structure

The project will feature a highly modular and configurable software architecture:

```
ADAPT-EEG/
├── data/               # Raw and preprocessed dataset storage
├── configs/            # YAML configurations for experiments (dataset, model, task, TTA method, experiment templates)
├── src/                # Core source code
│   ├── datasets/       # Data loading and preprocessing modules
│   ├── models/         # Model definitions (feature_extractors, heads)
│   │   ├── feature_extractors/
│   │   └── heads/
│   ├── losses/         # Loss functions
│   ├── tta_methods/    # Implementations of various TTA algorithms
│   ├── trainers/       # Trainer logic
│   └── utils/          # Utility functions
├── scripts/            # Experiment execution scripts
└── results/            # Experiment results, logs, checkpoints, and analysis plots
```

## Expected Outcomes

`ADAPT-EEG` aims to establish a robust, flexible research framework for BCI emotion recognition. By addressing EEG domain shift through advanced TTA, it seeks to significantly enhance the practical performance and robustness of BCI emotion recognition systems.

## Getting Started

*(Further instructions on installation, dataset preparation, and running experiments will be provided here upon project advancement.)*

## Usage

*(Examples of how to train models, apply TTA, and evaluate performance will be provided here.)*
