# Deep-Leaning-Architecture-Selection
A bayesian evidence based method to find a valid Temporal convolutional Network (TCN) architecture for Blood Glucose prediction. This repository contains research code, reference papers, manuscript drafts, and presentation materials related to diabetes modeling and Temporal Convolutional Networks (TCNs).

## Repository Structure

```text
Deep-Leaning-Architecture-Selection/
├── codes/
│   ├── Bayesian_LSTM/
│   ├── Bayesian_TCN/
│   ├── Bergman_glucose_model/
│   └── Optuna_search/
├── papers/
│   ├── Blood_glucose_prediction/
│   ├── ECG_data/
│   ├── Machine_learning_models/
│   ├── architecture_search/
│   ├── enc_decoder/
│   └── tcn/
├── reports/
│   ├── manuscript_evident/
│   └── neuralnetworks_submit_6_26/
└── slides/
    ├── Azharul_proposal_defense/
    └── review_bgpredict2021_paper/
```

## What Is In Each Folder?

### `codes/`
Main research code and model-related files.

- `codes/Bayesian_TCN/`: Python scripts for Bayesian TCN population training and validation.
- `codes/Bayesian_LSTM/`: Bayesian LSTM experiment on ECG data.
- `codes/Bergman_glucose_model/`: Bergman glucose model files, including a MATLAB script and CSV data.
- `codes/Optuna_search/`: Optuna-based hyperparameter search for TCNs on the OhioT1DM dataset.

### `papers/`
Reference papers used for background reading and research support.

- `papers/Blood_glucose_prediction/`: Papers on blood glucose forecasting, probabilistic constraints, and uncertainty quantification.
- `papers/ECG_data/`: Papers on ECG signals and normal-value reference ranges.
- `papers/Machine_learning_models/`: Papers on forecasting architectures (transformers, N-BEATS, Autoformer) and Bayesian/bandit-based hyperparameter optimization.
- `papers/architecture_search/`: Papers on neural architecture search, random search, and marginal-likelihood model selection.
- `papers/enc_decoder/`: Papers on encoder-decoder models, segmentation, and sequence modeling.
- `papers/tcn/`: Papers focused on TCNs, Bayesian TCNs, and related surrogate/control modeling topics.

### `reports/`
Manuscript drafts and journal submission materials.

- `reports/manuscript_evident/`: LaTeX source, figures, references, and compiled PDF for the Bayesian TCN population validation manuscript.
- `reports/neuralnetworks_submit_6_26/`: Cover letter and highlights prepared for the Neural Networks journal submission.

### `slides/`
Presentation materials, figures, and LaTeX slide sources.

- `slides/Azharul_proposal_defense/`: Proposal defense slides, figures, and slide template files.
- `slides/review_bgpredict2021_paper/`: Review presentation slides, PDFs, figures, and LaTeX source files.

## Summary

- `codes/` stores the implementation work.
- `papers/` stores the reading material.
- `reports/` stores the manuscript and submission documents.
- `slides/` stores presentation content.
