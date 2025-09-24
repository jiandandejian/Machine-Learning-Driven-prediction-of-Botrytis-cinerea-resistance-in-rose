# SNP Analysis Pipeline with Cross-Entropy Feature Selection

This repository contains a comprehensive machine learning pipeline for analyzing Single Nucleotide Polymorphisms (SNPs) using cross-entropy divergence for feature selection and multiple classification algorithms for prediction.

## Overview

The pipeline performs the following key steps:
1. Calculates cross-entropy divergence between SNP distributions in different datasets
2. Selects informative SNPs based on percentile thresholds
3. Trains and evaluates multiple machine learning models
4. Generates detailed performance reports for each model and threshold combination

## File Structure

```
├── Cross_Entropy_model_train_test.py    # Core functions for training and evaluation
├── Cross_Entropy_model_SNP_Results.py   # SNP selection and results summary
├── CR_LR_model_train_test.py           # Logistic Regression implementation
├── CR_RFC_model_train_test.py           # Random Forest implementation
├── CR_SVC_model_train_test.py          # Support Vector Classifier implementation
├── CR_LGBMC_model_train_test.py        # LightGBM implementation
├── CR_MLPC_model_train_test.py         # Multi-layer Perceptron implementation
└── README.md                           # This file
```

## Required Data Files

The pipeline expects the following input files in the same directory:
- `nature_GWAS_SNP_RS.tsv`: Training dataset with SNP data
- `F1_GWAS_SNP_RS.tsv`: Test dataset with SNP data

**Data Format**: Tab-separated files with SNP IDs as columns and samples as rows. The last column should contain binary labels ('R' for resistant, 'S' for susceptible).

## Dependencies

```python
pandas
numpy
scikit-learn
lightgbm
matplotlib
seaborn
scipy
collections
itertools
```

Install dependencies:
```bash
pip install pandas numpy scikit-learn lightgbm matplotlib seaborn scipy
```

## Core Components

### 1. Cross_Entropy_model_train_test.py
Contains the main utility functions:
- `calculate_prob_distributions()`: Computes probability distributions for SNP genotypes
- `calculate_cross_entropy()`: Calculates cross-entropy divergence between datasets
- `select_snps_by_percentile()`: Selects top SNPs based on divergence percentiles
- `train_and_evaluate()`: Trains models and computes evaluation metrics

### 2. Cross_Entropy_model_SNP_Results.py
Analyzes SNP characteristics:
- Identifies SNPs with maximum and minimum cross-entropy divergence
- Generates summary of selected SNPs by percentile thresholds
- Outputs results to `Cross_Entropy_model_SNP_Results.txt`

### 3. Model-Specific Scripts
Each model script follows the same pattern:
- **CR_LR_model_train_test.py**: Logistic Regression with L1/L2 regularization
- **CR_RFC_model_train_test.py**: Random Forest with various tree parameters
- **CR_SVC_model_train_test.py**: Support Vector Classifier with multiple kernels
- **CR_LGBMC_model_train_test.py**: LightGBM with gradient boosting parameters
- **CR_MLPC_model_train_test.py**: Multi-layer Perceptron neural network

## Usage

### Step 1: Analyze SNP Characteristics
```bash
python Cross_Entropy_model_SNP_Results.py
```
This generates `Cross_Entropy_model_SNP_Results.txt` with SNP selection summary.

### Step 2: Run Individual Model Training
Execute any of the model-specific scripts:
```bash
python CR_LR_model_train_test.py     # Logistic Regression
python CR_RFC_model_train_test.py     # Random Forest
python CR_SVC_model_train_test.py    # Support Vector Classifier
python CR_LGBMC_model_train_test.py  # LightGBM
python CR_MLPC_model_train_test.py   # Neural Network
```

Each script will:
- Perform hyperparameter grid search
- Train models across multiple SNP selection thresholds
- Generate detailed results file (e.g., `CR_LR_model_detailed_results_by_Threshold.txt`)

## Methodology

### Cross-Entropy Feature Selection
The pipeline uses cross-entropy divergence to measure the difference between SNP genotype distributions in training and test datasets:

```
H(P,Q) = -Σ P(x) * log(Q(x))
```

Where P and Q are probability distributions of SNP genotypes in the two datasets.

### SNP Selection Thresholds
SNPs are selected based on 8 percentile thresholds: [12.5%, 25%, 37.5%, 50%, 62.5%, 75%, 87.5%, 100%]

### Model Evaluation
Each model is evaluated using:
- **ROC AUC**: Area under the ROC curve
- **Average Precision**: Area under the precision-recall curve
- **F1 Score**: Harmonic mean of precision and recall
- **Accuracy**: Overall classification accuracy
- **Confusion Matrix**: True/false positive and negative counts

## Hyperparameter Grids

### Logistic Regression
- Regularization: L1, L2
- C values: [0.001, 0.01, 0.1, 1, 10, 100]
- Solvers: liblinear, lbfgs
- Class weights: None, balanced

### Random Forest
- n_estimators: [50, 100, 200]
- max_depth: [5, 10, 15]
- max_features: ['sqrt', 'log2']
- Criteria: ['entropy', 'log_loss']

### Support Vector Classifier
- Kernels: ['linear', 'rbf', 'poly', 'sigmoid']
- C values: [0.1, 1, 10, 100]
- Gamma: ['scale', 'auto', 0.01, 0.1, 1]

### LightGBM
- n_estimators: [100, 200]
- max_depth: [-1, 10, 15]
- Boosting types: ['gbdt', 'dart']
- Learning rate: [0.01]

### Multi-layer Perceptron
- Hidden layers: [(128, 64), (128, 64, 32)]
- Solvers: ['adam', 'sgd']
- Learning rates: [0.001, 0.01]
- Alpha regularization: [0.0001, 0.001]

## Output Files

Each model generates a detailed results file containing:
- Best accuracy for each percentile threshold
- Complete hyperparameter configurations
- All evaluation metrics (ROC AUC, precision, recall, F1-score)
- Confusion matrices
- Prediction probabilities and classifications

## Key Features

- **Automated Hyperparameter Tuning**: Comprehensive grid search for each algorithm
- **Cross-Entropy Feature Selection**: Novel approach for SNP prioritization
- **Multiple Evaluation Metrics**: Comprehensive model assessment
- **Detailed Logging**: Complete results saved for analysis
- **Scalable Design**: Easy to add new models or modify parameters

## Notes

- All models use `random_state=42` for reproducibility
- SNPs are sorted by chromosomal position after selection
- Smoothing (1e-9) is applied to probability distributions to handle zero probabilities
- Binary classification assumes 'R' (resistant) = 0, 'S' (susceptible) = 1

## License

This project is licensed under the MIT License - see the LICENSE file for details.
