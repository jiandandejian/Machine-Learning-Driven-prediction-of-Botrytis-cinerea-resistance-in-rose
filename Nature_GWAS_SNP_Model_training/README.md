# Rose Disease Classification

A comprehensive machine learning framework for binary trait classification in roses using Single Nucleotide Polymorphism (SNP) data from natural and F1 populations.

## Project Overview

This project implements a robust binary classification system for rose traits (e.g., disease resistance) by leveraging GWAS SNP sites shared between natural rose populations and F1 populations. The framework supports multiple machine learning algorithms with automated hyperparameter optimization and comprehensive evaluation metrics.

### Key Features

- **Genomic Data Processing**: Utilizes comprehensive SNP data from rose populations
- **Multiple ML Models**: Random Forest, SVM, LightGBM, Logistic Regression, and Neural Networks
- **Automated Optimization**: Grid search with stratified k-fold cross-validation
- **Comprehensive Evaluation**: Multiple performance metrics with detailed reporting
- **Reproducible Results**: Consistent random states and structured logging
- **Robust Validation**: 10-fold stratified cross-validation for reliable performance estimation

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Required packages (see [Installation](#installation))

### Installation

1. **Clone the repository:**
   ```bash
   git clone XXXXXXX
   cd XXXXXX
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install manually:
   ```bash
   pip install pandas>=1.3.0 numpy>=1.21.0 scikit-learn>=1.0.0 lightgbm
   ```

### Basic Usage

Run any classifier with:
```bash
python Nature_GWAS_Train_RFC.py    # Random Forest
python Nature_GWAS_Train_SVC.py   # Support Vector Classifier
python Nature_GWAS_Train_LGBMC.py # LightGBM
python Nature_GWAS_Train_LR.py    # Logistic Regression
python Nature_GWAS_Train_MLPC.py  # Neural Network
```

Each script will automatically:
1. Load the input data
2. Perform hyperparameter optimization
3. Execute 10-fold cross-validation
4. Generate comprehensive results

## Data Format

### Input File
The input file `nature_GWAS_SNP_RS.tsv` should be a tab-separated file with:

- **Rows**: Individual samples
- **Columns**: SNP features + classification column
- **Target Column**: `rank` with binary labels ('R' for Resistant, 'S' for Susceptible)

### Example Format
```tsv
| sample_id | SNP_1 | SNP_2 | ... | SNP_n | rank |
|-----------|-------|-------|-----|-------|------|
| sample1   | 0     | 1     | ... | 2     | R    |
| sample2   | 1     | 0     | ... | 1     | S    |
```

## Machine Learning Models

| Model | Key Parameters | Strengths |
|-------|---------------|-----------|
| **Random Forest** | `n_estimators`, `max_depth`, `min_samples_split` | Handles overfitting, feature importance |
| **SVM** | `C`, `kernel`, `gamma` | High-dimensional data, non-linear patterns |
| **LightGBM** | `n_estimators`, `max_depth`, `boosting_type` | Fast training, memory efficient |
| **Logistic Regression** | `C`, `penalty`, `solver` | Interpretable, probability estimates |
| **Neural Network** | `hidden_layer_sizes`, `learning_rate` | Complex pattern recognition |

### Hyperparameter Ranges

<details>
<summary>Click to expand hyperparameter details</summary>

#### Random Forest
- `n_estimators`: [50, 100, 200]
- `max_depth`: [5, 10, 15]
- `max_features`: ['sqrt', 'log2']
- `min_samples_split`: [5, 10]
- `min_samples_leaf`: [2, 4, 6]

#### Support Vector Machine
- `C`: [0.1, 1, 10, 100]
- `kernel`: ['linear', 'rbf', 'poly', 'sigmoid']
- `gamma`: ['scale', 'auto', 0.01, 0.1, 1]

#### LightGBM
- `n_estimators`: [100, 200]
- `max_depth`: [-1, 10, 15]
- `min_child_samples`: [20, 50]
- `boosting_type`: ['gbdt', 'dart']

#### Logistic Regression
- `C`: [0.001, 0.01, 0.1, 1, 10, 100]
- `penalty`: ['l1', 'l2']
- `solver`: ['liblinear', 'lbfgs']
- `max_iter`: [500, 1000, 2000, 3000]

#### Neural Network
- `hidden_layer_sizes`: [(128, 64), (128, 64, 32)]
- `activation`: ['relu']
- `solver`: ['adam', 'sgd']
- `learning_rate_init`: [0.001, 0.01]

</details>

## Performance Metrics

Each model is evaluated using:

- **Accuracy**: Overall classification correctness
- **ROC-AUC**: Area under the receiver operating characteristic curve
- **Precision**: Positive predictive value
- **Recall**: Sensitivity/True positive rate
- **F1-Score**: Harmonic mean of precision and recall

## Project Structure

```
rose-disease-classification/
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── LICENSE                             # MIT License
├── nature_GWAS_SNP_RS.tsv    # Input dataset
├── Nature_GWAS_Train_RFC.py             # Random Forest classifier
├── Nature_GWAS_Train_SVC.py            # Support Vector classifier
├── Nature_GWAS_Train_LGBMC.py          # LightGBM classifier
├── Nature_GWAS_Train_LR.py             # Logistic Regression classifier
├── Nature_GWAS_Train_MLPC.py           # Neural Network classifier
└── results/                           # Output directory
    ├── out_result_Nature_GWAS_RFC_endadta.txt
    ├── out_result_Nature_GWAS_SVC_endadta.txt
    ├── out_result_Nature_GWAS_LGBMC_endadta.txt
    ├── out_result_Nature_GWAS_LR_endadta.txt
    └── out_result_Nature_GWAS_MLPC_endadta.txt
```

## Results and Output

Each classifier generates a detailed results file containing:

- **Best hyperparameters** from grid search
- **Cross-validation metrics** for each fold
- **Average performance** across all folds
- **Model-specific insights** and recommendations

### Sample Output Structure
```
Best Parameters: {'n_estimators': 200, 'max_depth': 15, ...}

Cross-Validation Results:
Fold 1: Accuracy=0.85, ROC-AUC=0.88, F1=0.83
Fold 2: Accuracy=0.87, ROC-AUC=0.90, F1=0.85
...

Average Performance:
Accuracy: 0.86 ± 0.02
ROC-AUC: 0.89 ± 0.03
F1-Score: 0.84 ± 0.02
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

