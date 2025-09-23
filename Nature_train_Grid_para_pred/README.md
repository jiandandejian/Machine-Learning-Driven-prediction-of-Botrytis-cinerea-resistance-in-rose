# Nature Grid Search F1 Test

A comprehensive machine learning evaluation framework for binary trait classification using GWAS SNP data. This script trains multiple machine learning models on a natural population dataset and evaluates their performance on an F1 population test set.


## Overview

This framework implements a train-test evaluation approach for genomic trait prediction using five different machine learning algorithms. The system trains models on a natural population dataset and evaluates their generalization performance on an independent F1 population dataset.

### Key Features

- **Multiple ML Algorithms**: Implements 5 state-of-the-art classifiers (Random Forest, SVM, LightGBM, Logistic Regression, MLP)
- **Comprehensive Hyperparameter Tuning**: Extensive grid search optimization for each model
- **Cross-Population Validation**: Trains on natural population, tests on F1 population
- **Multiple Performance Metrics**: Evaluates models using accuracy, ROC-AUC, precision, recall, and F1-score
- **Reproducible Results**: Consistent random states across all models
- **Detailed Output**: Comprehensive results including best parameters and predictions

## Requirements

### System Requirements
- Python 3.8 or higher
- RAM: Minimum 8GB (16GB+ recommended for large datasets)
- CPU: Multi-core processor recommended for parallel processing

### Data Requirements
- **Training Data**: `nature_GWAS_SNP_RS.tsv` - Natural population SNP data
- **Test Data**: `F1_GWAS_SNP_RS.tsv` - F1 population SNP data

## Installation

```bash
# Clone or download the script
# Ensure you have the required dependencies
pip install -r requirements.txt
```

## Data Format

Both input files should follow this format:

| sample_id | SNP_1 | SNP_2 | ... | SNP_n | rank |
|-----------|-------|-------|-----|-------|------|
| sample1   | 0     | 1     | ... | 2     | R    |
| sample2   | 1     | 0     | ... | 1     | S    |

**Format Requirements:**
- **First Column**: Sample IDs (used as index)
- **Middle Columns**: SNP genotypes (0, 1, 2 for homozygous/heterozygous, NaN for missing)
- **Last Column**: Phenotype labeled 'rank' ('R' for resistant, 'S' for susceptible)
- **File Format**: Tab-separated values (.tsv)

## Usage

### Basic Usage

```bash
python Nature_Grid_Search_F1_test.py
```

### Expected Runtime
- **Small datasets** (< 1000 samples, < 10k SNPs): 10-30 minutes
- **Medium datasets** (1000-5000 samples, 10k-50k SNPs): 1-4 hours
- **Large datasets** (> 5000 samples, > 50k SNPs): 4+ hours

**Note**: Runtime depends heavily on dataset size and available computational resources.

## Algorithm Details

### 1. Random Forest Classifier (RFC)
- **Parameters**: n_estimators, max_depth, max_features, min_samples_split, min_samples_leaf
- **Strategy**: Ensemble learning with balanced class weights
- **Best for**: High-dimensional data with feature interactions

### 2. Support Vector Classifier (SVC)
- **Parameters**: C, kernel, gamma, degree, coef0
- **Strategy**: Multiple kernel types (linear, RBF, polynomial, sigmoid)
- **Best for**: High-dimensional data with clear margins

### 3. LightGBM Classifier (LGBMC)
- **Parameters**: n_estimators, max_depth, min_child_samples, colsample_bytree
- **Strategy**: Gradient boosting with efficient memory usage
- **Best for**: Large datasets with mixed data types

### 4. Logistic Regression (LR)
- **Parameters**: C, penalty, solver, class_weight
- **Strategy**: Linear classification with L1/L2 regularization
- **Best for**: Interpretable linear relationships

### 5. Multi-Layer Perceptron (MLP)
- **Parameters**: hidden_layer_sizes, activation, solver, learning_rate
- **Strategy**: Neural network with early stopping
- **Best for**: Complex non-linear patterns

## Performance Metrics

The framework evaluates each model using:

- **Accuracy**: Overall classification correctness
- **ROC-AUC**: Area under the receiver operating characteristic curve
- **Precision**: Positive predictive value (true positives / predicted positives)
- **Recall**: Sensitivity (true positives / actual positives)
- **F1-Score**: Harmonic mean of precision and recall

## Output Files

### Nature_Grid_Search_F1_test_Results.txt
Contains comprehensive results for all models including:
- Best hyperparameters found during grid search
- Predicted labels for test set
- Predicted probabilities for test set
- All performance metrics with 5 decimal precision

**Example Output Structure:**
```
=== Random Forest ===
RFC_Best_Params: {'bootstrap': True, 'class_weight': 'balanced', ...}
RFC_y_pred: [0 1 0 1 ...]
RFC_y_prob: [0.23 0.78 0.12 ...]
RFC_Accuracy: 0.85000
RFC_ROC_AUC: 0.89000
...
```

## Customization

### Modifying Hyperparameter Grids

Each model has a configurable parameter grid:

```python
# Example for Random Forest
RFC_param_grid = {
    'n_estimators': [50, 100, 200, 500],  # Add more values
    'max_depth': [5, 10, 15, 20],         # Extend range
    'max_features': ['sqrt', 'log2', 0.3], # Add float values
}
```

### Changing Cross-Validation Settings

```python
# Modify the StratifiedKFold parameters
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Reduce folds for speed
```

### Adding New Models

To add a new classifier:
1. Define parameter grid
2. Create GridSearchCV instance
3. Fit and predict
4. Calculate metrics
5. Add to output section

## Troubleshooting

### Common Issues

**Issue**: `FileNotFoundError` for input files
**Solution**: Ensure both `.tsv` files are in the same directory as the script

**Issue**: `Memory Error` during grid search
**Solution**: 
- Reduce parameter grid size
- Decrease `n_jobs` parameter
- Use smaller datasets for testing

**Issue**: `Convergence warnings` for Logistic Regression
**Solution**: Increase `max_iter` values in the parameter grid

**Issue**: Long runtime
**Solution**:
- Reduce parameter grid size
- Use fewer cross-validation folds
- Set `n_jobs=1` if memory is limited

**Issue**: Poor model performance
**Solution**:
- Check data quality and preprocessing
- Verify phenotype encoding (R/S mapping)
- Consider feature selection or dimensionality reduction

### Performance Optimization

```python
# For faster execution (reduced accuracy)
n_splits = 5  # Instead of 10
n_jobs = 4    # Match your CPU cores
# Reduce parameter grid sizes
```

## Computational Considerations

### Memory Usage
- **Minimum**: 4GB RAM for small datasets
- **Recommended**: 16GB+ RAM for typical GWAS datasets
- **Memory scales with**: Number of samples × Number of SNPs

### Parallel Processing
- Uses `n_jobs=-1` for maximum CPU utilization
- Can be adjusted based on available cores
- Memory usage increases with parallel jobs

## Citation

If you use this code in your research, please cite:

```bibtex
@software{nature_grid_search_2025,
  title={Nature Grid Search F1 Test: Machine Learning Framework for GWAS Classification},
  author={[Author Names]},
  year={2025},
  url={[Repository URL]},
  version={1.0.0}
}
```

## Related Resources

- [Scikit-learn Grid Search Documentation](https://scikit-learn.org/stable/modules/grid_search.html)
- [LightGBM Parameter Tuning](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html)
- [Cross-validation Best Practices](https://scikit-learn.org/stable/modules/cross_validation.html)

---

**Last Updated**: June 2025  
**Status**: Active Development