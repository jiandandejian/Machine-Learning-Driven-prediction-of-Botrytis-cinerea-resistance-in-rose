# GWAS Kinship-based Machine Learning Framework

A comprehensive machine learning framework for binary trait classification using genome-wide association study (GWAS) SNP data with kinship-based sample selection and clustering.


## Overview

This framework implements a two-stage approach for genomic trait prediction:

1. **Kinship Analysis**: Calculates genetic distances between samples and performs clustering based on kinship relationships
2. **Machine Learning**: Trains and evaluates multiple ML models on kinship-filtered datasets for robust binary classification

### Key Features

- **Kinship-based Sample Selection**: Clusters samples using genetic distance thresholds to reduce population stratification
- **Multiple ML Algorithms**: Supports 5 state-of-the-art classifiers (LR, RFC, SVC, MLPC, LGBMC)
- **Automated Hyperparameter Optimization**: Grid search with stratified k-fold cross-validation
- **Comprehensive Evaluation**: Multiple performance metrics with detailed per-fold reporting
- **Reproducible Results**: Consistent random states and cross-validation procedures
- **Robust Validation**: 10-fold stratified cross-validation for reliable performance estimation

## Requirements

### System Requirements
- Python 3.8 or higher
- RAM: Minimum 8GB (16GB+ recommended for large datasets)
- Storage: ~1GB free space for outputs

### Python Dependencies
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
lightgbm>=3.2.0
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone [YOUR_REPOSITORY_URL]
cd gwas-kinship-ml

# Install dependencies
pip install -r requirements.txt
# or install manually:
pip install pandas numpy scikit-learn lightgbm
```

### Data Preparation

Ensure your input file `nature_GWAS_SNP_RS.tsv` follows this format:

| sample_id | SNP_1 | SNP_2 | ... | SNP_n | rank |
|-----------|-------|-------|-----|-------|------|
| sample1   | 0     | 1     | ... | 2     | R    |
| sample2   | 1     | 0     | ... | 1     | S    |

**Format Requirements:**
- **Column 1**: Sample IDs (unique identifiers)
- **Columns 2 to N-1**: SNP genotypes (0, 1, 2 for homozygous/heterozygous, NaN for missing)
- **Last Column**: Phenotype labeled 'rank' ('R' for resistant, 'S' for susceptible)
- **File Format**: Tab-separated values (.tsv)

### Basic Usage

```bash
# Run individual models
python Nature_GWAS_Kinship_Train_LGBMC.py
python Nature_GWAS_Kinship_Train_LR.py
python Nature_GWAS_Kinship_Train_RFC.py
python Nature_GWAS_Kinship_Train_SVC.py
python Nature_GWAS_Kinship_Train_MLPC.py

# Run all models (if you have a batch script)
chmod +x run_all_models.sh
./run_all_models.sh
```

## Pipeline Details

### Stage 1: Kinship Analysis
1. **Distance Calculation**: Computes pairwise genetic distances using SNP data
2. **Clustering**: Groups samples based on distance thresholds (0.3, 0.35, 0.4)
3. **Subset Selection**: Identifies optimal cluster size (100-139 samples) for downstream analysis
4. **Output Generation**: Creates filtered dataset for ML training

### Stage 2: Machine Learning
1. **Data Preprocessing**: Encodes phenotypes (R→0, S→1) and handles missing values
2. **Model Training**: 10-fold stratified cross-validation with hyperparameter tuning
3. **Performance Evaluation**: Calculates accuracy, ROC-AUC, precision, recall, and F1-score
4. **Results Logging**: Saves best parameters and performance metrics

## File Structure

```
project/
├── Nature_GWAS_Kinship_Train_LGBMC.py    # LightGBM classifier
├── Nature_GWAS_Kinship_Train_LR.py       # Logistic regression
├── Nature_GWAS_Kinship_Train_MLPC.py     # Multi-layer perceptron
├── Nature_GWAS_Kinship_Train_RFC.py      # Random forest
├── Nature_GWAS_Kinship_Train_SVC.py      # Support vector classifier
├── nature_GWAS_SNP_RS.tsv            # Input data (user-provided)
├── requirements.txt                       # Dependencies
└── README.md                             # This file
```

## Output Files

| File | Description |
|------|-------------|
| `Genetic_distance_lower_triangle.txt` | Pairwise genetic distance matrix (MEGA format) |
| `Genetic_distance_summary.csv` | Clustering summary for all thresholds |
| `Genetic_distance_summary_by_threshold.txt` | Detailed cluster composition |
| `Genetic_distance_genotype_phenotype_threshold_X_Y.tsv` | Filtered dataset for ML training |
| `out_result_Nature_GWAS_[MODEL]_Kinship_endadta.txt` | Model results and hyperparameters |

## Configuration

### Customizing Distance Thresholds
```python
# Modify in any script
thresholds = [0.25, 0.3, 0.35, 0.4, 0.45]  # Add/remove thresholds
```

### Adjusting Sample Size Criteria
```python
# Change the clustering size requirement
if 80 <= len(clustered_ids) < 120:  # Modified range
```

### Hyperparameter Tuning
Each model has configurable parameter grids. Example for Random Forest:
```python
RFC_param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
```

## Performance Metrics

The framework evaluates models using:
- **Accuracy**: Overall classification correctness
- **ROC-AUC**: Area under the receiver operating characteristic curve
- **Precision**: Positive predictive value
- **Recall**: Sensitivity/True positive rate
- **F1-Score**: Harmonic mean of precision and recall

## Troubleshooting

### Common Issues

**Issue**: `FileNotFoundError: nature_GWAS_SNP_RS.tsv`
**Solution**: Ensure the input file is in the same directory as the scripts

**Issue**: `Memory Error` during distance calculation
**Solution**: Reduce dataset size or increase available RAM

**Issue**: No clustering output generated
**Solution**: Check if any threshold yields 100-139 clustered samples; adjust criteria if needed

**Issue**: Poor model performance
**Solution**: 
- Check data quality and missing value patterns
- Verify phenotype encoding (R/S labels)
- Consider different distance thresholds

### Getting Help

1. Check the troubleshooting section above
2. Review input data format requirements
3. Open an issue on GitHub with:
   - Error message
   - Input data characteristics
   - System specifications

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Related Resources

- [GWAS Best Practices](https://www.nature.com/articles/s41586-018-0579-z)
- [Population Stratification in GWAS](https://www.nature.com/articles/nrg2813)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)

---

**Maintained by**: [Your Name/Organization]  
**Last Updated**: June 2025  
**Status**: Active Development
