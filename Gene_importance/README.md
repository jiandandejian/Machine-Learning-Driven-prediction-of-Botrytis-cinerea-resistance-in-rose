# SNP Importance Analysis Tool

A Python tool for calculating Single Nucleotide Polymorphism (SNP) importance coefficients using iterative deletion method. This tool evaluates the contribution of individual SNPs to machine learning model performance by systematically removing each SNP and measuring the impact on prediction accuracy.

## Features

- **Multiple ML Models**: Support for 8 different machine learning algorithms
- **Comprehensive Metrics**: Calculates both accuracy difference and cross-entropy for importance assessment
- **Flexible Configuration**: Customizable model parameters via JSON input
- **Progress Tracking**: Real-time progress bars and detailed logging
- **Data Validation**: Comprehensive input data validation and error handling
- **Efficient Processing**: Optimized for large-scale SNP datasets

## Supported Models

- **MLPClassifier**: Multi-layer Perceptron Neural Network
- **RandomForestClassifier**: Random Forest ensemble method
- **LogisticRegression**: Linear logistic regression
- **LGBMClassifier**: Light Gradient Boosting Machine
- **XGBClassifier**: Extreme Gradient Boosting
- **DecisionTreeClassifier**: Decision tree algorithm
- **SVC**: Support Vector Classifier
- **SGDClassifier**: Stochastic Gradient Descent

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python SNP_importance_Explain.py \
    --train data/train.csv \
    --test data/test.csv \
    --output results/snp_importance.csv
```

### Advanced Usage with Custom Parameters

```bash
python SNP_importance_Explain.py \
    --train data/train.csv \
    --test data/test.csv \
    --output results/snp_importance.csv \
    --model_type RandomForestClassifier \
    --model_param '{"n_estimators": 200, "max_depth": 10, "random_state": 42}' \
    --n_jobs 4
```

### Command Line Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--train` | str | Yes | - | Path to training data CSV file |
| `--test` | str | Yes | - | Path to test data CSV file |
| `--output` | str | Yes | - | Output file path for results |
| `--model_type` | str | No | MLPClassifier | ML model to use |
| `--model_param` | str | No | - | JSON string of model parameters |
| `--n_jobs` | int | No | -1 | Number of parallel jobs |

## Input Data Format

### CSV File Structure

Both training and test CSV files should have the following structure:

```
,SNP_chr1_12345,SNP_chr1_67890,SNP_chr2_11111,...,Label
sample_1,0,1,2,...,R
sample_2,1,0,1,...,S
sample_3,2,2,0,...,R
...
```

### Requirements

- **Index Column**: First column should contain sample IDs
- **SNP Columns**: Each SNP should be named (e.g., `SNP_chr1_12345`)
- **Label Column**: Last column should contain binary labels (`R` for resistant, `S` for susceptible)
- **Data Types**: SNP values should be integers (0, 1, 2 for genotypes)
- **No Missing Values**: All cells must contain valid data

## Output Format

The tool generates a CSV file with the following columns:

| Column | Description |
|--------|-------------|
| `SNP` | SNP identifier |
| `Accuracy_Difference` | Baseline accuracy - accuracy without this SNP |
| `Cross_Entropy` | Cross-entropy between baseline and modified predictions |
| `Model_Type` | Machine learning model used |
| `Model_Parameters` | Model configuration parameters |
| `Train_Samples` | Number of training samples |
| `Test_Samples` | Number of test samples |
| `Calculation_Time` | Total processing time in seconds |

### Interpretation

- **Higher `Accuracy_Difference`**: More important SNP (removing it causes larger accuracy drop)
- **Higher `Cross_Entropy`**: SNP causes more divergence in prediction patterns
- **Positive values**: SNP contributes positively to model performance
- **Negative values**: SNP may be redundant or harmful to model performance


## Performance Considerations

### Computational Complexity

- **Time Complexity**: O(n × m) where n = number of SNPs, m = model training time
- **Memory Usage**: Depends on dataset size and model complexity
- **Recommended**: Use parallel processing (`--n_jobs`) for large datasets

### Optimization Tips

1. **Model Selection**: Faster models (LogisticRegression, SGDClassifier) for initial exploration
2. **Parameter Tuning**: Use cross-validation to optimize model parameters first
3. **Data Preprocessing**: Ensure clean, well-formatted input data
4. **Resource Management**: Monitor memory usage with large datasets

## Troubleshooting

### Common Issues

**File Not Found Error**
```bash
FileNotFoundError: File not found: data/train.csv
```
- Check file paths are correct
- Ensure files exist and are accessible

**Data Validation Error**
```bash
ValueError: Train and test datasets have different number of features
```
- Verify both files have identical SNP columns
- Check for missing or extra columns

**Memory Error**
```bash
MemoryError: Unable to allocate array
```
- Reduce dataset size or use more memory-efficient models
- Consider data sampling or feature selection

**JSON Parameter Error**
```bash
JSONDecodeError: Expecting property name enclosed in double quotes
```
- Use proper JSON format with double quotes
- Example: `'{"param": "value"}'` not `"{'param': 'value'}"`

### Best Practices

1. **Data Quality**: Clean and validate data before analysis
2. **Model Selection**: Choose appropriate model for your dataset size and complexity
3. **Parameter Tuning**: Optimize model parameters for better baseline performance
4. **Resource Planning**: Estimate computation time and memory requirements
5. **Result Validation**: Cross-check results with biological knowledge

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review example usage
3. Open an issue on GitHub with detailed error messages and data format

