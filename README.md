# Machine Learning Pipeline for Botrytis cinerea Resistance Prediction in Rose

A comprehensive machine learning analysis pipeline for predicting Botrytis cinerea resistance in rose, integrating SNP genotype data analysis, cross-population prediction, kinship correction, and key locus identification modules.

## Research Background

This study develops a systematic machine learning analysis pipeline for Botrytis cinerea resistance prediction in rose, comprising six core components: dataset construction, baseline model training, enhanced kinship-based model development, cross-population prediction, model performance evaluation, and key SNP locus identification. The pipeline effectively enables model transfer from genetically diverse natural populations to genetically narrow F1 populations, providing technical support for molecular breeding applications.

## Project Structure & Functional Modules

```
├── Nature_ALL_SNP_Model_training/           # Natural population full-SNP baseline model training
├── Nature_GWAS_SNP_Model_training/          # Natural population GWAS-filtered SNP model training
├── Nature_GWAS_SNP_Kinship_Model_training/  # Enhanced kinship-integrated model development
├── Nature_train_Grid_para_pred/             # Grid search optimization & cross-population prediction
├── CR_model_pred/                           # Cross-entropy optimized model prediction
├── KL_model_pred/                           # KL divergence optimized model prediction
└── Gene_importance/                         # Key SNP locus importance analysis
```

## Analysis Pipeline Details

### Module 1: Natural Population Full-SNP Baseline Model Training (`Nature_ALL_SNP_Model_training/`)
**Research Objective**: Establish baseline machine learning models using complete SNP datasets from natural populations
- **Input Data**: Rose natural population SNP genotype data + Botrytis cinerea resistance phenotype data
- **Modeling Strategy**: Utilize all available SNP loci for model training
- **Model Types**: Multiple algorithms including LR, RFC, SVC, LGBMC, MLPC
- **Evaluation Method**: 10-fold cross-validation
- **Output Results**: Baseline model performance benchmarks and comparative analysis

### Module 2: GWAS-Filtered SNP Model Training (`Nature_GWAS_SNP_Model_training/`)
**Research Objective**: Construct refined models using significant SNP loci identified through GWAS analysis
- **Input Data**: Key SNP subset after GWAS significance testing
- **Selection Criteria**: Statistical significance-based SNP locus filtering
- **Modeling Advantages**: Dimensionality reduction and noise interference mitigation
- **Comparative Analysis**: Performance comparison with full-SNP models
- **Output Results**: Optimized models after feature selection

### Module 3: Enhanced Kinship-Integrated Model Development (`Nature_GWAS_SNP_Kinship_Model_training/`)
**Research Objective**: Develop enhanced models incorporating genetic distance matrix K to account for population structure
- **Innovative Method**: Integration of kinship matrix to correct population stratification effects
- **Model Types**: LR_K, RFC_K, SVC_K, LGBMC_K, MLPC_K
- **Technical Features**: Effective control of false positive associations and improved model generalization
- **Application Value**: Better adaptation to genetic diversity in natural populations
- **Output Results**: Enhanced prediction models with kinship correction

### Module 4: Cross-Population Prediction Parameter Optimization (`Nature_train_Grid_para_pred/`)
**Research Objective**: Develop model transfer optimization strategies from natural populations to F1 populations
- **Core Technology**: Grid Search hyperparameter optimization
- **Transfer Strategy**: Model adaptation from high genetic diversity → narrow genetic background
- **Optimization Goal**: Enhance accuracy and stability of cross-population prediction
- **Validation Method**: Cross-validation and independent test set validation
- **Output Results**: Optimal parameter combinations and cross-population adapted models

### Module 5: Cross-Entropy Optimized Prediction (`CR_model_pred/`)
**Research Objective**: Model optimization based on cross-entropy quantification of SNP probability distribution differences
- **Theoretical Foundation**: Cross-entropy measures SNP distribution differences between populations
- **Selection Strategy**: SNP ranking and subset selection based on cross-entropy values
- **Application Scenario**: Feature selection optimization in cross-population prediction
- **Technical Advantages**: Precise quantification of genetic differences between populations
- **Output Results**: Cross-entropy optimized prediction models

### Module 6: KL Divergence Optimized Prediction (`KL_model_pred/`)
**Research Objective**: Utilize Kullback-Leibler divergence to optimize cross-population SNP feature selection
- **Theoretical Foundation**: KL divergence quantifies information loss in SNP probability distributions
- **Selection Principle**: Identify SNP subsets with minimal distribution differences between populations
- **Comparative Analysis**: Performance comparison with cross-entropy methods
- **Optimization Strategy**: Systematic evaluation of different proportions of SNP subsets
- **Output Results**: KL divergence-guided feature selection models

### Module 7: Key SNP Importance Analysis (`Gene_importance/`)
**Research Objective**: Evaluate SNP impact on model performance through iterative SNP removal
- **Analysis Method**: Importance scoring algorithm through individual SNP removal
- **Importance Calculation**: Quantification of SNP contribution based on model performance changes
- **Selection Target**: Identify key SNPs highly associated with Botrytis cinerea resistance
- **Biological Significance**: Provide candidate loci for marker-assisted selection
- **Output Results**: SNP importance rankings and key locus lists

