import pandas as pd
import numpy as np
import itertools
from itertools import product

# Import custom training and evaluation functions
from Cross_Entropy_model_train_test import (
    calculate_prob_distributions,
    calculate_cross_entropy,
    select_snps_by_percentile,
    train_and_evaluate
)

# Import model types from scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ParameterGrid

# ======================== Step 1: Compute KL Divergence ========================
# Load input data and compute smoothed SNP probability distributions
CR_prob_distributions = calculate_prob_distributions(df1='./nature_GWAS_SNP_RS.tsv', df2='./F1_GWAS_SNP_RS.tsv')

# Calculate cross-entropy divergence between two groups
CR_snp, CR_res = calculate_cross_entropy(CR_prob_distributions)

# ======================== Step 2: Select SNPs by CR percentile ========================
# Define thresholds and select top SNPs based on cross-entropy divergence
CR_percentile = [0.125, 0.250, 0.375, 0.500, 0.625, 0.750, 0.875, 1.000]
CR_select_snps = select_snps_by_percentile(CR_snp, CR_percentile)

# Sort selected SNPs by chromosomal position
for key in CR_select_snps:
    CR_select_snps[key] = sorted(
        CR_select_snps[key],
        key=lambda x: (int(x.split('_')[0][3:]), int(x.split('_')[1]))
    )

# ======================== Step 3: Train and Evaluate MLPClassifier ========================
# Define hyperparameter grid for MLPClassifier
MLPC_param_grid = {
    'hidden_layer_sizes': [(128, 64),(128, 64, 32)],
    'activation': ['relu'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001],
    'learning_rate': ['constant'],
    'learning_rate_init': [0.001, 0.01],
    'momentum': [0.9, 0.95],
    'early_stopping': [True],
    'max_iter': [300],
    'batch_size': ['auto', 16],
    'validation_fraction':[0.035],  
    'random_state': [42]  
}

# Expand hyperparameter combinations using sklearn's ParameterGrid
MLPC_param_grid = list(ParameterGrid(MLPC_param_grid))

print(f"Total valid parameter combinations: {len(MLPC_param_grid)}")

MLPC_results = []

for MLPC_idx, parameters in enumerate(MLPC_param_grid):
    print(f"\nEvaluating combination {MLPC_idx + 1}/{len(MLPC_param_grid)}: {parameters}")

    MLPC_result = train_and_evaluate(
        model_type=MLPClassifier,
        train_data_path='./nature_GWAS_SNP_RS.tsv',
        test_data_path='./F1_GWAS_SNP_RS.tsv',
        selected_snps=CR_select_snps,
        parameters=parameters,
    )

    for res in MLPC_result:
        res['MLPC_parameters'] = parameters
        res['MLPC_idx'] = MLPC_idx

    MLPC_results.append(MLPC_result)

# ======================== Step 4: Extract Best Accuracy for Each Threshold (MLPC) ========================
def obtain_best_MLPC_res(MLPC_results, percentile, output_file="CR_MLPC_model_detailed_results_by_Threshold.txt"):
    MLPC_acc_all = []
    for i in range(len(MLPC_results)):
        MLPC_acc_p = []
        for j in range(len(percentile)):
            MLPC_acc_p_t = MLPC_results[i][j]['accuracy']
            MLPC_acc_p.append(MLPC_acc_p_t)
        MLPC_acc_all.append(MLPC_acc_p)

    best_MLPC_acc_on_different_percentile = np.max(MLPC_acc_all, axis=0)
    best_MLPC_acc_on_different_percentile_ids = np.argmax(MLPC_acc_all, axis=0)

    # Write results to file
    with open(output_file, "a") as f:
        for idx, ids in enumerate(best_MLPC_acc_on_different_percentile_ids):
            # 写最佳准确率基本信息
            f.write(f"Best Accuracy at CR threshold {percentile[idx]}:\n")
            f.write(f"  Accuracy: {best_MLPC_acc_on_different_percentile[idx]:.5f}\n")
            f.write(f"  Parameter set ID: {MLPC_results[ids][idx]['MLPC_idx']}\n")
            f.write(f"  Parameters: {MLPC_results[ids][idx]['MLPC_parameters']}\n")
        
            # 写详细的模型评估指标
            model_details = MLPC_results[ids][idx]
            f.write("  Model Evaluation Metrics:\n")
            f.write(f"    ROC AUC: {model_details['roc_auc']:.5f}\n")
            f.write(f"    Average Precision: {model_details['average_precision']:.5f}\n")
            f.write(f"    F1 Score: {model_details['f1_score']:.5f}\n")
            f.write(f"    SNP Count: {model_details['snp_count']}\n")
            f.write(f"    Confusion Matrix: {model_details['confusion_matrix']}\n")
            f.write(f"    Precision List: {model_details['precision']}\n")
            f.write(f"    Recall List: {model_details['recall']}\n")
            f.write(f"    FPR List: {model_details['fpr']}\n")
            f.write(f"    TPR List: {model_details['tpr']}\n")
            f.write(f"    Thresholds (ROC): {model_details['thresholds_ft']}\n")
            f.write(f"    Thresholds (PR): {model_details['thresholds_pr']}\n")
            f.write(f"    y_test: {model_details['y_test']}\n")
            f.write(f"    y_pred: {model_details['y_pred']}\n")
            f.write("\n" + "="*80 + "\n\n")  # 分隔线，方便查看每组输出

    return best_MLPC_acc_on_different_percentile_ids

# Run MLPC evaluation and save output
acc_MLPC_all = obtain_best_MLPC_res(MLPC_results, CR_percentile)
