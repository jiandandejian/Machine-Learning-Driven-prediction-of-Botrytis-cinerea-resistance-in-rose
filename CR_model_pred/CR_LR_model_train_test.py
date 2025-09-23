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

# ======================== Step 3: Define Logistic Regression Hyperparameters ========================
# Define hyperparameter grid for Logistic Regression
C_list = [0.001, 0.01, 0.1, 1, 10, 100]
max_iter_list = [500, 1000, 2000, 3000]
penalty_list = ['l1', 'l2']  # 'elasticnet' is excluded due to solver limitations
solver_list = ['liblinear', 'lbfgs']
class_weight_list = [None, 'balanced']

LR_param_grid = []

for penalty, C, solver, class_weight, max_iter in product(penalty_list, C_list, solver_list, class_weight_list, max_iter_list):
    if penalty == 'l1' and solver not in ['liblinear']:
        continue
    if penalty == 'l2' and solver not in ['liblinear', 'lbfgs']:
        continue

    LR_param_grid.append({
        'penalty': penalty,
        'C': C,
        'solver': solver,
        'class_weight': class_weight,
        'max_iter': max_iter,
        'random_state': 42
    })

print(f"Total LR parameter combinations: {len(LR_param_grid)}")

# ======================== Step 4: Train and Evaluate Logistic Regression ========================
# Iterate through parameter combinations and train models
LR_results = []

for LR_idx, parameters in enumerate(LR_param_grid):
    print(f"\nEvaluating combination {LR_idx + 1}/{len(LR_param_grid)}: {parameters}")

    LR_result = train_and_evaluate(
        model_type=LogisticRegression,
        train_data_path='./nature_GWAS_SNP_RS.tsv',
        test_data_path='./F1_GWAS_SNP_RS.tsv',
        selected_snps=CR_select_snps,
        parameters=parameters
    )

    for res in LR_result:
        res['LR_parameters'] = parameters
        res['LR_idx'] = LR_idx

    LR_results.append(LR_result)

# ======================== Step 5: Extract Best Accuracy for Each Threshold ========================
def obtain_best_LR_res(LR_results, percentile, output_file="CR_LR_model_detailed_results_by_Threshold.txt"):
    LR_acc_all = []
    for i in range(len(LR_results)):
        LR_acc_p = []
        for j in range(len(percentile)):
            LR_acc_p_t = LR_results[i][j]['accuracy']
            LR_acc_p.append(LR_acc_p_t)
        LR_acc_all.append(LR_acc_p)

    best_LR_acc_on_different_percentile = np.max(LR_acc_all, axis=0)
    best_LR_acc_on_different_percentile_ids = np.argmax(LR_acc_all, axis=0)

    # Write results to file
    with open(output_file, "a") as f:
        for idx, ids in enumerate(best_LR_acc_on_different_percentile_ids):
            # 写最佳准确率基本信息
            f.write(f"Best Accuracy at CR threshold {percentile[idx]}:\n")
            f.write(f"  Accuracy: {best_LR_acc_on_different_percentile[idx]:.5f}\n")
            f.write(f"  Parameter set ID: {LR_results[ids][idx]['LR_idx']}\n")
            f.write(f"  Parameters: {LR_results[ids][idx]['LR_parameters']}\n")
        
            # 写详细的模型评估指标
            model_details = LR_results[ids][idx]
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

    return best_LR_acc_on_different_percentile_ids

# ======================== Step 6: Run Evaluation Summary ========================
acc_LR_all = obtain_best_LR_res(LR_results, CR_percentile)
