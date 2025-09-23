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

# ======================== Step 3: Train and Evaluate SVC Classifier ========================
# Define hyperparameter grid for SVC
SVC_param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'degree': [2, 3, 4],
    'coef0': [0.0, 0.1, 0.5, 1.0],
    'class_weight': [None, 'balanced'],
    'probability': [True],
    'random_state': [42]
}

# Expand hyperparameter combinations
svc_param_grid = list(ParameterGrid(SVC_param_grid))
print(f"Total valid parameter combinations: {len(svc_param_grid)}")

SVC_results = []

for SVC_idx, parameters in enumerate(svc_param_grid):
    print(f"\nEvaluating combination {SVC_idx + 1}/{len(svc_param_grid)}: {parameters}")

    SVC_result = train_and_evaluate(
        model_type=SVC,
        train_data_path='./nature_GWAS_SNP_RS.tsv',
        test_data_path='./F1_GWAS_SNP_RS.tsv',
        selected_snps=CR_select_snps,
        parameters=parameters
    )

    for res in SVC_result:
        res['SVC_parameters'] = parameters
        res['SVC_idx'] = SVC_idx

    SVC_results.append(SVC_result)

# ======================== Step 4: Extract Best Accuracy for Each Threshold (SVC) ========================
def obtain_best_SVC_res(SVC_results, percentile, output_file="CR_SVC_model_detailed_results_by_Threshold.txt"):
    SVC_acc_all = []
    for i in range(len(SVC_results)):
        SVC_acc_p = []
        for j in range(len(percentile)):
            SVC_acc_p_t = SVC_results[i][j]['accuracy']
            SVC_acc_p.append(SVC_acc_p_t)
        SVC_acc_all.append(SVC_acc_p)

    best_SVC_acc_on_different_percentile = np.max(SVC_acc_all, axis=0)
    best_SVC_acc_on_different_percentile_ids = np.argmax(SVC_acc_all, axis=0)

    # Write results to file
    with open(output_file, "a") as f:
        for idx, ids in enumerate(best_SVC_acc_on_different_percentile_ids):
            # 写最佳准确率基本信息
            f.write(f"Best Accuracy at CR threshold {percentile[idx]}:\n")
            f.write(f"  Accuracy: {best_SVC_acc_on_different_percentile[idx]:.5f}\n")
            f.write(f"  Parameter set ID: {SVC_results[ids][idx]['SVC_idx']}\n")
            f.write(f"  Parameters: {SVC_results[ids][idx]['SVC_parameters']}\n")
        
            # 写详细的模型评估指标
            model_details = SVC_results[ids][idx]
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

    return best_SVC_acc_on_different_percentile_ids

# Run SVC evaluation and save output
acc_SVC_all = obtain_best_SVC_res(SVC_results, CR_percentile)
