import pandas as pd
import numpy as np
import itertools
from itertools import product

# Import custom training and evaluation functions
from KL_Divergence_model_train_test import (
    calculate_prob_distributions,
    calculate_kl_divergence,
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
kl_prob_distributions = calculate_prob_distributions(df1='./nature_GWAS_SNP_RS.tsv', df2='./F1_GWAS_SNP_RS.tsv')

# Compute KL divergence between the two datasets
kl_snp, kl_res = calculate_kl_divergence(kl_prob_distributions)

# ======================== Step 2: Select SNPs by KL percentile ========================
# Define thresholds and select top SNPs based on KL divergence
kl_percentile = [0.125, 0.250, 0.375, 0.500, 0.625, 0.750, 0.875, 1.000]
kl_select_snps = select_snps_by_percentile(kl_snp, kl_percentile)

# Sort selected SNPs by chromosomal position
for key in kl_select_snps:
    kl_select_snps[key] = sorted(
        kl_select_snps[key], 
        key=lambda x: (int(x.split('_')[0][3:]), int(x.split('_')[1]))
    )

# ======================== Step 3: Train and Evaluate LightGBM Classifier ========================
# Define hyperparameter grid for LGBMClassifier
LGBMC_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [-1, 10, 15],
    'min_child_samples': [20, 50],
    'colsample_bytree': [0.8, 1.0],
    'subsample': [0.8, 1.0],
    'min_split_gain': [0.1, 0.2],
    'boosting_type': ['gbdt', 'dart'],
    'num_leaves': [31], 
    'learning_rate': [0.01], 
    'bagging_freq': [1], 
    'lambda_l1': [10], 
    'lambda_l2': [0],
    'random_state': [42],
    'verbosity' : [-1]
}

# Expand hyperparameter combinations using sklearn's ParameterGrid
LGBMC_param_grid = list(ParameterGrid(LGBMC_param_grid))

print(f"Total valid parameter combinations: {len(LGBMC_param_grid)}")

LGBMC_results = []

for LGBMC_idx, parameters in enumerate(LGBMC_param_grid):
    print(f"\nEvaluating combination {LGBMC_idx + 1}/{len(LGBMC_param_grid)}: {parameters}")

    LGBMC_result = train_and_evaluate(
        model_type=LGBMClassifier,
        train_data_path='./nature_GWAS_SNP_RS.tsv',
        test_data_path='./F1_GWAS_SNP_RS.tsv',
        selected_snps=kl_select_snps,
        parameters=parameters
    )

    for res in LGBMC_result:
        res['LGBMC_parameters'] = parameters
        res['LGBMC_idx'] = LGBMC_idx

    LGBMC_results.append(LGBMC_result)

# ======================== Step 4: Extract Best Accuracy for Each Threshold (LGBM) ========================
def obtain_best_LGBMC_res(LGBMC_results, percentile, output_file="KL_LGBMC_model_detailed_results_by_Threshold.txt"):
    LGBMC_acc_all = []
    for i in range(len(LGBMC_results)):
        LGBMC_acc_p = []
        for j in range(len(percentile)):
            LGBMC_acc_p_t = LGBMC_results[i][j]['accuracy']
            LGBMC_acc_p.append(LGBMC_acc_p_t)
        LGBMC_acc_all.append(LGBMC_acc_p)

    best_LGBMC_acc_on_different_percentile = np.max(LGBMC_acc_all, axis=0)
    best_LGBMC_acc_on_different_percentile_ids = np.argmax(LGBMC_acc_all, axis=0)

    # Write results to file
    with open(output_file, "a") as f:
        for idx, ids in enumerate(best_LGBMC_acc_on_different_percentile_ids):
            # 写最佳准确率基本信息
            f.write(f"Best Accuracy at KL threshold {percentile[idx]}:\n")
            f.write(f"  Accuracy: {best_LGBMC_acc_on_different_percentile[idx]:.5f}\n")
            f.write(f"  Parameter set ID: {LGBMC_results[ids][idx]['LGBMC_idx']}\n")
            f.write(f"  Parameters: {LGBMC_results[ids][idx]['LGBMC_parameters']}\n")
        
            # 写详细的模型评估指标
            model_details = LGBMC_results[ids][idx]
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

    return best_LGBMC_acc_on_different_percentile_ids

# Run LGBM evaluation and save output
acc_LGBMC_all = obtain_best_LGBMC_res(LGBMC_results, kl_percentile)

