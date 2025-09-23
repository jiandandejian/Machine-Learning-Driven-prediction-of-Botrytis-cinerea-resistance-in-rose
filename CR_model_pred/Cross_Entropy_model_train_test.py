from collections import Counter
import pandas as pd
import scipy.stats as stats
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, f1_score, confusion_matrix, accuracy_score, auc
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_prob_distributions(df1, df2, smooth=1e-9):
    # Load datasets
    df1 = pd.read_csv(df1, index_col=0, sep='\t')
    df2 = pd.read_csv(df2, index_col=0, sep='\t')

    # Get SNP ID columns (assume the last column is the label column and ignore it)
    snp_ids = df1.columns[:-1]
    prob_distributions = {}

    for snp in snp_ids:
        # Get SNP values in both datasets and compute frequency
        freq1, freq2 = Counter(df1[snp]), Counter(df2[snp])

        # Normalize frequencies to obtain probability distributions
        total_count1, total_count2 = sum(freq1.values()), sum(freq2.values())
        prob_dist1 = {k: v / total_count1 for k, v in freq1.items()}
        prob_dist2 = {k: v / total_count2 for k, v in freq2.items()}

        # Get all unique keys from both distributions
        all_keys = set(prob_dist1.keys()).union(prob_dist2.keys())
        unique_elements = set(prob_dist1.keys()) ^ set(prob_dist2.keys())

        # Skip incompatible SNPs with disjoint keys
        if len(unique_elements) == 3:
            continue

        # Apply smoothing and align keys
        p_smoothed = np.array([prob_dist1.get(k, 0) + smooth for k in sorted(all_keys)])
        q_smoothed = np.array([prob_dist2.get(k, 0) + smooth for k in sorted(all_keys)])

        # Re-normalize after smoothing
        p_smoothed /= p_smoothed.sum()
        q_smoothed /= q_smoothed.sum()

        # Store smoothed distributions
        prob_distributions[snp] = (p_smoothed, q_smoothed)

    return prob_distributions

def calculate_cross_entropy(prob_distributions):
    cross_entropy_snp = {}
    cross_entropy_res = []

    for snp, (p_smoothed, q_smoothed) in prob_distributions.items():
        # Compute cross-entropy
        cross_entropy = -np.sum(p_smoothed * np.log(q_smoothed))
        cross_entropy_snp[snp] = cross_entropy
        cross_entropy_res.append(cross_entropy)

    return cross_entropy_snp, cross_entropy_res

def select_snps_by_percentile(df, percentiles):
    sorted_snps = sorted(df.items(), key=lambda x: x[1])
    total_snps = len(sorted_snps)

    selected_snps = {}

    for percentile in percentiles:
        top_n = int(total_snps * percentile)
        selected_snps[percentile] = [snp for snp, _ in sorted_snps[:top_n]]

    return selected_snps

def train_and_evaluate(model_type=None, train_data_path=None, test_data_path=None, selected_snps=None, parameters={}):
    # Validate input
    if train_data_path is None or test_data_path is None or selected_snps is None:
        raise ValueError("Train data path, test data path, and selected SNPs cannot be None")

    # Load training and testing datasets
    train_df = pd.read_csv(train_data_path, index_col=0, sep='\t')
    test_df = pd.read_csv(test_data_path, index_col=0, sep='\t')

    # Convert labels to binary integers
    y_train = train_df.iloc[:, -1].map({'R': 0, 'S': 1}).astype(np.int8)
    y_test = test_df.iloc[:, -1].map({'R': 0, 'S': 1}).astype(np.int8)

    results = []

    for i, snps in selected_snps.items():
        X_train = train_df.loc[:, snps]
        X_test = test_df.loc[:, snps]

        model = model_type(**parameters)
        model.fit(X_train, y_train)

        y_test_prob = model.predict_proba(X_test)[:, 1]
        y_test_pred = model.predict(X_test)

        # Compute evaluation metrics
        fpr, tpr, thresholds_ft = roc_curve(y_test, y_test_prob)
        valid_indices_ft = ~np.isinf(thresholds_ft)
        fpr, tpr, thresholds_ft = fpr[valid_indices_ft], tpr[valid_indices_ft], thresholds_ft[valid_indices_ft]

        roc_auc = roc_auc_score(y_test, y_test_prob)

        precision, recall, thresholds_pr = precision_recall_curve(y_test, y_test_prob)
        valid_indices_pr = ~np.isinf(thresholds_pr)
        precision, recall, thresholds_pr = precision[:-1][valid_indices_pr], recall[:-1][valid_indices_pr], thresholds_pr[valid_indices_pr]

        avg_precision = average_precision_score(y_test, y_test_prob)
        f1 = f1_score(y_test, y_test_pred)
        cm = confusion_matrix(y_test, y_test_pred)
        accuracy = accuracy_score(y_test, y_test_pred)
        snp_count = len(snps)

        # Store results
        model_results = {
            'model_name': f'Threshold {i}',
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds_ft': thresholds_ft.tolist(),
            'roc_auc': roc_auc,
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds_pr': thresholds_pr.tolist(),
            'average_precision': avg_precision,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'accuracy': accuracy,
            'snp_count': snp_count,
            'y_test': y_test.tolist(),
            'y_pred': y_test_pred.tolist()
        }
        results.append(model_results)

    return results