# -------------------------
# Import required libraries
# -------------------------
# os: used for file path operations, e.g., loading dataset
# product: generates Cartesian product of hyperparameter options
# pandas and numpy: for reading data tables and numerical computing
# sklearn.model_selection: for cross-validation splitting and grid search
# sklearn.linear_model: logistic regression classifier
# sklearn.metrics: used to compute model evaluation metrics (accuracy, AUC, precision, recall, F1)

import os
from itertools import product
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

# =============================================================
# This script applies logistic regression to predict rose
# gray mold resistance (R: Resistant, S: Susceptible).
#
# Major functions include:
# 1. Load SNP data and extract features and labels;
# 2. Construct hyperparameter grid and apply GridSearchCV;
# 3. Evaluate model performance using 10-fold cross-validation;
# 4. Output average accuracy, ROC-AUC, precision, recall, F1 metrics.
# Label definition:
#   - R (Resistant) is mapped to 0
#   - S (Susceptible) is mapped to 1
# =============================================================

# File path and loading
data_dir = './'
file_name = 'nature_GWAS_SNP_RS.tsv'
file_path = os.path.join(data_dir, file_name)
data = pd.read_csv(file_path, index_col=0, sep='\t')

# Feature matrix and label vector
X = data.drop(columns=['rank'])
y = data['rank'].map({'R': 0, 'S': 1})

# Stratified 10-fold cross-validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2025)
results_for_run = []
all_results = []

# Define compatible hyperparameter grid manually
LR_param_grid = []
C_list = [0.001, 0.01, 0.1, 1, 10, 100]
max_iter_list = [500, 1000, 2000, 3000]  # Increased max_iter to reduce convergence warnings
penalty_list = ['l1', 'l2']
solver_list = ['liblinear', 'lbfgs']  # Removed 'saga' to reduce convergence warnings
class_weight_list = [None, 'balanced']

for penalty, C, solver, class_weight, max_iter in product(penalty_list, C_list, solver_list, class_weight_list, max_iter_list):
    if penalty == 'l1' and solver not in ['liblinear']:
        continue
    if penalty == 'l2' and solver not in ['liblinear', 'lbfgs']:
        continue

    LR_param_grid.append({
        'penalty': [penalty],
        'C': [C],
        'solver': [solver],
        'class_weight': [class_weight],
        'max_iter': [max_iter]
    })

# Run GridSearchCV only on valid combinations
grid_search = GridSearchCV(
    estimator=LogisticRegression(random_state=42),
    param_grid=LR_param_grid,
    cv=skf,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X, y)
best_params = grid_search.best_params_

# 10-fold cross-validation using the best parameters
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = LogisticRegression(**best_params, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_value = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results_for_run.append((accuracy, roc_value, precision, recall, f1))

# Average metrics
average_accuracy = np.mean([acc for acc, _, _, _, _ in results_for_run])
average_roc = np.mean([roc for _, roc, _, _, _ in results_for_run])
average_precision = np.mean([prec for _, _, prec, _, _ in results_for_run])
average_recall = np.mean([rec for _, _, _, rec, _ in results_for_run])
average_f1 = np.mean([f1 for _, _, _, _, f1 in results_for_run])

# Collect final results
all_results.append((file_name, results_for_run, average_roc, average_precision, average_recall, average_f1))

# Write output
with open('out_result_Nature_GWAS_LR_endadta.txt', 'a') as file:
    print(f"Nature_GWAS_LR_Best_parameters: {best_params}", file=file)

    for file_name, results, avg_roc, avg_prec, avg_recall, avg_f1 in all_results:
        avg_acc = np.mean([acc for acc, _, _, _, _ in results])
        print(f">{file_name}:Average-Accuracy:{avg_acc:.5f}"
              f":Average-ROC-AUC-Score:{avg_roc:.5f}"
              f":Average-Precision:{avg_prec:.5f}"
              f":Average-Recall:{avg_recall:.5f}"
              f":Average-F1-Score:{avg_f1:.5f}", file=file)

        for i, (accuracy, roc, precision, recall, f1) in enumerate(results, start=1):
            print(f"{i}:Accuracy:{accuracy:.5f}"
                  f":ROC-AUC:{roc:.5f}"
                  f":Precision:{precision:.5f}"
                  f":Recall:{recall:.5f}"
                  f":F1-Score:{f1:.5f}", file=file)