import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from itertools import product

# ========== Load training and testing datasets ==========

# Natural population (training set)
data1 = pd.read_csv("./nature_GWAS_SNP_RS.tsv", index_col=0, sep='\t')
X_train = data1.drop(columns=['rank'])
y_train = data1['rank'].map({'R': 0, 'S': 1})

# Clustered population (test set)
data2 = pd.read_csv("./F1_GWAS_SNP_RS.tsv", index_col=0, sep='\t')
X_test = data2.drop(columns=['rank'])
y_test = data2['rank'].map({'R': 0, 'S': 1})

# ===================== Random Forest LGBMClassifier Section =====================
RFC_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4, 6],
    'bootstrap': [True],
    'class_weight': ['balanced'],
    'criterion': ['entropy', 'log_loss']
}

RFC_skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2025)
RFC_grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=RFC_param_grid,
    cv=RFC_skf,
    scoring='accuracy',
    n_jobs=-1
)
RFC_grid_search.fit(X_train, y_train)
RFC_best_params = RFC_grid_search.best_params_

RFC_model = RandomForestClassifier(**RFC_best_params, random_state=42)
RFC_model.fit(X_train, y_train)
RFC_y_pred = RFC_model.predict(X_test)
RFC_y_prob = RFC_model.predict_proba(X_test)[:, 1]

RFC_accuracy = accuracy_score(y_test, RFC_y_pred)
RFC_roc = roc_auc_score(y_test, RFC_y_prob)
RFC_precision = precision_score(y_test, RFC_y_pred)
RFC_recall = recall_score(y_test, RFC_y_pred)
RFC_f1 = f1_score(y_test, RFC_y_pred)

# ===================== SVC Section =====================
SVC_param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'degree': [2, 3, 4],
    'coef0': [0.0, 0.1, 0.5, 1.0],
    'class_weight': [None, 'balanced'],
    'probability': [True]
}

SVC_skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2025)
SVC_grid_search = GridSearchCV(
    estimator=SVC(random_state=42),
    param_grid=SVC_param_grid,
    scoring='accuracy',
    cv=SVC_skf,
    n_jobs=-1
)
SVC_grid_search.fit(X_train, y_train)
SVC_best_params = SVC_grid_search.best_params_

SVC_model = SVC(**SVC_best_params, random_state=42)
SVC_model.fit(X_train, y_train)
SVC_y_pred = SVC_model.predict(X_test)
SVC_y_prob = SVC_model.predict_proba(X_test)[:, 1]

SVC_accuracy = accuracy_score(y_test, SVC_y_pred)
SVC_roc = roc_auc_score(y_test, SVC_y_prob)
SVC_precision = precision_score(y_test, SVC_y_pred)
SVC_recall = recall_score(y_test, SVC_y_pred)
SVC_f1 = f1_score(y_test, SVC_y_pred)

# ===================== LightGBM Classifier Section =====================
LGBMC_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [-1, 10, 15],
    'min_child_samples': [20, 50],
    'colsample_bytree': [0.8, 1.0],
    'subsample': [0.8, 1.0],
    'min_split_gain': [0.1, 0.2],
    'boosting_type': ['gbdt', 'dart']
}

LGBMC_skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2025)
LGBMC_grid_search = GridSearchCV(
    estimator=LGBMClassifier(random_state=42, verbosity=-1),
    param_grid=LGBMC_param_grid,
    scoring='accuracy',
    cv=LGBMC_skf,
    n_jobs=-1
)
LGBMC_grid_search.fit(X_train, y_train)
LGBMC_best_params = LGBMC_grid_search.best_params_

LGBMC_model = LGBMClassifier(**LGBMC_best_params, random_state=42, verbosity=-1)
LGBMC_model.fit(X_train, y_train)
LGBMC_y_pred = LGBMC_model.predict(X_test)
LGBMC_y_prob = LGBMC_model.predict_proba(X_test)[:, 1]

LGBMC_accuracy = accuracy_score(y_test, LGBMC_y_pred)
LGBMC_roc = roc_auc_score(y_test, LGBMC_y_prob)
LGBMC_precision = precision_score(y_test, LGBMC_y_pred)
LGBMC_recall = recall_score(y_test, LGBMC_y_pred)
LGBMC_f1 = f1_score(y_test, LGBMC_y_pred)

# ===================== Logistic Regression Section =====================
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

LR_skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2025)
LR_grid_search = GridSearchCV(
    estimator=LogisticRegression(random_state=42),
    param_grid=LR_param_grid,
    scoring='accuracy',
    cv=LR_skf,
    n_jobs=-1
)
LR_grid_search.fit(X_train, y_train)
LR_best_params = LR_grid_search.best_params_

LR_model = LogisticRegression(**LR_best_params, random_state=42)
LR_model.fit(X_train, y_train)
LR_y_pred = LR_model.predict(X_test)
LR_y_prob = LR_model.predict_proba(X_test)[:, 1]

LR_accuracy = accuracy_score(y_test, LR_y_pred)
LR_roc = roc_auc_score(y_test, LR_y_prob)
LR_precision = precision_score(y_test, LR_y_pred)
LR_recall = recall_score(y_test, LR_y_pred)
LR_f1 = f1_score(y_test, LR_y_pred)

# ===================== MLP Classifier Section =====================
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
    'random_state': [42]  
}

MLPC_skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2025)
MLPC_grid_search = GridSearchCV(
    estimator=MLPClassifier(),
    param_grid=MLPC_param_grid,
    cv=MLPC_skf,
    scoring='accuracy',
    n_jobs=-1
)
MLPC_grid_search.fit(X_train, y_train)
MLPC_best_params = MLPC_grid_search.best_params_

MLPC_model = MLPClassifier(**MLPC_best_params)
MLPC_model.fit(X_train, y_train)
MLPC_y_pred = MLPC_model.predict(X_test)
MLPC_y_prob = MLPC_model.predict_proba(X_test)[:, 1]

MLPC_accuracy = accuracy_score(y_test, MLPC_y_pred)
MLPC_roc = roc_auc_score(y_test, MLPC_y_prob)
MLPC_precision = precision_score(y_test, MLPC_y_pred)
MLPC_recall = recall_score(y_test, MLPC_y_pred)
MLPC_f1 = f1_score(y_test, MLPC_y_pred)

# ===================== Output Combined Results =====================
with open('Nature_Grid_Search_F1_test_Results.txt', 'w') as f:
    f.write("=== Random Forest ===\n")
    f.write(f"RFC_Best_Params: {RFC_best_params}\n")
    f.write(f"RFC_y_pred: {RFC_y_pred}\n")
    f.write(f"RFC_y_prob: {RFC_y_prob}\n")
    f.write(f"RFC_Accuracy: {RFC_accuracy:.5f}\n")
    f.write(f"RFC_ROC_AUC: {RFC_roc:.5f}\n")
    f.write(f"RFC_Precision: {RFC_precision:.5f}\n")
    f.write(f"RFC_Recall: {RFC_recall:.5f}\n")
    f.write(f"RFC_F1_Score: {RFC_f1:.5f}\n\n")

    f.write("=== SVC ===\n")
    f.write(f"SVC_Best_Params: {SVC_best_params}\n")
    f.write(f"SVC_y_pred: {SVC_y_pred}\n")
    f.write(f"SVC_y_prob: {SVC_y_prob}\n")
    f.write(f"SVC_Accuracy: {SVC_accuracy:.5f}\n")
    f.write(f"SVC_ROC_AUC: {SVC_roc:.5f}\n")
    f.write(f"SVC_Precision: {SVC_precision:.5f}\n")
    f.write(f"SVC_Recall: {SVC_recall:.5f}\n")
    f.write(f"SVC_F1_Score: {SVC_f1:.5f}\n\n")

    f.write("=== LightGBM Classifier===\n")
    f.write(f"LGBMC_Best_Params: {LGBMC_best_params}\n")
    f.write(f"LGBMC_y_pred: {LGBMC_y_pred}\n")
    f.write(f"LGBMC_y_prob: {LGBMC_y_prob}\n")
    f.write(f"LGBMC_Accuracy: {LGBMC_accuracy:.5f}\n")
    f.write(f"LGBMC_ROC_AUC: {LGBMC_roc:.5f}\n")
    f.write(f"LGBMC_Precision: {LGBMC_precision:.5f}\n")
    f.write(f"LGBMC_Recall: {LGBMC_recall:.5f}\n")
    f.write(f"LGBMC_F1_Score: {LGBMC_f1:.5f}\n\n")

    f.write("=== Logistic Regression ===\n")
    f.write(f"LR_Best_Params: {LR_best_params}\n")
    f.write(f"LR_y_pred: {LR_y_pred}\n")
    f.write(f"LR_y_prob: {LR_y_prob}\n")
    f.write(f"LR_Accuracy: {LR_accuracy:.5f}\n")
    f.write(f"LR_ROC_AUC: {LR_roc:.5f}\n")
    f.write(f"LR_Precision: {LR_precision:.5f}\n")
    f.write(f"LR_Recall: {LR_recall:.5f}\n")
    f.write(f"LR_F1_Score: {LR_f1:.5f}\n\n")

    f.write("=== MLP Classifier ===\n")
    f.write(f"MLPC_Best_Params: {MLPC_best_params}\n")
    f.write(f"MLPC_y_pred: {MLPC_y_pred}\n")
    f.write(f"MLPC_y_prob: {MLPC_y_prob}\n")
    f.write(f"MLPC_Accuracy: {MLPC_accuracy:.5f}\n")
    f.write(f"MLPC_ROC_AUC: {MLPC_roc:.5f}\n")
    f.write(f"MLPC_Precision: {MLPC_precision:.5f}\n")
    f.write(f"MLPC_Recall: {MLPC_recall:.5f}\n")
    f.write(f"MLPC_F1_Score: {MLPC_f1:.5f}\n")
