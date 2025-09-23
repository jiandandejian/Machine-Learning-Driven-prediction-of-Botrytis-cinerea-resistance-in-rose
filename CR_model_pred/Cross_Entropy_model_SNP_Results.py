import pandas as pd
import numpy as np
import itertools
from itertools import product

# Load custom functions
from Cross_Entropy_model_train_test import (
    calculate_prob_distributions,
    calculate_cross_entropy,
    select_snps_by_percentile,
    train_and_evaluate
)

# Load models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier 
from sklearn.model_selection import ParameterGrid

# Load data files and calculate SNP genotype probability distributions
CR_prob_distributions = calculate_prob_distributions(
    df1='./nature_GWAS_SNP_RS.tsv',
    df2='./F1_GWAS_SNP_RS.tsv'
)

# Calculate cross-entropy divergence between two groups
CR_snp, CR_res = calculate_cross_entropy(CR_prob_distributions)

# Find the SNPs with the maximum and minimum cross-entropy divergence values
max_CR_snp_id = max(CR_snp, key=CR_snp.get)
max_CR_value = CR_snp[max_CR_snp_id]
min_CR_snp_id = min(CR_snp, key=CR_snp.get)
min_CR_value = CR_snp[min_CR_snp_id]

# Select SNPs based on cross-entropy divergence percentile thresholds
CR_percentile = [0.125, 0.250, 0.375, 0.500, 0.625, 0.750, 0.875, 1.000]
CR_select_snps = select_snps_by_percentile(CR_snp, CR_percentile)

# Sort selected SNPs by chromosome and position
for key in CR_select_snps:
    CR_select_snps[key] = sorted(
        CR_select_snps[key],
        key=lambda x: (int(x.split('_')[0][3:]), int(x.split('_')[1]))
    )

# Write cross-entropy divergence results and selected SNPs to a text file
with open("Cross_Entropy_model_SNP_Results.txt", "w") as f:
    f.write(f"Max CR value: {max_CR_value}, SNP ID: {max_CR_snp_id}\n")
    f.write(f"Min CR value: {min_CR_value}, SNP ID: {min_CR_snp_id}\n\n")
    f.write("Selected SNPs by percentile:\n")
    for k, v in CR_select_snps.items():
        f.write(f"Percentile {k}: {v}\n")
