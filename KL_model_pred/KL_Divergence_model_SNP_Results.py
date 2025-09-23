import pandas as pd
import numpy as np
import itertools
from itertools import product

# Load custom functions
from KL_Divergence_model_train_test import (
    calculate_prob_distributions,
    calculate_kl_divergence,
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
kl_prob_distributions = calculate_prob_distributions(
    df1='./nature_GWAS_SNP_RS.tsv',
    df2='./F1_GWAS_SNP_RS.tsv'
)

# Calculate KL divergence between two groups
kl_snp, kl_res = calculate_kl_divergence(kl_prob_distributions)

# Find the SNPs with the maximum and minimum KL divergence values
max_kl_snp_id = max(kl_snp, key=kl_snp.get)
max_kl_value = kl_snp[max_kl_snp_id]
min_kl_snp_id = min(kl_snp, key=kl_snp.get)
min_kl_value = kl_snp[min_kl_snp_id]

# Select SNPs based on KL divergence percentile thresholds
kl_percentile = [0.125, 0.250, 0.375, 0.500, 0.625, 0.750, 0.875, 1.000]
kl_select_snps = select_snps_by_percentile(kl_snp, kl_percentile)

# Sort selected SNPs by chromosome and position
for key in kl_select_snps:
    kl_select_snps[key] = sorted(
        kl_select_snps[key],
        key=lambda x: (int(x.split('_')[0][3:]), int(x.split('_')[1]))
    )

# Write KL divergence results and selected SNPs to a text file
with open("KL_Divergence_model_SNP_Results.txt", "w") as f:
    f.write(f"Max KL value: {max_kl_value}, SNP ID: {max_kl_snp_id}\n")
    f.write(f"Min KL value: {min_kl_value}, SNP ID: {min_kl_snp_id}\n\n")
    f.write("Selected SNPs by percentile:\n")
    for k, v in kl_select_snps.items():
        f.write(f"Percentile {k}: {v}\n")
