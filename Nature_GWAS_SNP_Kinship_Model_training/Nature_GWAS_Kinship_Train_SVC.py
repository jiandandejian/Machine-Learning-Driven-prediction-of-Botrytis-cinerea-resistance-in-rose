import os
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC  # Import Support Vector Classifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

# --- Genetic Distance Calculation and Clustering (from Genetic_distance_Kinship.py) ---

# Load SNP genotype data
# Assumes SNP data file is named "nature_GWAS_SNP_RS.tsv" and is tab-separated.
df = pd.read_csv("./nature_GWAS_SNP_RS.tsv", sep="\t")

# Extract SNP genotype data, excluding the first column (sample name) and the last column (phenotype)
snp_data = df.iloc[:, 1:-1]  #
# Extract sample IDs from the first column of the DataFrame
sample_ids = df.iloc[:, 0].tolist()

# Get the number of samples
n_samples = len(snp_data)
# Initialize a zero matrix for genetic distances between all pairs of samples
distance_matrix = np.zeros((n_samples, n_samples))

# Compute pairwise genetic distance matrix
# Iterate through all sample pairs (i, j)
for i in range(n_samples):
    for j in range(i): # Only compute the lower triangle, as the distance matrix is symmetric
        vec_i = snp_data.iloc[i]
        vec_j = snp_data.iloc[j]
        # Create a mask to exclude SNP sites that are NaN (missing values) in either sample
        mask = ~(vec_i.isna() | vec_j.isna())
        # Calculate the total number of common (non-missing) SNP sites
        total = mask.sum()
        # Calculate genetic distance: 1 - (number of identical sites / total common sites)
        # If there are no common sites, distance is set to NaN
        distance = 1 - (vec_i[mask] == vec_j[mask]).sum() / total if total > 0 else np.nan
        # Store the calculated distance symmetrically in the distance matrix
        distance_matrix[i, j] = distance
        distance_matrix[j, i] = distance

# Save the lower triangular distance matrix in MEGA format
# MEGA format is a common file format for phylogenetic analysis
with open("Genetic_distance_lower_triangle.txt", "w") as f:
    f.write("#mega\n")
    f.write("TITLE: Lower-left triangular matrix of genetic distances\n\n")
    # Write each sample ID
    for sid in sample_ids:
        f.write(f"#{sid}\n")
    # Write the lower triangular part of the distance matrix
    for i in range(1, n_samples):
        row = " ".join(f"{distance_matrix[i, j]:.5f}" for j in range(i))
        f.write(row + "\n")

# Build a pairwise distance dictionary for easy lookup
pairwise_distance = {}
for i in range(n_samples):
    for j in range(i):
        pairwise_distance[f"{i}_{j}"] = distance_matrix[i, j]

# Function for clustering samples based on a distance threshold
def cluster_samples_by_threshold(distance_dict, threshold, sample_ids):
    # Initialize sets and lists for clustering results
    grouped = set() # Indices of samples already grouped
    all_grouped_ids = set() # Global IDs of all grouped samples
    group_list = [] # List to store sample indices for each cluster

    # Filter valid pairs where distance is less than or equal to the threshold
    valid_pairs = {k: v for k, v in distance_dict.items() if v <= threshold}
    # Build a graph where nodes are sample indices and edges represent distances below the threshold
    graph = defaultdict(set)
    for k in valid_pairs:
        i, j = map(int, k.split('_'))
        graph[i].add(j) # Add bidirectional edge
        graph[j].add(i) #

    # Depth-First Search (DFS) function to find connected components (clusters)
    def dfs(node, visited, cluster):
        visited.add(node) # Mark current node as visited
        cluster.append(node) # Add current node to the current cluster
        for neighbor in graph[node]: # Iterate through neighbors of the current node
            if neighbor not in visited: # If a neighbor has not been visited, recursively call DFS
                dfs(neighbor, visited, cluster)

    visited = set() # Set to keep track of visited nodes for DFS
    # Iterate through all samples, find unvisited nodes, and start a new DFS to form a new cluster
    for node in range(n_samples):
        if node not in visited and node in graph: # Ensure node has connections in the graph (i.e., not isolated)
            cluster = []
            dfs(node, visited, cluster)
            group_list.append(sorted(cluster)) # Add the found cluster (sorted by index) to the list
            all_grouped_ids.update(cluster) # Update the set of all grouped samples

    # Identify samples that were not grouped
    ungrouped = sorted(set(range(n_samples)) - all_grouped_ids)
    return group_list, ungrouped

# Define different distance thresholds and run clustering for each
thresholds = [0.3, 0.35, 0.4]
cluster_results = {} # Dictionary to store clustering results for different thresholds

for thresh in thresholds:
    clusters, ungrouped = cluster_samples_by_threshold(pairwise_distance, thresh, sample_ids)
    # Convert sample indices in clusters to actual sample IDs
    clustered = [sample_ids[i] for group in clusters for i in group]
    cluster_results[thresh] = {
        "clusters": [[sample_ids[i] for i in group] for group in clusters], # Store sample IDs within each cluster
        "ungrouped": [sample_ids[i] for i in ungrouped], # Store IDs of ungrouped samples
        "clustered": clustered # Store IDs of all clustered samples
    }

# Generate summary statistics for clustering results
summary_df = pd.DataFrame({
    "Threshold": thresholds,
    "Cluster Count": [len(cluster_results[t]["clusters"]) for t in thresholds], # Number of clusters for each threshold
    "Clustered Samples Count": [len(cluster_results[t]["clustered"]) for t in thresholds], # Total number of clustered samples for each threshold
    "Ungrouped Samples Count": [len(cluster_results[t]["ungrouped"]) for t in thresholds] # Total number of ungrouped samples for each threshold
})
summary_df.to_csv("Genetic_distance_summary.csv", index=False)
print("Summary saved as Genetic_distance_summary.csv")

# Output detailed clustering results by threshold to a text file
with open("Genetic_distance_summary_by_threshold.txt", "w") as f:
    for thresh in thresholds:
        f.write(f"# Clustering results (distance <= {thresh})\n\n")
        for i, group in enumerate(cluster_results[thresh]["clusters"], 1):
            f.write(f"Cluster {i}:\t" + ", ".join(str(x) for x in group) + "\n")
        f.write("\n# Ungrouped samples:\n")
        f.write(", ".join(str(x) for x in cluster_results[thresh]["ungrouped"]) + "\n")
        f.write("\n" + "="*60 + "\n\n")
print("Cluster detail summary saved as Genetic_distance_summary_by_threshold.txt")

# Output genotype/phenotype data if the clustered sample count meets the criteria, for downstream model training
generated_data_file = "" # Variable to store the name of the generated file
for thresh in thresholds:
    suffix = str(thresh).replace('.', '_') # Convert threshold to a filename suffix (e.g., 0.35 -> 0_35)
    clustered_ids = cluster_results[thresh]["clustered"]
    # Save genotype/phenotype data for this threshold if the number of clustered samples is between 100 and 139 (inclusive)
    if 100 <= len(clustered_ids) < 140:
        # Construct the output filename without specific model names
        file_name_for_model = f"Genetic_distance_genotype_phenotype_threshold_{suffix}.tsv"
        # Filter the DataFrame to include only the clustered samples
        clustered_df = df[df.iloc[:, 0].isin(clustered_ids)]
        clustered_df.to_csv(file_name_for_model, sep="\t", index=False)
        print(f"Genotype/phenotype data saved for threshold {thresh}: {file_name_for_model}")
        # Capture the name of the first suitable file generated for use in subsequent model training
        generated_data_file = file_name_for_model
        break # Stop after finding the first suitable file, as we typically need only one input file for the next step

# --- Model Training Section (Support Vector Classifier) ---
# This block of code will only execute if a suitable data file was generated by the preceding clustering step.

if generated_data_file: #
    # Set the data directory
    data_dir = './'
    # Use the dynamically generated genotype/phenotype file as input
    file_path = os.path.join(data_dir, generated_data_file)
    data = pd.read_csv(file_path, index_col=0, sep='\t')

    # Prepare feature matrix X and label vector y
    X = data.drop(columns=['rank'])
    y = data['rank'].map({'R': 0, 'S': 1})

    # Initialize Stratified 10-fold Cross-Validation
    # StratifiedKFold ensures that the class proportions in each fold are maintained consistent with the original dataset.
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2025)
    results_for_run = [] # List to store metrics for each cross-validation run
    all_results = [] # List to store the final summarized results

    # Support Vector Classifier hyperparameter grid
    SVC_param_grid = {
        'C': [0.1, 1, 10, 100], # Regularization parameter
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'], # Specifies the kernel type to be used in the algorithm
        'gamma': ['scale', 'auto', 0.01, 0.1, 1], # Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
        'degree': [2, 3, 4], # Degree of the polynomial kernel function ('poly'). Ignored by other kernels.
        'coef0': [0.0, 0.1, 0.5, 1.0], # Independent term in kernel function. It is only significant in 'poly' and 'sigmoid'.
        'class_weight': [None, 'balanced'], # Weights associated with classes
        'probability': [True] # Whether to enable probability estimates (required for ROC AUC)
    }

    # Perform hyperparameter tuning using GridSearchCV
    # estimator: The model to be optimized
    # param_grid: The hyperparameter grid to search over
    # cv: Cross-validation strategy
    # scoring: Metric used to evaluate the performance of the cross-validated model
    # n_jobs: Number of jobs to run in parallel (-1 means use all available CPU cores)
    grid_search = GridSearchCV(
        estimator=SVC(random_state=42), # Instantiate Support Vector Classifier with random state
        param_grid=SVC_param_grid,
        cv=skf,
        scoring='accuracy',
        n_jobs=-1
    )

    # Execute grid search on the complete dataset to find the best hyperparameter combination
    grid_search.fit(X, y)
    best_params = grid_search.best_params_ # Get the best hyperparameters found

    # Perform 10-fold cross-validation using the best parameters
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index] # Feature sets for training and testing
        y_train, y_test = y.iloc[train_index], y.iloc[test_index] # Label sets for training and testing

        # Initialize and train the SVC model with the best parameters
        model = SVC(**best_params, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test) # Predict classes
        y_prob = model.predict_proba(X_test)[:, 1] # Predict probabilities for the positive class

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_value = roc_auc_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Store the evaluation results for the current fold
        results_for_run.append((accuracy, roc_value, precision, recall, f1))

    # Calculate average metrics across all folds
    average_accuracy = np.mean([acc for acc, _, _, _, _ in results_for_run])
    average_roc = np.mean([roc for _, roc, _, _, _ in results_for_run])
    average_precision = np.mean([prec for _, _, prec, _, _ in results_for_run])
    average_recall = np.mean([rec for _, _, _, rec, _ in results_for_run])
    average_f1 = np.mean([f1 for _, _, _, _, f1 in results_for_run])

    # Collect final results, including filename, results per run, and average metrics
    all_results.append((generated_data_file, results_for_run, average_roc, average_precision, average_recall, average_f1))

    # Write model training results to a specific file for SVC
    with open('out_result_Nature_GWAS_SVC_Kinship_endadta.txt', 'a') as file:
        # Output best parameters
        print(f"Nature_GWAS_SVC_Kinship_Best_parameters: {best_params}", file=file)

        for file_name, results, avg_roc, avg_prec, avg_recall, avg_f1 in all_results:
            avg_acc = np.mean([acc for acc, _, _, _, _ in results])
            # Print summarized average metrics
            print(f">{file_name}:Average-Accuracy:{avg_acc:.5f}"
                  f":Average-ROC-AUC-Score:{avg_roc:.5f}"
                  f":Average-Precision:{avg_prec:.5f}"
                  f":Average-Recall:{avg_recall:.5f}"
                  f":Average-F1-Score:{avg_f1:.5f}", file=file)

            # Print detailed metrics for each cross-validation fold
            for i, (accuracy, roc, precision, recall, f1) in enumerate(results, start=1):
                print(f"{i}:Accuracy:{accuracy:.5f}"
                      f":ROC-AUC:{roc:.5f}"
                      f":Precision:{precision:.5f}"
                      f":Recall:{recall:.5f}"
                      f":F1-Score:{f1:.5f}", file=file)
else:
    # If no suitable clustered data file was generated, print an informative message
    print("No suitable clustered data file generated for model training (clustered sample count not >= 100).")