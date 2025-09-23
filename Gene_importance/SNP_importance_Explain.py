# Used iterative SNP deletion to estimate the importance coefficient (IC)
# Written by HY on 2025/1/2 in Beijing

import sklearn.ensemble
import sklearn.linear_model
from sklearn.metrics import accuracy_score
import numpy as np
from typing import List, Dict, Tuple
import sklearn.svm
import sklearn.tree
from tqdm import tqdm
import argparse
from sklearn.neural_network import MLPClassifier
import pandas as pd
import logging
from pathlib import Path
import json
import sklearn
import time
import lightgbm
import xgboost

def cross_entp(v1, v2):
    epsilon = 1e-10
    v1 = v1 + epsilon
    v2 = v2 + epsilon
    v1 = v1 / np.sum(v1)
    v2 = v2 / np.sum(v2)
    return -np.sum(v1 * np.log(v2))

model_dict = {'MLPClassifier': MLPClassifier, 
            'RandomForestClassifier': sklearn.ensemble.RandomForestClassifier,
            'LogisticRegression': sklearn.linear_model.LogisticRegression,
            'LGBMClassifier': lightgbm.LGBMClassifier, 
            'XGBClassifier': xgboost.XGBClassifier,
            'DecisionTreeClassifier': sklearn.tree.DecisionTreeClassifier, 
            'SVC': sklearn.svm.SVC, 
            'SGDClassifier': sklearn.linear_model.SGDClassifier}

# def sort_snp_list(snp_list):
#     return sorted(snp_list, key=lambda x: (int(x.split('_')[0][3:]), int(x.split('_')[1])))

def split_train_test_data(dat):
    train_data = dat['train']
    test_data = dat['test']
    x_train_dat = train_data.iloc[:, :-1]  # All columns except last for features
    y_train_dat = train_data.iloc[:, -1].replace({'R': 0, 'S': 1}).astype(np.int8)  # Last column for target
    x_test_dat = test_data.iloc[:, :-1]  # All columns except last for features
    y_test_dat = test_data.iloc[:, -1].replace({'R': 0, 'S': 1}).astype(np.int8)  # Last column for target
    return x_train_dat, y_train_dat, x_test_dat, y_test_dat


class IC:
    def __init__(self, 
                ml_model,
                data: Dict[str, pd.DataFrame], 
                model_param: Dict) -> None:
        """Initialize IC calculator with model and data.
        
        Args:
            ml_model: Machine learning model class
            data: Dictionary containing 'train' and 'test' DataFrames
            model_param: Dictionary of model parameters
        """
        self.model = ml_model
        self.data = data
        self.model_param = model_param
        
        # Cache split data to avoid repeated splitting
        self._cached_split_data = split_train_test_data(data)
        
        # Calculate baseline performance
        self.baseline_acc, self.baseline_ypred = self._calculate_baseline_accuracy()
        self.snps = data['train'].columns.tolist()[:-1]
        logging.info(f'Initialized IC calculator with {len(self.snps)} SNPs')

    @property
    def get_ypred_res(self):
        """Return ypred results for all SNPs."""
        return self._ypred_res

    def _train_and_evaluate_model(self, x_train: pd.DataFrame, y_train: pd.Series, 
                                x_test: pd.DataFrame, y_test: pd.Series) -> np.ndarray:
        """Train model and make predictions on test data.
        
        Args:
            x_train: Training features
            y_train: Training labels
            x_test: Test features
            y_test: Test labels
            
        Returns:
            np.ndarray: Predicted labels
        """
        model_instance = self.model(**self.model_param)
        model_instance.fit(x_train, y_train)
        return model_instance.predict(x_test)

    def _calculate_baseline_accuracy(self) -> Tuple[float, np.ndarray]:
        """Calculate baseline accuracy using all SNPs.
        
        Returns:
            Tuple[float, np.ndarray]: Baseline accuracy and predictions
        """
        x_train, y_train, x_test, y_test = self._cached_split_data
        y_pred = self._train_and_evaluate_model(x_train, y_train, x_test, y_test)
        return accuracy_score(y_test, y_pred), y_pred
    
    def _calculate_snp_importance(self, snp: str) -> Tuple[float, float]:
        """Calculate importance metrics for a single SNP by removing it.
        
        Args:
            snp: SNP to remove and calculate importance for
            
        Returns:
            Tuple[float, float]: Accuracy difference and cross entropy
        """
        x_train, y_train, x_test, y_test = self._cached_split_data
        
        # Remove SNP from features
        x_train_rm = x_train.drop(snp, axis=1)
        x_test_rm = x_test.drop(snp, axis=1)
        
        # Get predictions with SNP removed
        y_pred_rm = self._train_and_evaluate_model(x_train_rm, y_train, x_test_rm, y_test)
        accuracy_rm = accuracy_score(y_test, y_pred_rm)
        
        # Calculate importance metrics
        acc_diff = self.baseline_acc - accuracy_rm
        cross_entropy = cross_entp(self.baseline_ypred, y_pred_rm)
        
        return acc_diff, cross_entropy

    def run(self) -> Dict[str, List[float]]:
        """Calculate importance coefficients for all SNPs.
        
        Returns:
            Dict[str, List[float]]: Dictionary mapping SNP names to their importance metrics
        """
        snp_importance = {}
        
        # Calculate importance for each SNP with progress bar
        for snp in tqdm(self.snps, desc="Calculating SNP importance"):
            acc_diff, cross_entropy = self._calculate_snp_importance(snp)
            snp_importance[snp] = [acc_diff, cross_entropy]
            
        return snp_importance
    

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def validate_file_path(path: str) -> Path:
    """Validate and return Path object for given file path."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path

def main():
    try:
        # Parse arguments with improved help messages and type hints
        parser = argparse.ArgumentParser(description='Calculate SNP importance coefficients using iterative deletion method')
        parser.add_argument('--train', type=str, required=True,
                          help='Path to training data CSV file containing SNP data and labels')
        parser.add_argument('--test', type=str, required=True,
                          help='Path to test data CSV file containing SNP data and labels')
        parser.add_argument('--output', type=str, required=True,
                          help='Output file path for saving importance coefficients in CSV format')
        parser.add_argument('--model_type', type=str, required=False, default='MLPClassifier',
                          choices=['MLPClassifier', 'RandomForestClassifier', 'LogisticRegression',
                                    'LGBMClassifier', 'XGBClassifier', 'DecisionTreeClassifier', 'SVC', 'SGDClassifier'],
                          help='Type of machine learning model to use for importance calculation')
        parser.add_argument('--model_param', type=str, required=False,
                          help='JSON string of model parameters (optional). Example: \'{"hidden_layer_sizes": (100,), "max_iter": 300}\'')
        parser.add_argument('--n_jobs', type=int, default=-1,
                          help='Number of jobs to run in parallel (-1 means using all processors)')
        args = parser.parse_args()

        # Validate input files with more detailed error messages
        train_path = validate_file_path(args.train)
        test_path = validate_file_path(args.test)
        logging.info(f"Validated input files: Train={train_path}, Test={test_path}")

        # Load data with improved error handling and data validation
        logging.info("Loading and validating data...")
        try:
            data = {
                'train': pd.read_csv(train_path, index_col=0),
                'test': pd.read_csv(test_path, index_col=0)
            }
            # Enhanced data validation
            for dataset in ['train', 'test']:
                if data[dataset].empty:
                    raise ValueError(f"{dataset} dataset is empty")
                if data[dataset].isnull().values.any():
                    raise ValueError(f"{dataset} dataset contains missing values")
                # Check for consistent number of features
                if len(data['train'].columns) != len(data['test'].columns):
                    raise ValueError("Train and test datasets have different number of features")
        except Exception as e:
            logging.error(f"Error loading or validating data: {e}")
            raise

        # Model configuration with parameter validation
        model_param = {
            'activation': 'logistic',
            'solver': 'adam',
            'early_stopping': True,
            'hidden_layer_sizes': (1000,),
            'max_iter': 500,
            'random_state': 42,
            'validation_fraction': 0.035,
            'alpha': 0.001,
            'verbose': False,
        }

        # Update model parameters if custom parameters are provided
        if args.model_param:
            try:
                custom_params = json.loads(args.model_param)
                model_param.update(custom_params)
                logging.info(f"Updated model parameters with custom settings: {custom_params}")
            except json.JSONDecodeError as e:
                logging.error(f"Invalid JSON format for model parameters: {e}")
                raise

        # Initialize and run IC calculation with progress tracking
        logging.info("Starting IC calculation...")
        try:
            # Dynamically import the selected model class with better error handling
            try:
                model_class = model_dict[args.model_type]
            except KeyError:
                raise ValueError(f"Model type {args.model_type} not found in model dictionary")
            
            # Initialize IC calculator with timing
            start_time = time.time()
            ic = IC(model_class, data, model_param)
            print(f'the baseline acc is {ic.baseline_acc}')
            
            # Calculate importance coefficients
            res = ic.run()
            
            elapsed_time = time.time() - start_time
            logging.info(f"IC calculation completed successfully in {elapsed_time:.2f} seconds")
        except Exception as e:
            logging.error(f"Error during IC calculation: {e}")
            raise

        # Save results with improved error handling and file validation
        output_path = Path(args.output)
        logging.info(f"Saving results to {output_path}...")
        try:
            # Convert results dictionary to DataFrame with additional metadata
            df = pd.DataFrame.from_dict(res, orient='index', columns=['Accuracy_Difference', 'Cross_Entropy'])
            df.index.name = 'SNP'
            df.reset_index(inplace=True)
            
            # Add comprehensive metadata to the DataFrame
            df['Model_Type'] = args.model_type
            df['Model_Parameters'] = str(model_param)
            df['Train_Samples'] = len(data['train'])
            df['Test_Samples'] = len(data['test'])
            df['Calculation_Time'] = elapsed_time
            
            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as CSV with proper formatting and compression
            df.to_csv(output_path, index=False, float_format='%.6f', compression='gzip' if output_path.suffix == '.gz' else None)
            logging.info(f"Results saved successfully to {output_path}")
        except Exception as e:
            logging.error(f"Error saving results: {e}")
            raise

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        logging.error("Program terminated due to errors")
        raise

if __name__ == '__main__':
    main()