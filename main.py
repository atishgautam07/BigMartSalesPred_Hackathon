"""
BigMart Sales Prediction Pipeline
Main script to run the complete ML pipeline from raw data to predictions.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import time
from datetime import datetime

# Import custom modules
from src.ingest.ingest import load_and_preprocess_data
from src.transform.feature_engineering import create_features
from src.transform.feature_encoder import encode_features
from src.train.model_trainer import train_models


class BigMartPipeline:
    """
    Complete pipeline for BigMart sales prediction.
    Handles data flow from raw inputs to final predictions.
    """
    
    def __init__(self, base_path: str, optimize_hyperparams: bool = True, n_trials: int = 50):
        self.base_path = Path(base_path)
        self.paths = self._setup_directory_structure()
        self.optimize_hyperparams = optimize_hyperparams
        self.n_trials = n_trials
        self.run_info = {
            'start_time': datetime.now(),
            'steps_completed': []
        }

    def _setup_directory_structure(self) -> dict:
        """Create necessary directories and return path dictionary."""
        paths = {
            'raw': self.base_path / 'raw',
            'processed': self.base_path / 'processed',
            'models': self.base_path / 'models',
            'logs': self.base_path / 'logs',
            'submissions': self.base_path / 'submissions'
        }
        
        # Create directories if they don't exist
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
            
        return paths
    
    def run_pipeline(self, skip_steps: list = None):
        """
        Run the complete ML pipeline.
        
        Args:
            skip_steps: List of step names to skip if already completed
        """
        skip_steps = skip_steps or []
        
        print("="*60)
        print("BigMart Sales Prediction Pipeline")
        print(f"Started at: {self.run_info['start_time']}")
        print("="*60)
        
        # Step 1: Data Ingestion and Cleaning
        if 'ingest' not in skip_steps:
            print("\n[Step 1/4] Data Ingestion and Cleaning")
            train_clean, test_clean = self._run_data_ingestion()
            self.run_info['steps_completed'].append('ingest')
        else:
            print("\n[Step 1/4] Loading previously cleaned data...")
            train_clean = pd.read_csv(self.paths['processed'] / 'train_cleaned.csv')
            test_clean = pd.read_csv(self.paths['processed'] / 'test_cleaned.csv')
            
        # Step 2: Feature Engineering
        if 'features' not in skip_steps:
            print("\n[Step 2/4] Feature Engineering")
            train_features, test_features = self._run_feature_engineering()
            self.run_info['steps_completed'].append('features')
        else:
            print("\n[Step 2/4] Loading previously engineered features...")
            train_features = pd.read_csv(self.paths['processed'] / 'train_features.csv')
            test_features = pd.read_csv(self.paths['processed'] / 'test_features.csv')
            
        # Step 3: Feature Encoding
        if 'encoding' not in skip_steps:
            print("\n[Step 3/4] Feature Encoding")
            X_train, X_test, y_train = self._run_feature_encoding()
            self.run_info['steps_completed'].append('encoding')
        else:
            print("\n[Step 3/4] Loading previously encoded features...")
            X_train = pd.read_csv(self.paths['processed'] / 'train_encoded.csv')
            X_test = pd.read_csv(self.paths['processed'] / 'test_encoded.csv')
            y_train = train_features['Item_Outlet_Sales']
            
        # Step 4: Model Training and Prediction
        print("\n[Step 4/4] Model Training and Prediction")
        predictions = self._run_model_training(X_train, X_test, y_train, train_features)
        self.run_info['steps_completed'].append('modeling')
        
        # Create final submission
        self._create_submission(test_clean, predictions)
        
        # Print summary
        self._print_summary()
        
    def _run_data_ingestion(self) -> tuple:
        """Run data ingestion and cleaning step."""
        start_time = time.time()
        
        train_clean, test_clean = load_and_preprocess_data(
            str(self.paths['raw'])
        )
        
        # Save cleaned data
        train_clean.to_csv(self.paths['processed'] / 'train_cleaned.csv', index=False)
        test_clean.to_csv(self.paths['processed'] / 'test_cleaned.csv', index=False)
        
        elapsed = time.time() - start_time
        print(f"Data cleaning completed in {elapsed:.1f} seconds")
        
        return train_clean, test_clean
    
    def _run_feature_engineering(self) -> tuple:
        """Run feature engineering step."""
        start_time = time.time()
        
        train_features, test_features = create_features(
            train_path=str(self.paths['processed'] / 'train_cleaned.csv'),
            test_path=str(self.paths['processed'] / 'test_cleaned.csv'),
            output_train_path=str(self.paths['processed'] / 'train_features.csv'),
            output_test_path=str(self.paths['processed'] / 'test_features.csv')
        )
        
        elapsed = time.time() - start_time
        print(f"Feature engineering completed in {elapsed:.1f} seconds")
        
        return train_features, test_features
    
    def _run_feature_encoding(self) -> tuple:
        """Run feature encoding step."""
        start_time = time.time()
        
        X_train, X_test, y_train = encode_features(
            train_features_path=str(self.paths['processed'] / 'train_features.csv'),
            test_features_path=str(self.paths['processed'] / 'test_features.csv'),
            output_train_path=str(self.paths['processed'] / 'train_encoded.csv'),
            output_test_path=str(self.paths['processed'] / 'test_encoded.csv')
        )
        
        elapsed = time.time() - start_time
        print(f"Feature encoding completed in {elapsed:.1f} seconds")
        
        return X_train, X_test, y_train
    
    def _run_model_training(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                           y_train: pd.Series, train_features: pd.DataFrame) -> np.ndarray:
        """Run model training and prediction step."""
        start_time = time.time()
        
        # Get stratification column
        outlet_types = train_features['Outlet_Type']
        
        scores, predictions, train_pred = train_models(
            X_train_path=str(self.paths['processed'] / 'train_encoded.csv'),
            y_train=y_train,
            X_test_path=str(self.paths['processed'] / 'test_encoded.csv'),
            outlet_types=outlet_types,
            output_dir=str(self.paths['models']),
            optimize_hyperparams=self.optimize_hyperparams,
            n_trials=self.n_trials
        )
        
        # Save model scores
        scores_df = pd.DataFrame(list(scores.items()), columns=['Model', 'RMSE'])
        scores_df.to_csv(self.paths['logs'] / 'model_scores.csv', index=False)
        
        # Create train predictions
        train_df = pd.read_csv(str(self.paths['processed'] / 'train_features.csv'))
        train_df['Pred_Item_Outlet_Sales'] = np.expm1(train_pred)
        train_df['log_sales'] = train_df['Item_Outlet_Sales']
        train_df['Item_Outlet_Sales'] = np.expm1(train_df['Item_Outlet_Sales'])
            
        train_df.to_csv(str(self.paths['processed'] / 'train_predicted.csv'), index=False)
        print("\nTrain predictions file created for analysis!")
        
        elapsed = time.time() - start_time
        print(f"Model training completed in {elapsed:.1f} seconds")
        
        return predictions
    
    def _create_submission(self, test_data: pd.DataFrame, predictions: np.ndarray):
        """Create submission file with predictions."""
        print("\nCreating submission file...")
        
        # Handle log transformation if used
        if 'log_sales' in test_data.columns:
            print("Note: Predictions are on original scale (inverse log applied)")
            
        submission = pd.DataFrame({
            'Item_Identifier': test_data['Item_Identifier'],
            'Outlet_Identifier': test_data['Outlet_Identifier'],
            'Item_Outlet_Sales': np.expm1(predictions)
        })
        
        # Ensure no negative predictions
        submission['Item_Outlet_Sales'] = submission['Item_Outlet_Sales'].clip(lower=0)
        
        # Save with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        submission_path = self.paths['submissions'] / f'submission_{timestamp}.csv'
        submission.to_csv(submission_path, index=False)
        
        # Also save as latest
        submission.to_csv(self.paths['submissions'] / 'submission_latest.csv', index=False)
        
        print(f"Submission saved to: {submission_path}")
        print(f"Prediction statistics:")
        print(f"  Mean: {np.expm1(predictions).mean():.2f}")
        print(f"  Std: {np.expm1(predictions).std():.2f}")
        print(f"  Min: {np.expm1(predictions).min():.2f}")
        print(f"  Max: {np.expm1(predictions).max():.2f}")
        
    def _print_summary(self):
        """Print pipeline execution summary."""
        end_time = datetime.now()
        total_time = (end_time - self.run_info['start_time']).total_seconds()
        
        print("\n" + "="*60)
        print("Pipeline Execution Summary")
        print("="*60)
        print(f"Start time: {self.run_info['start_time']}")
        print(f"End time: {end_time}")
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Steps completed: {', '.join(self.run_info['steps_completed'])}")
        print("\nOutput files:")
        print(f"  - Cleaned data: {self.paths['processed']}")
        print(f"  - Featured data: {self.paths['processed']}")
        print(f"  - Models: {self.paths['models']}")
        print(f"  - Submission: {self.paths['submissions']}")
        print("="*60)


def main():
    """Main entry point for the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='BigMart Sales Prediction Pipeline')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Base path for data directory')
    parser.add_argument('--skip-steps', nargs='+', default=[],
                       choices=['ingest', 'features', 'encoding'],
                       help='Steps to skip if already completed')
    parser.add_argument('--no-hyperparam-tuning', action='store_true',
                       help='Disable hyperparameter optimization')
    parser.add_argument('--n-trials', type=int, default=50,
                       help='Number of trials for hyperparameter optimization')
    
    args = parser.parse_args()
    
    # Initialize and run pipeline
    pipeline = BigMartPipeline(
        args.data_path, 
        optimize_hyperparams=not args.no_hyperparam_tuning,
        n_trials=args.n_trials
    )
    pipeline.run_pipeline(skip_steps=args.skip_steps)


if __name__ == "__main__":
    # For direct execution without command line args
    data_path = "/Users/whysocurious/Documents/MLDSAIProjects/SalesPred_Hackathon/data"
    
    # Run with hyperparameter optimization (default)
    pipeline = BigMartPipeline(data_path, optimize_hyperparams=True, n_trials=150)
    
    # Run full pipeline
    pipeline.run_pipeline()
    
    # Or run without hyperparameter tuning for faster results
    # pipeline = BigMartPipeline(data_path, optimize_hyperparams=False)
    # pipeline.run_pipeline(skip_steps=['ingest', 'features'])