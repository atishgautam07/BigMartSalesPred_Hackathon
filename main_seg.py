"""
BigMart Sales Prediction Pipeline - Segmented Models Version
Main script to run the complete ML pipeline with segmented Random Forest models,
custom weighted RMSE loss, and Box-Cox transformation.
"""

from typing import Dict
import pandas as pd
import numpy as np
import os
from pathlib import Path
import time
from datetime import datetime

from sklearn.metrics import mean_squared_error

# Import custom modules (updated imports)
from src.ingest.ingest import load_and_preprocess_data, DataPreprocessor
from src.transform.feature_engineering import create_features
from src.transform.feature_encoder import encode_features
from src.train.segmented_model_trainer import SegmentedModelTrainer, train_segmented_models


class BigMartSegmentedPipeline:
    """
    Complete pipeline for BigMart sales prediction using segmented models.
    Handles data flow from raw inputs to final predictions with separate models
    for different outlet types, outlet identifiers, and MRP bins.
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
        self.preprocessor = None  # Store preprocessor for segment extraction

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
        Run the complete segmented ML pipeline.
        
        Args:
            skip_steps: List of step names to skip if already completed
        """
        skip_steps = skip_steps or []
        
        print("="*60)
        print("BigMart Segmented Sales Prediction Pipeline")
        print("Features: Random Forest Models per Segment + Weighted RMSE + Box-Cox")
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
            
        # Step 4: Segmented Model Training and Prediction
        print("\n[Step 4/4] Segmented Model Training and Prediction")
        predictions = self._run_segmented_model_training(X_train, X_test, y_train, train_features, test_features)
        self.run_info['steps_completed'].append('modeling')
        
        # Create final submission
        self._create_submission(test_clean, predictions)
        
        # Print summary
        self._print_summary()
        
    def _run_data_ingestion(self) -> tuple:
        """Run data ingestion and cleaning step."""
        start_time = time.time()
        
        # Initialize preprocessor and store it for later use
        self.preprocessor = DataPreprocessor()
        
        # Load and preprocess data
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
    
    def _run_segmented_model_training(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                                y_train: pd.Series, train_features: pd.DataFrame, 
                                test_features: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Run segmented model training and prediction step with multiple outputs."""
        start_time = time.time()
        
        print("Preparing segment information...")
        
        if self.preprocessor is None:
            self.preprocessor = DataPreprocessor()
            self.preprocessor.fit(train_features)
        
        segment_info_train = self.preprocessor.get_segment_info(train_features)
        segment_info_test = self.preprocessor.get_segment_info(test_features)
        
        print(f"Train segments shape: {segment_info_train.shape}")
        print(f"Test segments shape: {segment_info_test.shape}")
        
        # Print segment distributions
        print("\nSegment distributions:")
        for col in ['Outlet_Type', 'Item_MRP_Bins']:
            if col in segment_info_train.columns:
                print(f"\n{col} distribution (train):")
                print(segment_info_train[col].value_counts())
        
        # Save segment information
        segment_info_train.to_csv(self.paths['processed'] / 'segment_info_train.csv', index=False)
        segment_info_test.to_csv(self.paths['processed'] / 'segment_info_test.csv', index=False)
        
        # Train segmented models with updated trainer
        scores, all_predictions, trainer = train_segmented_models(
            X_train_path=str(self.paths['processed'] / 'train_encoded.csv'),
            y_train=y_train,
            X_test_path=str(self.paths['processed'] / 'test_encoded.csv'),
            segment_info_train=segment_info_train,
            segment_info_test=segment_info_test,
            output_dir=str(self.paths['models']),
            optimize_hyperparams=self.optimize_hyperparams,
            n_trials=self.n_trials
        )
        
        # Save comprehensive results
        self._save_segmented_scores(scores)
        self._save_train_predictions(trainer.train_predictions, train_features, y_train)
        self._save_feature_selection_results(trainer)
        
        elapsed = time.time() - start_time
        print(f"Segmented model training completed in {elapsed:.1f} seconds")
        
        return all_predictions
    
    def _save_segmented_scores(self, scores: Dict[str, Dict]):
        """Save segmented model scores to files."""
        
        # Create comprehensive scores summary
        all_scores = []
        
        for segment_type, segment_scores in scores.items():
            for segment, score in segment_scores.items():
                all_scores.append({
                    'Segment_Type': segment_type,
                    'Segment': segment,
                    'Weighted_RMSE': score
                })
        
        scores_df = pd.DataFrame(all_scores)
        scores_df.to_csv(self.paths['logs'] / 'segmented_model_scores.csv', index=False)
        
        # Create summary by segment type
        summary_scores = []
        for segment_type, segment_scores in scores.items():
            avg_score = np.mean(list(segment_scores.values()))
            min_score = np.min(list(segment_scores.values()))
            max_score = np.max(list(segment_scores.values()))
            n_models = len(segment_scores)
            
            summary_scores.append({
                'Segment_Type': segment_type,
                'Avg_Weighted_RMSE': avg_score,
                'Min_Weighted_RMSE': min_score,
                'Max_Weighted_RMSE': max_score,
                'Num_Models': n_models
            })
        
        summary_df = pd.DataFrame(summary_scores)
        summary_df.to_csv(self.paths['logs'] / 'segment_summary_scores.csv', index=False)
        
        print("\nSegmented Model Performance Summary:")
        print(summary_df.to_string(index=False))
    
    def _create_submission(self, test_data: pd.DataFrame, all_predictions: Dict[str, np.ndarray]):
        """Create multiple submission files with different prediction methods."""
        print("\nCreating multiple submission files...")
        
        # Base submission data
        base_submission = pd.DataFrame({
            'Item_Identifier': test_data['Item_Identifier'],
            'Outlet_Identifier': test_data['Outlet_Identifier']
        })
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create individual submission files
        submission_files = {}
        
        submission_mapping = {
            'outlet_type': 'submission_outlet_type_segments',
            'outlet_identifier': 'submission_outlet_id_segments', 
            'mrp_bins': 'submission_mrp_bin_segments',
            'best_single': 'submission_best_single_model',
            'combined': 'submission_weighted_average'
        }
        
        for pred_type, predictions in all_predictions.items():
            if pred_type in submission_mapping:
                # Ensure no negative predictions
                predictions_clean = np.maximum(predictions, 0)
                
                submission = base_submission.copy()
                submission['Item_Outlet_Sales'] = predictions_clean
                
                # Save with timestamp
                filename = f"{submission_mapping[pred_type]}_{timestamp}.csv"
                filepath = self.paths['submissions'] / filename
                submission.to_csv(filepath, index=False)
                
                # Also save as latest
                latest_filename = f"{submission_mapping[pred_type]}_latest.csv"
                latest_filepath = self.paths['submissions'] / latest_filename
                submission.to_csv(latest_filepath, index=False)
                
                submission_files[pred_type] = {
                    'filepath': filepath,
                    'predictions': predictions_clean
                }
                
                print(f"✓ {submission_mapping[pred_type]}.csv")
        
        # Create submission comparison summary
        self._create_submission_summary(submission_files, timestamp)
        
        print(f"\nAll submissions saved to: {self.paths['submissions']}")
    
    def _analyze_predictions(self, test_data: pd.DataFrame, predictions: np.ndarray, output_dir: Path):
        """Create detailed prediction analysis."""
        print("Creating prediction analysis...")
        
        # Load segment information
        segment_info = pd.read_csv(self.paths['processed'] / 'segment_info_test.csv')
        
        # Create analysis dataframe
        analysis_df = pd.DataFrame({
            'Item_Identifier': test_data['Item_Identifier'],
            'Outlet_Identifier': test_data['Outlet_Identifier'],
            'Outlet_Type': segment_info['Outlet_Type'],
            'Item_MRP_Bins': segment_info['Item_MRP_Bins'],
            'Predicted_Sales': predictions
        })
        
        # Analysis by segment type
        segment_analysis = []
        
        for segment_type in ['Outlet_Type', 'Item_MRP_Bins']:
            segment_stats = analysis_df.groupby(segment_type)['Predicted_Sales'].agg([
                'count', 'mean', 'std', 'min', 'max', 'median'
            ]).round(2)
            
            segment_stats['Segment_Type'] = segment_type
            segment_stats['Segment'] = segment_stats.index
            segment_analysis.append(segment_stats.reset_index(drop=True))
        
        # Combine and save analysis
        full_analysis = pd.concat(segment_analysis, ignore_index=True)
        full_analysis.to_csv(output_dir / 'prediction_analysis_by_segment.csv', index=False)
        
        # Save detailed predictions
        analysis_df.to_csv(output_dir / 'detailed_predictions.csv', index=False)
        
        print("Prediction analysis completed!")
    
    def _save_train_predictions(self, train_predictions: pd.DataFrame, 
                           train_features: pd.DataFrame, y_train: pd.Series):
        """Save training predictions for analysis."""
        print("Saving training predictions...")
        
        # Combine with original data
        train_analysis = pd.DataFrame({
            'Item_Identifier': train_features['Item_Identifier'],
            'Outlet_Identifier': train_features['Outlet_Identifier'],
            'Outlet_Type': train_features['Outlet_Type'],
            'Item_MRP_Bins': train_features['Item_MRP_Bins'],
            'Actual_Sales': y_train
        })
        
        # Add predictions
        for col in train_predictions.columns:
            train_analysis[col] = train_predictions[col]
        
        # Calculate residuals for each prediction type
        for pred_col in train_predictions.columns:
            residual_col = f'residual_{pred_col.replace("pred_", "")}'
            train_analysis[residual_col] = train_analysis['Actual_Sales'] - train_analysis[pred_col]
        
        # Calculate prediction accuracy metrics
        metrics_summary = []
        for pred_col in train_predictions.columns:
            pred_name = pred_col.replace('pred_', '')
            rmse = np.sqrt(mean_squared_error(train_analysis['Actual_Sales'], train_analysis[pred_col]))
            mae = np.mean(np.abs(train_analysis['Actual_Sales'] - train_analysis[pred_col]))
            
            metrics_summary.append({
                'Model': pred_name,
                'RMSE': rmse,
                'MAE': mae
            })
        
        metrics_df = pd.DataFrame(metrics_summary)
        
        # Save files
        train_analysis.to_csv(self.paths['processed'] / 'train_predictions_analysis.csv', index=False)
        metrics_df.to_csv(self.paths['logs'] / 'train_prediction_metrics.csv', index=False)
        
        print("Training predictions saved for analysis!")
        print("\nTrain Prediction Metrics:")
        print(metrics_df.to_string(index=False))

    def _save_feature_selection_results(self, trainer):
        """Save feature selection results for analysis."""
        if not hasattr(trainer, 'selected_features') or not trainer.selected_features:
            return
        
        print("Saving feature selection results...")
        
        # Compile feature selection summary
        feature_summary = []
        
        for segment_type, segment_features in trainer.selected_features.items():
            for segment, features in segment_features.items():
                feature_summary.append({
                    'Segment_Type': segment_type,
                    'Segment': segment,
                    'Selected_Features_Count': len(features),
                    'Selected_Features': ', '.join(features[:10]) + ('...' if len(features) > 10 else '')
                })
        
        feature_summary_df = pd.DataFrame(feature_summary)
        feature_summary_df.to_csv(self.paths['logs'] / 'feature_selection_summary.csv', index=False)
        
        # Save detailed feature lists
        detailed_features = {}
        for segment_type, segment_features in trainer.selected_features.items():
            detailed_features[segment_type] = segment_features
        
        import json
        with open(self.paths['logs'] / 'selected_features_detailed.json', 'w') as f:
            json.dump(detailed_features, f, indent=2)
        
        print("Feature selection results saved!")
        print(f"\nFeature Selection Summary:")
        print(feature_summary_df.head(10).to_string(index=False))
        if len(feature_summary_df) > 10:
            print(f"... and {len(feature_summary_df) - 10} more segments")

    # MODIFY the _create_submission method to handle multiple predictions
    def _create_submission(self, test_data: pd.DataFrame, all_predictions: Dict[str, np.ndarray]):
        """Create multiple submission files with different prediction methods."""
        print("\nCreating multiple submission files...")
        
        # Base submission data
        base_submission = pd.DataFrame({
            'Item_Identifier': test_data['Item_Identifier'],
            'Outlet_Identifier': test_data['Outlet_Identifier']
        })
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create individual submission files
        submission_files = {}
        
        submission_mapping = {
            'outlet_type': 'submission_outlet_type_segments',
            'outlet_identifier': 'submission_outlet_id_segments', 
            'mrp_bins': 'submission_mrp_bin_segments',
            'best_single': 'submission_best_single_model',
            'combined': 'submission_weighted_average'
        }
        
        for pred_type, predictions in all_predictions.items():
            if pred_type in submission_mapping:
                # Ensure no negative predictions
                predictions_clean = np.maximum(predictions, 0)
                
                submission = base_submission.copy()
                submission['Item_Outlet_Sales'] = predictions_clean
                
                # Save with timestamp
                filename = f"{submission_mapping[pred_type]}_{timestamp}.csv"
                filepath = self.paths['submissions'] / filename
                submission.to_csv(filepath, index=False)
                
                # Also save as latest
                latest_filename = f"{submission_mapping[pred_type]}_latest.csv"
                latest_filepath = self.paths['submissions'] / latest_filename
                submission.to_csv(latest_filepath, index=False)
                
                submission_files[pred_type] = {
                    'filepath': filepath,
                    'predictions': predictions_clean
                }
                
                print(f"✓ {submission_mapping[pred_type]}.csv")
        
        # Create submission comparison summary
        self._create_submission_summary(submission_files, timestamp)
        
        print(f"\nAll submissions saved to: {self.paths['submissions']}")

    def _create_submission_summary(self, submission_files: Dict, timestamp: str):
        """Create a summary comparing all submission approaches."""
        
        summary_data = []
        
        for pred_type, file_info in submission_files.items():
            predictions = file_info['predictions']
            
            summary_data.append({
                'Submission_Type': pred_type,
                'Mean_Prediction': predictions.mean(),
                'Std_Prediction': predictions.std(),
                'Min_Prediction': predictions.min(),
                'Max_Prediction': predictions.max(),
                'Median_Prediction': np.median(predictions),
                'Zero_Predictions': (predictions == 0).sum(),
                'File_Path': file_info['filepath'].name
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary
        summary_path = self.paths['logs'] / f'submission_comparison_{timestamp}.csv'
        summary_df.to_csv(summary_path, index=False)
        
        # Also save as latest
        latest_summary_path = self.paths['logs'] / 'submission_comparison_latest.csv'
        summary_df.to_csv(latest_summary_path, index=False)
        
        print("\nSubmission Comparison Summary:")
        print(summary_df.round(2).to_string(index=False))
        
        # Identify best and worst predictions by variance
        most_confident = summary_df.loc[summary_df['Std_Prediction'].idxmin(), 'Submission_Type']
        least_confident = summary_df.loc[summary_df['Std_Prediction'].idxmax(), 'Submission_Type']
        
        print(f"\nMost confident model (lowest std): {most_confident}")
        print(f"Least confident model (highest std): {least_confident}")
    
    def _print_summary(self):
        """Print pipeline execution summary."""
        end_time = datetime.now()
        total_time = (end_time - self.run_info['start_time']).total_seconds()
        
        print("\n" + "="*60)
        print("Segmented Pipeline Execution Summary")
        print("="*60)
        print(f"Pipeline Type: Segmented Random Forest Models")
        print(f"Loss Function: Weighted RMSE")
        print(f"Target Transform: Box-Cox")
        print(f"Segments: Outlet Type, Outlet ID, MRP Bins")
        print(f"Feature Selection: RF Importance + RFE")
        print(f"Start time: {self.run_info['start_time']}")
        print(f"End time: {end_time}")
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Steps completed: {', '.join(self.run_info['steps_completed'])}")
        
        print("\nOutput files generated:")
        print(f"  - Cleaned data: {self.paths['processed']}")
        print(f"  - Featured data: {self.paths['processed']}")
        print(f"  - Segmented models: {self.paths['models']}")
        print(f"  - Train predictions: {self.paths['processed'] / 'train_predictions_analysis.csv'}")
        print(f"  - Feature selection results: {self.paths['logs']}")
        print(f"  - Multiple submissions: {self.paths['submissions']}")
        print(f"  - Model analysis: {self.paths['logs']}")
        
        # Show submission files created
        submission_files = [
            'submission_outlet_type_segments_latest.csv',
            'submission_outlet_id_segments_latest.csv',
            'submission_mrp_bin_segments_latest.csv',
            'submission_best_single_model_latest.csv',
            'submission_weighted_average_latest.csv'
        ]
        
        print("\nSubmission files created:")
        for filename in submission_files:
            filepath = self.paths['submissions'] / filename
            if filepath.exists():
                print(f"  ✓ {filename}")
        
        print("="*60)
        
        # Print final model information
        if (self.paths['logs'] / 'segment_summary_scores.csv').exists():
            print("\nFinal Model Performance:")
            summary_df = pd.read_csv(self.paths['logs'] / 'segment_summary_scores.csv')
            print(summary_df.to_string(index=False))
        
        # Print train prediction metrics if available
        if (self.paths['logs'] / 'train_prediction_metrics.csv').exists():
            print("\nTrain Prediction Performance:")
            train_metrics = pd.read_csv(self.paths['logs'] / 'train_prediction_metrics.csv')
            print(train_metrics.to_string(index=False))
        
        # Print feature selection summary if available
        if (self.paths['logs'] / 'feature_selection_summary.csv').exists():
            print("\nFeature Selection Summary (Top 5 segments):")
            feature_summary = pd.read_csv(self.paths['logs'] / 'feature_selection_summary.csv')
            print(feature_summary.head().to_string(index=False))
        
        print("\nAll outputs ready for analysis and submission!")

def main():
    """Main entry point for the segmented pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='BigMart Segmented Sales Prediction Pipeline')
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
    pipeline = BigMartSegmentedPipeline(
        args.data_path, 
        optimize_hyperparams=not args.no_hyperparam_tuning,
        n_trials=args.n_trials
    )
    pipeline.run_pipeline(skip_steps=args.skip_steps)


if __name__ == "__main__":
    # For direct execution without command line args
    data_path = "/Users/whysocurious/Documents/MLDSAIProjects/SalesPred_Hackathon/data"
    
    # Run with hyperparameter optimization for segmented models
    pipeline = BigMartSegmentedPipeline(data_path, optimize_hyperparams=True, n_trials=200)
    
    # Run full pipeline
    pipeline.run_pipeline()
    
    # Or run without hyperparameter tuning for faster results
    # pipeline = BigMartSegmentedPipeline(data_path, optimize_hyperparams=False)
    # pipeline.run_pipeline(skip_steps=['ingest', 'features'])