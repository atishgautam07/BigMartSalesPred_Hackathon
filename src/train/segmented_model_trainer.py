import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.preprocessing import PowerTransformer
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


class SegmentedModelTrainer:
    """
    Handles segmented model training with separate Random Forest models for different segments.
    Implements custom weighted RMSE loss and Box-Cox transformation.
    """
    
    def __init__(self, optimize_hyperparams: bool = True, n_trials: int = 50):
        self.segment_models = {}
        self.segment_scores = {}
        self.segment_transformers = {}
        self.segment_weights = {}
        self.feature_importance = {}
        self.optimize_hyperparams = optimize_hyperparams
        self.n_trials = n_trials
        self.best_params = {}
        self.study_results = {}
        self.selected_features = {}  # NEW: Store selected features per segment
        self.individual_predictions = {}  # NEW: Store individual segment predictions
        self.train_predictions = None  # NEW: Store train predictions
        self.global_transformer = None
        self.segment_definitions = {
            'outlet_type': 'Outlet_Type',
            'outlet_identifier': 'Outlet_Identifier', 
            'mrp_bins': 'Item_MRP_Bins'
        }
        
    def train_segmented_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                              segment_info: pd.DataFrame,
                              stratify_col: Optional[pd.Series] = None,
                              val_size: float = 0.2) -> Dict[str, Dict]:
        """
        Train separate Random Forest models for each segment.
        
        Args:
            X_train: Training features
            y_train: Training target
            segment_info: DataFrame with segment information (outlet_type, outlet_id, mrp_bins)
            stratify_col: Column to use for stratified split
            val_size: Validation set size
            
        Returns:
            Dictionary of segment scores
        """
        print("=== Segmented Model Training Pipeline ===")
        
        # Apply Box-Cox transformation to target
        y_train_transformed = self._apply_boxcox_transformation(y_train)
        
        # Create train-validation split
        X_tr, X_val, y_tr, y_val, seg_tr, seg_val = self._create_validation_split(
            X_train, y_train_transformed, segment_info, stratify_col, val_size
        )
        
        # Train models for each segment type
        for segment_type, segment_col in self.segment_definitions.items():
            print(f"\n=== Training {segment_type.upper()} Segmented Models ===")
            self._train_segment_models(segment_type, segment_col, X_tr, X_val, 
                                     y_tr, y_val, seg_tr, seg_val)
        
        return self.segment_scores
    
    def _apply_boxcox_transformation(self, y_train: pd.Series) -> pd.Series:
        """Apply Box-Cox transformation to target variable."""
        print("Applying Box-Cox transformation to target variable...")
        
        # Ensure all values are positive (Box-Cox requirement)
        y_positive = y_train + 1  # Add 1 to handle zeros
        
        # Apply Box-Cox transformation
        self.global_transformer = PowerTransformer(method='box-cox', standardize=False)
        y_transformed = self.global_transformer.fit_transform(y_positive.values.reshape(-1, 1))
        y_transformed = pd.Series(y_transformed.flatten(), index=y_train.index)
        
        # Check transformation effectiveness
        original_skew = y_train.skew()
        transformed_skew = y_transformed.skew()
        print(f"Original target skewness: {original_skew:.3f}")
        print(f"Box-Cox transformed skewness: {transformed_skew:.3f}")
        
        return y_transformed
    
    def _inverse_boxcox_transformation(self, y_transformed: np.ndarray) -> np.ndarray:
        """Apply inverse Box-Cox transformation."""
        y_original = self.global_transformer.inverse_transform(y_transformed.reshape(-1, 1))
        return y_original.flatten() - 1  # Subtract 1 to reverse the +1 applied before transformation
    
    def _weighted_rmse_loss(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           sample_weights: np.ndarray = None) -> float:
        """
        Calculate weighted RMSE loss function.
        Higher weights for higher sales values to focus on important predictions.
        """
        if sample_weights is None:
            # Create weights based on target values (higher sales get higher weights)
            y_true_original = self._inverse_boxcox_transformation(y_true)
            sample_weights = np.sqrt(y_true_original + 1)  # Square root weighting
            sample_weights = sample_weights / sample_weights.mean()  # Normalize
        
        # Calculate weighted squared errors
        squared_errors = (y_true - y_pred) ** 2
        weighted_squared_errors = squared_errors * sample_weights
        
        # Return weighted RMSE
        return np.sqrt(weighted_squared_errors.mean())
    
    def _select_features_for_segment(self, X_seg: pd.DataFrame, y_seg: pd.Series, 
                               segment_name: str, min_features: int = 10) -> List[str]:
        """
        Select optimal features for a specific segment using RF importance + RFE.
        
        Args:
            X_seg: Segment features
            y_seg: Segment target
            segment_name: Name of the segment
            min_features: Minimum number of features to keep
            
        Returns:
            List of selected feature names
        """
        print(f"  Selecting features for {segment_name}...")
        
        # Skip feature selection for very small segments
        if len(X_seg) < 50:
            print(f"    Skipping feature selection - insufficient data ({len(X_seg)} samples)")
            return X_seg.columns.tolist()
        
        # Step 1: Calculate feature importance using Random Forest
        rf_selector = RandomForestRegressor(
            n_estimators=400, 
            max_depth=17, 
            random_state=42, 
            n_jobs=-1
        )
        rf_selector.fit(X_seg, y_seg)
        
        # Get feature importance scores
        feature_importance = pd.DataFrame({
            'feature': X_seg.columns,
            'importance': rf_selector.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Step 2: Remove bottom 20% features based on importance
        n_features = len(X_seg.columns)
        n_keep_importance = max(min_features, int(n_features * 0.8))
        
        important_features = feature_importance.head(n_keep_importance)['feature'].tolist()
        X_seg_filtered = X_seg[important_features]
        
        print(f"    After importance filtering: {len(important_features)} features")
        
        # Step 3: Use Recursive Feature Elimination (only if we have enough features)
        if len(important_features) > min_features * 1.5:  # Only if we can meaningfully reduce
            n_final_features = max(min_features, int(len(important_features) * 0.7))
            
            rfe_selector = RFE(
                estimator=RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1),
                n_features_to_select=n_final_features,
                step=1
            )
            
            rfe_selector.fit(X_seg_filtered, y_seg)
            selected_features = X_seg_filtered.columns[rfe_selector.support_].tolist()
            
            print(f"    After RFE: {len(selected_features)} features selected")
        else:
            selected_features = important_features
            print(f"    Skipping RFE: keeping {len(selected_features)} important features")
        
        # Step 4: Validate feature set on holdout (only if we have enough data)
        if len(X_seg) > 100:
            X_train_val, X_holdout, y_train_val, y_holdout = train_test_split(
                X_seg, y_seg, test_size=0.2, random_state=42
            )
            
            # Compare full important features vs selected features
            rf_full = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
            rf_selected = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
            
            # Ensure we use available features only
            available_important = [f for f in important_features if f in X_train_val.columns]
            available_selected = [f for f in selected_features if f in X_train_val.columns]
            
            rf_full.fit(X_train_val[available_important], y_train_val)
            rf_selected.fit(X_train_val[available_selected], y_train_val)
            
            pred_full = rf_full.predict(X_holdout[available_important])
            pred_selected = rf_selected.predict(X_holdout[available_selected])
            
            # Convert predictions to original scale for RMSE calculation
            y_holdout_orig = self._inverse_boxcox_transformation(y_holdout.values)
            pred_full_orig = self._inverse_boxcox_transformation(pred_full)
            pred_selected_orig = self._inverse_boxcox_transformation(pred_selected)
            
            rmse_full = np.sqrt(np.mean((y_holdout_orig - pred_full_orig) ** 2))
            rmse_selected = np.sqrt(np.mean((y_holdout_orig - pred_selected_orig) ** 2))
            
            print(f"    Validation RMSE - Important: {rmse_full:.2f}, Selected: {rmse_selected:.2f}")
            
            # Use selected features only if they perform similarly or better
            if rmse_selected <= rmse_full * 1.05:  # Allow 5% tolerance
                return selected_features
            else:
                print(f"    Using important features (selected features performed worse)")
                return important_features
        
        return selected_features
    
    def _create_validation_split(self, X: pd.DataFrame, y: pd.Series, 
                               segment_info: pd.DataFrame,
                               stratify_col: Optional[pd.Series], 
                               val_size: float) -> Tuple:
        """Create stratified train-validation split."""
        print(f"Creating {int((1-val_size)*100)}/{int(val_size*100)} train/validation split...")
        
        X_tr, X_val, y_tr, y_val, seg_tr, seg_val = train_test_split(
            X, y, segment_info, test_size=val_size, random_state=42,
            stratify=stratify_col
        )
        
        print(f"Training set: {X_tr.shape}")
        print(f"Validation set: {X_val.shape}")
        
        return X_tr, X_val, y_tr, y_val, seg_tr, seg_val
    
    def _train_segment_models(self, segment_type: str, segment_col: str,
                            X_tr: pd.DataFrame, X_val: pd.DataFrame,
                            y_tr: pd.Series, y_val: pd.Series,
                            seg_tr: pd.DataFrame, seg_val: pd.DataFrame):
        """Train Random Forest models for each segment with feature selection."""
        
        # Get unique segments
        unique_segments = seg_tr[segment_col].unique()
        print(f"Training models for {len(unique_segments)} {segment_type} segments")
        
        segment_models = {}
        segment_scores = {}
        segment_selected_features = {}  # NEW: Track selected features
        
        for segment in unique_segments:
            print(f"\nTraining model for {segment_type}: {segment}")
            
            # Filter data for this segment
            train_mask = seg_tr[segment_col] == segment
            val_mask = seg_val[segment_col] == segment
            
            if train_mask.sum() < 10:
                print(f"  Skipping {segment} - insufficient training data ({train_mask.sum()} samples)")
                continue
                
            X_seg_tr = X_tr[train_mask]
            y_seg_tr = y_tr[train_mask]
            X_seg_val = X_val[val_mask] if val_mask.sum() > 0 else None
            y_seg_val = y_val[val_mask] if val_mask.sum() > 0 else None
            
            print(f"  Training samples: {len(X_seg_tr)}")
            if X_seg_val is not None:
                print(f"  Validation samples: {len(X_seg_val)}")
            
            # NEW: Feature selection for this segment
            selected_features = self._select_features_for_segment(
                X_seg_tr, y_seg_tr, f"{segment_type}_{segment}"
            )
            segment_selected_features[segment] = selected_features
            
            # Use selected features for training
            X_seg_tr_selected = X_seg_tr[selected_features]
            X_seg_val_selected = X_seg_val[selected_features] if X_seg_val is not None else None
            
            # Train Random Forest for this segment
            if self.optimize_hyperparams and len(X_seg_tr_selected) > 50:
                best_params = self._optimize_rf_params_segment(X_seg_tr_selected, y_seg_tr, segment)
                model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
                print (f"  Best hyperparameters for {segment}: {best_params}")
            else:
                model = RandomForestRegressor(
                    n_estimators=400,
                    max_depth=17,
                    min_samples_split=40,
                    min_samples_leaf=25,
                    max_features=0.6,
                    bootstrap=True,
                    random_state=42,
                    n_jobs=-1
                )
            
            # Calculate sample weights and train
            y_seg_tr_original = self._inverse_boxcox_transformation(y_seg_tr.values)
            sample_weights = np.sqrt(y_seg_tr_original + 1)
            sample_weights = sample_weights / sample_weights.mean()
            
            model.fit(X_seg_tr_selected, y_seg_tr, sample_weight=sample_weights)
            
            # Evaluate model
            if X_seg_val_selected is not None and len(X_seg_val_selected) > 0:
                val_pred = model.predict(X_seg_val_selected)
                y_val_original = self._inverse_boxcox_transformation(y_seg_val.values)
                pred_original = self._inverse_boxcox_transformation(val_pred)
                
                val_weights = np.sqrt(y_val_original + 1)
                val_weights = val_weights / val_weights.mean()
                
                weighted_rmse = np.sqrt(((y_val_original - pred_original) ** 2 * val_weights).mean())
                segment_scores[segment] = weighted_rmse
                print(f"  Weighted RMSE: {weighted_rmse:.2f}")
            else:
                train_pred = model.predict(X_seg_tr_selected)
                train_pred_original = self._inverse_boxcox_transformation(train_pred)
                train_weighted_rmse = np.sqrt(((y_seg_tr_original - train_pred_original) ** 2 * sample_weights).mean())
                segment_scores[segment] = train_weighted_rmse
                print(f"  Training Weighted RMSE: {train_weighted_rmse:.2f}")
            
            segment_models[segment] = model
            
        # Store results
        self.segment_models[segment_type] = segment_models
        self.segment_scores[segment_type] = segment_scores
        self.selected_features[segment_type] = segment_selected_features  # NEW
        
        # Calculate weights
        total_samples = sum(len(seg_tr[seg_tr[segment_col] == seg]) for seg in segment_models.keys())
        segment_weights = {}
        for segment in segment_models.keys():
            segment_size = len(seg_tr[seg_tr[segment_col] == segment])
            segment_weights[segment] = segment_size / total_samples
        
        self.segment_weights[segment_type] = segment_weights
        
        print(f"\n{segment_type.upper()} segment summary:")
        print(f"  Models trained: {len(segment_models)}")
        print(f"  Average weighted RMSE: {np.mean(list(segment_scores.values())):.2f}")
        
    
    def _optimize_rf_params_segment(self, X_seg: pd.DataFrame, y_seg: pd.Series, 
                                   segment_name: str) -> dict:
        """Optimize Random Forest hyperparameters for a specific segment."""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 1500),
                'max_depth': trial.suggest_int('max_depth', 2, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 10, 100),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 75),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
            }
            
            # Use cross-validation for optimization
            cv = KFold(n_splits=min(5, len(X_seg) // 20), shuffle=True, random_state=42)
            fold_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(cv.split(X_seg)):
                X_fold_train, X_fold_val = X_seg.iloc[train_idx], X_seg.iloc[val_idx]
                y_fold_train, y_fold_val = y_seg.iloc[train_idx], y_seg.iloc[val_idx]
                
                # Calculate sample weights
                y_fold_train_orig = self._inverse_boxcox_transformation(y_fold_train.values)
                sample_weights = np.sqrt(y_fold_train_orig + 1)
                sample_weights = sample_weights / sample_weights.mean()
                
                model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
                model.fit(X_fold_train, y_fold_train, sample_weight=sample_weights)
                pred = model.predict(X_fold_val)
                
                # Calculate weighted RMSE
                y_val_orig = self._inverse_boxcox_transformation(y_fold_val.values)
                pred_orig = self._inverse_boxcox_transformation(pred)
                val_weights = np.sqrt(y_val_orig + 1)
                val_weights = val_weights / val_weights.mean()
                
                fold_rmse = np.sqrt(((y_val_orig - pred_orig) ** 2 * val_weights).mean())
                fold_scores.append(fold_rmse)
            
            return np.mean(fold_scores)
        
        sampler = TPESampler(seed=42)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        study.optimize(objective, n_trials=self.n_trials, #min(self.n_trials // 2, 25),
                        show_progress_bar=True)
        
        self.study_results[f'{segment_name}'] = study
        return study.best_params
    
    def debug_feature_mismatch(self, X_test: pd.DataFrame, segment_type: str, segment: str):
        """Debug feature mismatch issues."""
        if segment_type not in self.selected_features or segment not in self.selected_features[segment_type]:
            print(f"No selected features found for {segment_type}:{segment}")
            return
        
        model_features = self.selected_features[segment_type][segment]
        test_features = X_test.columns.tolist()
        
        missing_in_test = set(model_features) - set(test_features)
        extra_in_test = set(test_features) - set(model_features)
        
        print(f"Debug {segment_type}:{segment}")
        print(f"  Model expects {len(model_features)} features")
        print(f"  Test data has {len(test_features)} features")
        print(f"  Missing in test: {len(missing_in_test)} features")
        if missing_in_test:
            print(f"    First 5 missing: {list(missing_in_test)[:5]}")
        print(f"  Extra in test: {len(extra_in_test)} features")
        if extra_in_test:
            print(f"    First 5 extra: {list(extra_in_test)[:5]}")

    # ADD this method to validate feature consistency before prediction
    def validate_feature_consistency(self, X_test: pd.DataFrame):
        """Validate that test features are consistent with trained models."""
        print("Validating feature consistency...")
        
        test_features = set(X_test.columns)
        inconsistencies = []
        
        for segment_type, segment_features in self.selected_features.items():
            for segment, features in segment_features.items():
                model_features = set(features)
                missing_features = model_features - test_features
                
                if missing_features:
                    inconsistencies.append({
                        'segment_type': segment_type,
                        'segment': segment,
                        'missing_count': len(missing_features),
                        'missing_features': list(missing_features)[:5]  # Show first 5
                    })
        
        if inconsistencies:
            print(f"Found {len(inconsistencies)} segments with missing features:")
            for issue in inconsistencies[:5]:  # Show first 5 issues
                print(f"  {issue['segment_type']}:{issue['segment']} - {issue['missing_count']} missing features")
                print(f"    Examples: {issue['missing_features']}")
        else:
            print("All features are consistent!")
        
        return len(inconsistencies) == 0

    def predict_segmented(self, X_test: pd.DataFrame, 
                     segment_info: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using segmented models and combine results.
        
        Args:
            X_test: Test features
            segment_info: DataFrame with segment information for test data
            
        Returns:
            Combined predictions on original scale
        """
        print("Making segmented predictions...")
        
        n_samples = len(X_test)
        
        # Initialize prediction arrays for each segment type
        predictions_by_type = {}
        weights_by_type = {}
        
        for segment_type, segment_col in self.segment_definitions.items():
            if segment_type not in self.segment_models:
                continue
                
            print(f"Generating {segment_type} predictions...")
            
            predictions = np.zeros(n_samples)
            prediction_weights = np.zeros(n_samples)
            
            segment_models = self.segment_models[segment_type]
            segment_weights = self.segment_weights[segment_type]
            selected_features = self.selected_features.get(segment_type, {})
            
            for segment, model in segment_models.items():
                # Find samples belonging to this segment
                segment_mask = segment_info[segment_col] == segment
                
                if segment_mask.sum() > 0:
                    # Get selected features for this segment
                    if segment in selected_features:
                        features_for_segment = selected_features[segment]
                        # Ensure all features exist in test data
                        available_features = [f for f in features_for_segment if f in X_test.columns]
                        if len(available_features) == 0:
                            print(f"Warning: No selected features available for segment {segment}, using all features")
                            available_features = X_test.columns.tolist()
                    else:
                        # Fallback to all features if no selection was done
                        available_features = X_test.columns.tolist()
                    
                    # Make predictions for this segment using selected features
                    X_segment = X_test.loc[segment_mask, available_features]
                    segment_pred = model.predict(X_segment)
                    
                    # Convert to original scale
                    segment_pred_original = self._inverse_boxcox_transformation(segment_pred)
                    
                    # Store predictions and weights
                    predictions[segment_mask] = segment_pred_original
                    prediction_weights[segment_mask] = segment_weights.get(segment, 1.0)
            
            # Handle unseen segments by using average prediction from all models
            unknown_mask = prediction_weights == 0
            if unknown_mask.sum() > 0:
                print(f"  Found {unknown_mask.sum()} samples with unknown {segment_type}, using ensemble prediction")
                
                # Use all models to predict unknown segments
                ensemble_preds = []
                for segment, model in segment_models.items():
                    # Get selected features for this segment
                    if segment in selected_features:
                        features_for_segment = selected_features[segment]
                        available_features = [f for f in features_for_segment if f in X_test.columns]
                    else:
                        available_features = X_test.columns.tolist()
                    
                    if available_features:
                        try:
                            unknown_pred = model.predict(X_test.loc[unknown_mask, available_features])
                            unknown_pred_original = self._inverse_boxcox_transformation(unknown_pred)
                            ensemble_preds.append(unknown_pred_original)
                        except Exception as e:
                            print(f"Warning: Could not predict with model for segment {segment}: {e}")
                            continue
                
                if ensemble_preds:
                    predictions[unknown_mask] = np.mean(ensemble_preds, axis=0)
                    prediction_weights[unknown_mask] = 1.0
                else:
                    # Final fallback: use reasonable default
                    predictions[unknown_mask] = 2000  # Reasonable default for sales
                    prediction_weights[unknown_mask] = 1.0
                    print(f"Warning: Using fallback value (2000) for {unknown_mask.sum()} unknown segments")
            
            predictions_by_type[segment_type] = predictions
            weights_by_type[segment_type] = prediction_weights
        
        # Combine predictions from different segment types using weighted average
        print("Combining predictions from different segment types...")
        
        # Calculate segment type weights based on performance
        type_weights = {}
        for segment_type in predictions_by_type.keys():
            if segment_type in self.segment_scores and self.segment_scores[segment_type]:
                avg_score = np.mean(list(self.segment_scores[segment_type].values()))
                # Inverse weight (better performance = higher weight)
                type_weights[segment_type] = 1.0 / (avg_score + 1e-6)
            else:
                type_weights[segment_type] = 1.0
        
        # Normalize weights
        total_weight = sum(type_weights.values())
        if total_weight > 0:
            type_weights = {k: v / total_weight for k, v in type_weights.items()}
        else:
            # Equal weights if no scores available
            type_weights = {k: 1.0 / len(predictions_by_type) for k in predictions_by_type.keys()}
        
        print("Segment type weights:")
        for segment_type, weight in type_weights.items():
            print(f"  {segment_type}: {weight:.3f}")
        
        # Weighted combination
        final_predictions = np.zeros(n_samples)
        for segment_type, predictions in predictions_by_type.items():
            final_predictions += predictions * type_weights[segment_type]
        
        # Ensure no negative predictions
        final_predictions = np.maximum(final_predictions, 0)
        
        return final_predictions
    
    def cross_validate_segments(self, X_train: pd.DataFrame, y_train: pd.Series,
                               segment_info: pd.DataFrame,
                               cv_folds: int = 5) -> Dict[str, Dict]:
        """Perform cross-validation for segmented models."""
        print("\n=== Cross-Validation for Segmented Models ===")
        
        y_train_transformed = self._apply_boxcox_transformation(y_train)
        
        cv_scores = {}
        
        for segment_type in self.segment_definitions.keys():
            if segment_type in self.segment_models:
                print(f"\nCross-validating {segment_type} models...")
                
                cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                fold_scores = []
                
                for fold, (train_idx, val_idx) in enumerate(cv.split(X_train)):
                    # Split data
                    X_fold_tr = X_train.iloc[train_idx]
                    X_fold_val = X_train.iloc[val_idx]
                    y_fold_tr = y_train_transformed.iloc[train_idx]
                    y_fold_val = y_train_transformed.iloc[val_idx]
                    seg_fold_tr = segment_info.iloc[train_idx]
                    seg_fold_val = segment_info.iloc[val_idx]
                    
                    # Train temporary models for this fold
                    temp_trainer = SegmentedModelTrainer(optimize_hyperparams=False)
                    temp_trainer.global_transformer = self.global_transformer
                    temp_trainer._train_segment_models(
                        segment_type, self.segment_definitions[segment_type],
                        X_fold_tr, X_fold_val, y_fold_tr, y_fold_val,
                        seg_fold_tr, seg_fold_val
                    )
                    
                    # Make predictions
                    if segment_type in temp_trainer.segment_models:
                        fold_pred = temp_trainer._predict_single_segment_type(
                            X_fold_val, seg_fold_val, segment_type
                        )
                        
                        # Calculate weighted RMSE
                        y_val_orig = self._inverse_boxcox_transformation(y_fold_val.values)
                        val_weights = np.sqrt(y_val_orig + 1)
                        val_weights = val_weights / val_weights.mean()
                        
                        fold_rmse = np.sqrt(((y_val_orig - fold_pred) ** 2 * val_weights).mean())
                        fold_scores.append(fold_rmse)
                
                if fold_scores:
                    cv_scores[segment_type] = {
                        'mean': np.mean(fold_scores),
                        'std': np.std(fold_scores),
                        'scores': fold_scores
                    }
                    print(f"{segment_type} CV Weighted RMSE: {np.mean(fold_scores):.2f} (+/- {np.std(fold_scores):.2f})")
        
        return cv_scores
    
    def predict_individual_segments(self, X_test: pd.DataFrame, 
                                  segment_info: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Generate predictions for each segment type individually.
        
        Args:
            X_test: Test features
            segment_info: Test segment information
            
        Returns:
            Dictionary mapping segment type to predictions
        """
        print("Generating individual segment predictions...")
        
        individual_predictions = {}
        
        for segment_type, segment_col in self.segment_definitions.items():
            if segment_type not in self.segment_models:
                continue
                
            predictions = self._predict_single_segment_type(X_test, segment_info, segment_type)
            individual_predictions[segment_type] = predictions
            
        self.individual_predictions = individual_predictions
        return individual_predictions
    
    def _predict_single_segment_type(self, X_test: pd.DataFrame, 
                               segment_info: pd.DataFrame,
                               segment_type: str) -> np.ndarray:
        """Make predictions using a single segment type with selected features."""
        segment_col = self.segment_definitions[segment_type]
        segment_models = self.segment_models[segment_type]
        selected_features = self.selected_features.get(segment_type, {})
        
        predictions = np.zeros(len(X_test))
        
        for segment, model in segment_models.items():
            segment_mask = segment_info[segment_col] == segment
            if segment_mask.sum() > 0:
                # Get selected features for this segment
                if segment in selected_features:
                    features_for_segment = selected_features[segment]
                    # Ensure all features exist in test data
                    available_features = [f for f in features_for_segment if f in X_test.columns]
                    if len(available_features) == 0:
                        print(f"Warning: No selected features available for segment {segment}, using all features")
                        available_features = X_test.columns.tolist()
                else:
                    # Fallback to all features if no selection was done
                    available_features = X_test.columns.tolist()
                
                X_segment = X_test.loc[segment_mask, available_features]
                segment_pred = model.predict(X_segment)
                segment_pred_original = self._inverse_boxcox_transformation(segment_pred)
                predictions[segment_mask] = segment_pred_original
        
        # Handle unknown segments
        unknown_mask = predictions == 0
        if unknown_mask.sum() > 0:
            print(f"Handling {unknown_mask.sum()} unknown segments for {segment_type}")
            ensemble_preds = []
            
            for segment, model in segment_models.items():
                # Get features for this segment
                if segment in selected_features:
                    features_for_segment = selected_features[segment]
                    available_features = [f for f in features_for_segment if f in X_test.columns]
                else:
                    available_features = X_test.columns.tolist()
                
                if available_features:
                    try:
                        unknown_pred = model.predict(X_test.loc[unknown_mask, available_features])
                        unknown_pred_original = self._inverse_boxcox_transformation(unknown_pred)
                        ensemble_preds.append(unknown_pred_original)
                    except Exception as e:
                        print(f"Warning: Could not predict with model for segment {segment}: {e}")
                        continue
            
            if ensemble_preds:
                predictions[unknown_mask] = np.mean(ensemble_preds, axis=0)
            else:
                # Final fallback: use overall mean
                overall_mean = 2000  # Reasonable default for sales
                predictions[unknown_mask] = overall_mean
                print(f"Warning: Using fallback mean ({overall_mean}) for unknown segments")
        
        return predictions
    
    def _get_available_features(self, model_features: List[str], available_features: List[str]) -> List[str]:
        """
        Get intersection of model features and available features.
        
        Args:
            model_features: Features the model was trained on
            available_features: Features available in the dataset
            
        Returns:
            List of features that are both in model and available
        """
        available = [f for f in model_features if f in available_features]
        
        if len(available) == 0:
            print("Warning: No model features available, using all available features")
            return available_features
        
        if len(available) < len(model_features):
            missing = set(model_features) - set(available)
            print(f"Warning: {len(missing)} model features not available: {list(missing)[:5]}...")
        
        return available
    
    def predict_train_data(self, X_train: pd.DataFrame, 
                      segment_info_train: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for training data using segmented models.
        
        Args:
            X_train: Training features
            segment_info_train: Training segment information
            
        Returns:
            DataFrame with predictions from all segment types
        """
        print("Generating training data predictions...")
        
        n_samples = len(X_train)
        train_predictions = pd.DataFrame(index=X_train.index)
        
        # Generate predictions for each segment type
        for segment_type, segment_col in self.segment_definitions.items():
            if segment_type not in self.segment_models:
                continue
                
            print(f"Generating {segment_type} train predictions...")
            
            predictions = np.zeros(n_samples)
            segment_models = self.segment_models[segment_type]
            selected_features = self.selected_features.get(segment_type, {})
            
            for segment, model in segment_models.items():
                segment_mask = segment_info_train[segment_col] == segment
                
                if segment_mask.sum() > 0:
                    # Get selected features for this segment
                    if segment in selected_features:
                        features_for_segment = selected_features[segment]
                        # Ensure all features exist in training data
                        available_features = [f for f in features_for_segment if f in X_train.columns]
                        if len(available_features) == 0:
                            available_features = X_train.columns.tolist()
                    else:
                        available_features = X_train.columns.tolist()
                    
                    X_segment = X_train.loc[segment_mask, available_features]
                    segment_pred = model.predict(X_segment)
                    
                    # Convert to original scale
                    segment_pred_original = self._inverse_boxcox_transformation(segment_pred)
                    predictions[segment_mask] = segment_pred_original
            
            # Handle samples with no predictions (unknown segments)
            unknown_mask = predictions == 0
            if unknown_mask.sum() > 0:
                print(f"Handling {unknown_mask.sum()} unknown training segments for {segment_type}")
                # Use ensemble of all models for unknown segments
                ensemble_preds = []
                for segment, model in segment_models.items():
                    if segment in selected_features:
                        features_for_segment = selected_features[segment]
                        available_features = [f for f in features_for_segment if f in X_train.columns]
                    else:
                        available_features = X_train.columns.tolist()
                    
                    if available_features:
                        try:
                            unknown_pred = model.predict(X_train.loc[unknown_mask, available_features])
                            unknown_pred_original = self._inverse_boxcox_transformation(unknown_pred)
                            ensemble_preds.append(unknown_pred_original)
                        except Exception as e:
                            print(f"Warning: Could not predict with model for segment {segment}: {e}")
                            continue
                
                if ensemble_preds:
                    predictions[unknown_mask] = np.mean(ensemble_preds, axis=0)
            
            train_predictions[f'pred_{segment_type}'] = predictions
        
        # Generate combined prediction using same logic as test prediction
        if len(train_predictions.columns) > 1:
            # Calculate weights based on segment performance
            type_weights = {}
            for segment_type in self.segment_definitions.keys():
                if segment_type in self.segment_scores and self.segment_scores[segment_type]:
                    avg_score = np.mean(list(self.segment_scores[segment_type].values()))
                    type_weights[segment_type] = 1.0 / (avg_score + 1e-6)
            
            if type_weights:
                # Normalize weights
                total_weight = sum(type_weights.values())
                type_weights = {k: v / total_weight for k, v in type_weights.items()}
                
                # Create combined prediction
                combined_pred = np.zeros(n_samples)
                for segment_type, weight in type_weights.items():
                    if f'pred_{segment_type}' in train_predictions.columns:
                        combined_pred += train_predictions[f'pred_{segment_type}'] * weight
                
                train_predictions['pred_combined'] = combined_pred
        
        # Store for later use
        self.train_predictions = train_predictions
        
        print(f"Train predictions generated for {len(train_predictions.columns)} models")
        return train_predictions
    
    def get_feature_importance(self, feature_names: List[str], 
                             segment_type: str = 'outlet_type',
                             top_n: int = 20) -> pd.DataFrame:
        """Get feature importance from segmented models."""
        if segment_type not in self.segment_models:
            return pd.DataFrame()
        
        # Aggregate feature importance across segments
        segment_models = self.segment_models[segment_type]
        segment_weights = self.segment_weights[segment_type]
        
        if not segment_models:
            return pd.DataFrame()
        
        # Initialize importance array
        n_features = len(feature_names)
        aggregated_importance = np.zeros(n_features)
        
        # Weight and aggregate importance scores
        for segment, model in segment_models.items():
            weight = segment_weights.get(segment, 1.0)
            importance = model.feature_importances_
            aggregated_importance += importance * weight
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': aggregated_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def save_models(self, output_dir: str):
        """Save segmented models and transformations."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save Box-Cox transformer
        if self.global_transformer:
            joblib.dump(self.global_transformer, f"{output_dir}/boxcox_transformer.pkl")
        
        # Save segmented models
        for segment_type, models in self.segment_models.items():
            segment_dir = f"{output_dir}/{segment_type}_models"
            os.makedirs(segment_dir, exist_ok=True)
            
            for segment, model in models.items():
                # Clean segment name for filename
                clean_segment = str(segment).replace(' ', '_').replace('/', '_')
                joblib.dump(model, f"{segment_dir}/{clean_segment}_model.pkl")
        
        # Save weights and scores
        joblib.dump(self.segment_weights, f"{output_dir}/segment_weights.pkl")
        joblib.dump(self.segment_scores, f"{output_dir}/segment_scores.pkl")
        
        # Save best hyperparameters
        if self.best_params:
            joblib.dump(self.best_params, f"{output_dir}/best_hyperparameters.pkl")
        
        print(f"Segmented models saved to {output_dir}")
    
    def plot_segment_performance(self, segment_type: str = 'outlet_type'):
        """Plot performance across segments."""
        if segment_type not in self.segment_scores:
            print(f"No scores available for {segment_type}")
            return
        
        scores = self.segment_scores[segment_type]
        weights = self.segment_weights[segment_type]
        
        # Create DataFrame for plotting
        plot_data = pd.DataFrame({
            'Segment': list(scores.keys()),
            'Weighted_RMSE': list(scores.values()),
            'Data_Weight': [weights.get(seg, 0) for seg in scores.keys()]
        })
        
        # Sort by score
        plot_data = plot_data.sort_values('Weighted_RMSE')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: RMSE by segment
        bars1 = ax1.bar(range(len(plot_data)), plot_data['Weighted_RMSE'])
        ax1.set_xlabel('Segment')
        ax1.set_ylabel('Weighted RMSE')
        ax1.set_title(f'{segment_type.title()} Model Performance')
        ax1.set_xticks(range(len(plot_data)))
        ax1.set_xticklabels(plot_data['Segment'], rotation=45, ha='right')
        
        # Add value labels
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{height:.1f}', ha='center', va='bottom')
        
        # Plot 2: Data weight by segment
        bars2 = ax2.bar(range(len(plot_data)), plot_data['Data_Weight'])
        ax2.set_xlabel('Segment')
        ax2.set_ylabel('Data Weight')
        ax2.set_title(f'{segment_type.title()} Data Distribution')
        ax2.set_xticks(range(len(plot_data)))
        ax2.set_xticklabels(plot_data['Segment'], rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()


# Wrapper function to maintain compatibility with existing pipeline
# UPDATE train_segmented_models function to return trainer object
# MODIFY the train_segmented_models wrapper function to include validation
def train_segmented_models(X_train_path: str, 
                          y_train: pd.Series,
                          X_test_path: str,
                          segment_info_train: pd.DataFrame,
                          segment_info_test: pd.DataFrame,
                          output_dir: str,
                          optimize_hyperparams: bool = True,
                          n_trials: int = 50) -> Tuple[Dict[str, Dict], Dict[str, np.ndarray], SegmentedModelTrainer]:
    """
    Updated function to return trainer object and individual predictions.
    
    Returns:
        Tuple of (segment_scores, all_predictions_dict, trainer_object)
    """
    print("=== Segmented BigMart Model Training Pipeline ===")
    
    # Load encoded data
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    
    print(f"Loaded training data: {X_train.shape}")
    print(f"Loaded test data: {X_test.shape}")
    print(f"Training features: {X_train.columns.tolist()[:10]}...")  # Show first 10 features
    print(f"Test features: {X_test.columns.tolist()[:10]}...")  # Show first 10 features
    
    # Check if train and test have same features
    train_features = set(X_train.columns)
    test_features = set(X_test.columns)
    
    if train_features != test_features:
        missing_in_test = train_features - test_features
        extra_in_test = test_features - train_features
        
        print(f"WARNING: Feature mismatch detected!")
        print(f"  Missing in test: {len(missing_in_test)} features")
        if missing_in_test:
            print(f"    Examples: {list(missing_in_test)[:5]}")
        print(f"  Extra in test: {len(extra_in_test)} features") 
        if extra_in_test:
            print(f"    Examples: {list(extra_in_test)[:5]}")
        
        # Use intersection of features
        common_features = list(train_features & test_features)
        print(f"Using {len(common_features)} common features")
        X_train = X_train[common_features]
        X_test = X_test[common_features]
    
    # Initialize trainer
    trainer = SegmentedModelTrainer(optimize_hyperparams=optimize_hyperparams, n_trials=n_trials)
    
    # Train segmented models
    scores = trainer.train_segmented_models(X_train, y_train, segment_info_train)
    
    # Validate feature consistency before prediction
    print("\nValidating features before prediction...")
    trainer.validate_feature_consistency(X_test)
    
    # Generate all types of predictions
    try:
        individual_predictions = trainer.predict_individual_segments(X_test, segment_info_test)
        combined_predictions = trainer.predict_segmented(X_test, segment_info_test)
    except Exception as e:
        print(f"Error during prediction: {e}")
        print("Attempting to debug feature issues...")
        
        # Debug the first segment to understand the issue
        for segment_type in trainer.segment_definitions.keys():
            if segment_type in trainer.segment_models:
                first_segment = list(trainer.segment_models[segment_type].keys())[0]
                trainer.debug_feature_mismatch(X_test, segment_type, first_segment)
                break
        
        raise e
    
    # Create predictions dictionary
    all_predictions = {
        'outlet_type': individual_predictions.get('outlet_type', combined_predictions),
        'outlet_identifier': individual_predictions.get('outlet_identifier', combined_predictions),
        'mrp_bins': individual_predictions.get('mrp_bins', combined_predictions),
        'combined': combined_predictions
    }
    
    # Determine best single model
    if scores:
        best_segment_type = min(scores.keys(), key=lambda k: np.mean(list(scores[k].values())))
        all_predictions['best_single'] = individual_predictions.get(best_segment_type, combined_predictions)
        print(f"\nBest single segment type: {best_segment_type}")
    else:
        all_predictions['best_single'] = combined_predictions
        print("\nNo scores available, using combined predictions as best single")
    
    # Generate train predictions
    try:
        train_predictions = trainer.predict_train_data(X_train, segment_info_train)
    except Exception as e:
        print(f"Warning: Could not generate train predictions: {e}")
        # Create dummy train predictions
        train_predictions = pd.DataFrame({
            'pred_combined': np.full(len(X_train), 2000)
        }, index=X_train.index)
        trainer.train_predictions = train_predictions
    
    # Save models and results
    trainer.save_models(output_dir)
    
    # Display results
    print("\n=== Segmented Model Performance Summary ===")
    for segment_type, segment_scores in scores.items():
        if segment_scores:  # Only show if we have scores
            avg_score = np.mean(list(segment_scores.values()))
            print(f"\n{segment_type.upper()} segments:")
            print(f"  Average Weighted RMSE: {avg_score:.2f}")
            print(f"  Number of models: {len(segment_scores)}")
    
    return scores, all_predictions, trainer