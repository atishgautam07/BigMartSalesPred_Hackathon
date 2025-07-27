import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.preprocessing import PowerTransformer
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


class StratifiedModelTrainer:
    """
    Trains separate Random Forest models for different strata (Outlet_Type, Outlet_Identifier, Item_MRP_Bins).
    Uses weighted RMSE loss and Box-Cox transformation for target variable.
    """
    
    def __init__(self, optimize_hyperparams: bool = True, n_trials: int = 50):
        self.models = {}
        self.transformers = {}
        self.scores = {}
        self.feature_importance = {}
        self.strata_info = {}
        self.optimize_hyperparams = optimize_hyperparams
        self.n_trials = n_trials
        self.best_params = {}
        self.study_results = {}
        self.weights = {}
        
    def prepare_target_transformation(self, y_train: pd.Series, 
                                    strata_col: pd.Series) -> pd.Series:
        """
        Apply Box-Cox transformation to target variable for each stratum.
        
        Args:
            y_train: Target variable (original scale)
            strata_col: Stratification column
            
        Returns:
            Transformed target variable
        """
        print("Applying Box-Cox transformation to target variable...")
        
        y_transformed = y_train.copy()
        self.transformers = {}
        
        for stratum in strata_col.unique():
            mask = strata_col == stratum
            y_stratum = y_train[mask]
            
            # Ensure positive values for Box-Cox
            y_positive = y_stratum + 1 if y_stratum.min() <= 0 else y_stratum
            
            # Apply Box-Cox transformation
            transformer = PowerTransformer(method='box-cox', standardize=False)
            y_stratum_transformed = transformer.fit_transform(y_positive.values.reshape(-1, 1)).flatten()
            
            self.transformers[stratum] = transformer
            y_transformed[mask] = y_stratum_transformed
            
            print(f"Stratum {stratum}: lambda={transformer.lambdas_[0]:.3f}, "
                  f"skewness: {stats.skew(y_stratum):.3f} -> {stats.skew(y_stratum_transformed):.3f}")
        
        return y_transformed
    
    def inverse_transform_predictions(self, predictions: np.ndarray, 
                                    strata_col: pd.Series) -> np.ndarray:
        """
        Apply inverse Box-Cox transformation to predictions.
        
        Args:
            predictions: Transformed predictions
            strata_col: Stratification column
            
        Returns:
            Predictions on original scale
        """
        predictions_original = np.zeros_like(predictions)
        
        for stratum in strata_col.unique():
            mask = strata_col == stratum
            if np.any(mask):
                transformer = self.transformers[stratum]
                pred_stratum = predictions[mask].reshape(-1, 1)
                pred_original = transformer.inverse_transform(pred_stratum).flatten()
                
                # Subtract 1 if we added it during transformation
                if hasattr(self, '_added_one') and self._added_one.get(stratum, False):
                    pred_original = pred_original - 1
                    
                predictions_original[mask] = pred_original
        
        return predictions_original
    
    def calculate_weighted_rmse(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               weights: np.ndarray = None) -> float:
        """
        Calculate weighted RMSE.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            weights: Sample weights
            
        Returns:
            Weighted RMSE
        """
        if weights is None:
            weights = np.ones(len(y_true))
        weights = 1.0 / (y_true + 100)
        
        squared_errors = (y_true - y_pred) ** 2
        weighted_mse = np.average(squared_errors, weights=weights)
        return np.sqrt(weighted_mse)
    
    def compute_sample_weights(self, y: pd.Series, strata_col: pd.Series, 
                              weight_method: str = 'inverse_variance') -> np.ndarray:
        """
        Compute sample weights based on target distribution.
        
        Args:
            y: Target variable
            strata_col: Stratification column
            weight_method: Method for computing weights
            
        Returns:
            Sample weights
        """
        weights = np.ones(len(y))
        
        if weight_method == 'inverse_variance':
            for stratum in strata_col.unique():
                mask = strata_col == stratum
                stratum_var = y[mask].var()
                if stratum_var > 0:
                    weights[mask] = 1.0 / stratum_var
                    
        elif weight_method == 'frequency':
            stratum_counts = strata_col.value_counts()
            total_samples = len(y)
            for stratum in strata_col.unique():
                mask = strata_col == stratum
                frequency_weight = total_samples / (len(stratum_counts) * stratum_counts[stratum])
                weights[mask] = frequency_weight
                
        elif weight_method == 'sales_based':
            # Higher weights for higher sales values
            weights = y / y.mean()
            
        # Normalize weights
        weights = weights / weights.mean()
        self.weights = weights
        
        return weights
    
    def train_stratified_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                               outlet_type: pd.Series, outlet_id: pd.Series,
                               mrp_bins: pd.Series, val_size: float = 0.2) -> Dict[str, float]:
        """
        Train separate models for different stratification schemes.
        
        Args:
            X_train: Training features
            y_train: Training target (original scale)
            outlet_type: Outlet type column
            outlet_id: Outlet identifier column
            mrp_bins: MRP bins column
            val_size: Validation set size
            
        Returns:
            Dictionary of model scores
        """
        print("=== Stratified Model Training Pipeline ===")
        
        # Define stratification schemes
        stratification_schemes = {
            'outlet_type': outlet_type,
            'outlet_id': outlet_id,
            'mrp_bins': mrp_bins
        }
        
        overall_scores = {}
        
        for scheme_name, strata_col in stratification_schemes.items():
            print(f"\n--- Training models for {scheme_name} stratification ---")
            
            # Apply Box-Cox transformation per stratum
            y_transformed = self.prepare_target_transformation(y_train, strata_col)
            
            # Compute sample weights
            sample_weights = self.compute_sample_weights(y_train, strata_col, 'inverse_variance')
            
            # Train models for each stratum
            scheme_models = {}
            scheme_scores = {}
            scheme_importance = {}
            
            for stratum in strata_col.unique():
                print(f"\nTraining model for {scheme_name} = {stratum}")
                
                # Get data for this stratum
                mask = strata_col == stratum
                X_stratum = X_train[mask]
                y_stratum = y_transformed[mask]
                weights_stratum = sample_weights[mask]
                
                if len(X_stratum) < 10:  # Skip if too few samples
                    print(f"Skipping {stratum} - insufficient samples ({len(X_stratum)})")
                    continue
                
                # Train-validation split
                X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
                    X_stratum, y_stratum, weights_stratum,
                    test_size=val_size, random_state=42
                )
                
                # Train Random Forest model
                model, score, importance = self._train_single_rf_model(
                    X_tr, X_val, y_tr, y_val, w_tr, w_val, 
                    f"{scheme_name}_{stratum}"
                )
                
                scheme_models[stratum] = model
                scheme_scores[stratum] = score
                scheme_importance[stratum] = importance
                
                # Store stratum info
                self.strata_info[f"{scheme_name}_{stratum}"] = {
                    'n_samples': len(X_stratum),
                    'target_mean': y_train[mask].mean(),
                    'target_std': y_train[mask].std()
                }
            
            # Store models and scores
            self.models[scheme_name] = scheme_models
            self.scores[scheme_name] = scheme_scores
            self.feature_importance[scheme_name] = scheme_importance
            
            # Calculate overall scheme score
            scheme_score = np.mean(list(scheme_scores.values()))
            overall_scores[scheme_name] = scheme_score
            
            print(f"\n{scheme_name} stratification overall score: {scheme_score:.4f}")
        
        return overall_scores
    
    def _train_single_rf_model(self, X_tr: pd.DataFrame, X_val: pd.DataFrame,
                              y_tr: pd.Series, y_val: pd.Series,
                              w_tr: np.ndarray, w_val: np.ndarray,
                              model_name: str) -> Tuple[RandomForestRegressor, float, pd.DataFrame]:
        """
        Train a single Random Forest model with hyperparameter optimization.
        
        Args:
            X_tr, X_val: Training and validation features
            y_tr, y_val: Training and validation targets (transformed)
            w_tr, w_val: Training and validation weights
            model_name: Name for this model
            
        Returns:
            Tuple of (model, weighted_rmse_score, feature_importance)
        """
        print(f"  Training Random Forest for {model_name}...")
        print(f"  Training samples: {len(X_tr)}, Validation samples: {len(X_val)}")
        
        if self.optimize_hyperparams and len(X_tr) > 100:  # Only optimize if enough data
            print("  Running hyperparameter optimization...")
            best_params = self._optimize_rf_params_weighted(X_tr, y_tr, w_tr, model_name)
            self.best_params[model_name] = best_params
            
            model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
        else:
            # Use default parameters for small datasets
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                bootstrap=True,
                random_state=42,
                n_jobs=-1
            )
        
        # Train model with sample weights
        model.fit(X_tr, y_tr, sample_weight=w_tr)
        
        # Make predictions
        val_pred = model.predict(X_val)
        
        # Calculate weighted RMSE on transformed scale
        weighted_rmse = self.calculate_weighted_rmse(y_val, val_pred, w_val)
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feature': X_tr.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"  Model trained - Weighted RMSE: {weighted_rmse:.4f}")
        
        return model, weighted_rmse, importance_df
    
    def _optimize_rf_params_weighted(self, X_tr: pd.DataFrame, y_tr: pd.Series,
                                   w_tr: np.ndarray, model_name: str) -> dict:
        """
        Optimize Random Forest hyperparameters using weighted cross-validation.
        """
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 400),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 5, 30),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 15),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5, 0.7]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
            }
            
            # Use cross-validation with weights
            cv = KFold(n_splits=min(5, len(X_tr) // 20), shuffle=True, random_state=42)
            fold_scores = []
            
            for train_idx, val_idx in cv.split(X_tr):
                X_fold_train, X_fold_val = X_tr.iloc[train_idx], X_tr.iloc[val_idx]
                y_fold_train, y_fold_val = y_tr.iloc[train_idx], y_tr.iloc[val_idx]
                w_fold_train, w_fold_val = w_tr[train_idx], w_tr[val_idx]
                
                model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
                model.fit(X_fold_train, y_fold_train, sample_weight=w_fold_train)
                pred = model.predict(X_fold_val)
                
                # Calculate weighted RMSE
                fold_rmse = self.calculate_weighted_rmse(y_fold_val, pred, w_fold_val)
                fold_scores.append(fold_rmse)
            
            return np.mean(fold_scores)
        
        sampler = TPESampler(seed=42)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        study.optimize(objective, n_trials=min(self.n_trials, 30), show_progress_bar=True)
        
        self.study_results[model_name] = study
        return study.best_params
    
    def predict_stratified(self, X_test: pd.DataFrame, 
                          outlet_type_test: pd.Series,
                          outlet_id_test: pd.Series,
                          mrp_bins_test: pd.Series,
                          strategy: str = 'ensemble') -> np.ndarray:
        """
        Make predictions using stratified models.
        
        Args:
            X_test: Test features
            outlet_type_test: Test outlet types
            outlet_id_test: Test outlet identifiers
            mrp_bins_test: Test MRP bins
            strategy: Prediction strategy ('ensemble', 'best_scheme', or specific scheme)
            
        Returns:
            Final predictions on original scale
        """
        print(f"Making stratified predictions using strategy: {strategy}")
        
        # Define test stratification
        test_strata = {
            'outlet_type': outlet_type_test,
            'outlet_id': outlet_id_test,
            'mrp_bins': mrp_bins_test
        }
        
        if strategy == 'ensemble':
            # Ensemble predictions from all schemes
            all_predictions = []
            scheme_weights = []
            
            for scheme_name in self.models.keys():
                scheme_pred = self._predict_single_scheme(
                    X_test, test_strata[scheme_name], scheme_name
                )
                if scheme_pred is not None:
                    all_predictions.append(scheme_pred)
                    # Weight by inverse of scheme score
                    weight = 1.0 / (self.scores[scheme_name] if scheme_name in self.scores 
                                   else np.mean(list(self.scores[scheme_name].values())))
                    scheme_weights.append(weight)
            
            if all_predictions:
                # Weighted average of predictions
                scheme_weights = np.array(scheme_weights) / np.sum(scheme_weights)
                final_predictions = np.average(all_predictions, axis=0, weights=scheme_weights)
            else:
                raise ValueError("No valid predictions generated")
                
        elif strategy == 'best_scheme':
            # Use best performing scheme
            best_scheme = min(self.scores.keys(), 
                            key=lambda k: np.mean(list(self.scores[k].values())))
            final_predictions = self._predict_single_scheme(
                X_test, test_strata[best_scheme], best_scheme
            )
            
        else:
            # Use specific scheme
            if strategy not in self.models:
                raise ValueError(f"Strategy {strategy} not available")
            final_predictions = self._predict_single_scheme(
                X_test, test_strata[strategy], strategy
            )
        
        return final_predictions
    
    def _predict_single_scheme(self, X_test: pd.DataFrame, 
                              strata_col: pd.Series, 
                              scheme_name: str) -> np.ndarray:
        """
        Make predictions using models from a single stratification scheme.
        """
        predictions = np.zeros(len(X_test))
        scheme_models = self.models[scheme_name]
        
        for stratum in strata_col.unique():
            mask = strata_col == stratum
            
            if stratum in scheme_models:
                # Use stratum-specific model
                model = scheme_models[stratum]
                X_stratum = X_test[mask]
                
                if len(X_stratum) > 0:
                    pred_transformed = model.predict(X_stratum)
                    
                    # Inverse transform predictions
                    pred_original = self.inverse_transform_predictions(
                        pred_transformed, pd.Series([stratum] * len(pred_transformed))
                    )
                    predictions[mask] = pred_original
            else:
                # Fallback to most similar stratum or overall model
                print(f"Warning: No model for {stratum}, using fallback")
                # Use model from most similar stratum (by sample size)
                similar_stratum = max(scheme_models.keys(), 
                                    key=lambda s: self.strata_info.get(f"{scheme_name}_{s}", {}).get('n_samples', 0))
                model = scheme_models[similar_stratum]
                
                X_stratum = X_test[mask]
                if len(X_stratum) > 0:
                    pred_transformed = model.predict(X_stratum)
                    pred_original = self.inverse_transform_predictions(
                        pred_transformed, pd.Series([similar_stratum] * len(pred_transformed))
                    )
                    predictions[mask] = pred_original
        
        return predictions
    
    def cross_validate_stratified(self, X_train: pd.DataFrame, y_train: pd.Series,
                                 outlet_type: pd.Series, outlet_id: pd.Series,
                                 mrp_bins: pd.Series, cv_folds: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation for stratified models.
        """
        print("\n=== Stratified Cross-Validation ===")
        
        cv_scores = {}
        
        for scheme_name in self.models.keys():
            print(f"Cross-validating {scheme_name} models...")
            
            scheme_strata = {
                'outlet_type': outlet_type,
                'outlet_id': outlet_id,
                'mrp_bins': mrp_bins
            }[scheme_name]
            
            # Stratified CV
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            fold_scores = []
            for train_idx, val_idx in skf.split(X_train, scheme_strata):
                # Training fold
                X_fold_tr = X_train.iloc[train_idx]
                y_fold_tr = y_train.iloc[train_idx]
                strata_fold_tr = scheme_strata.iloc[train_idx]
                
                # Validation fold
                X_fold_val = X_train.iloc[val_idx]
                y_fold_val = y_train.iloc[val_idx]
                strata_fold_val = scheme_strata.iloc[val_idx]
                
                # Train temporary models on fold
                temp_trainer = StratifiedModelTrainer(optimize_hyperparams=False)
                temp_trainer.train_stratified_models(
                    X_fold_tr, y_fold_tr, strata_fold_tr, strata_fold_tr, strata_fold_tr
                )
                
                # Predict on validation fold
                fold_pred = temp_trainer._predict_single_scheme(
                    X_fold_val, strata_fold_val, scheme_name
                )
                
                # Calculate score
                fold_rmse = np.sqrt(mean_squared_error(y_fold_val, fold_pred))
                fold_scores.append(fold_rmse)
            
            cv_scores[scheme_name] = (np.mean(fold_scores), np.std(fold_scores))
            print(f"{scheme_name} CV RMSE: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
        
        return cv_scores
    
    def get_feature_importance(self, scheme_name: str = None, top_n: int = 20) -> pd.DataFrame:
        """
        Get aggregated feature importance across all models in a scheme.
        """
        if scheme_name is None:
            scheme_name = list(self.feature_importance.keys())[0]
        
        if scheme_name not in self.feature_importance:
            return pd.DataFrame()
        
        # Aggregate importance across all models in scheme
        scheme_importance = self.feature_importance[scheme_name]
        all_features = set()
        for imp_df in scheme_importance.values():
            all_features.update(imp_df['feature'].tolist())
        
        # Calculate weighted average importance
        feature_scores = {}
        for feature in all_features:
            scores = []
            weights = []
            for stratum, imp_df in scheme_importance.items():
                if feature in imp_df['feature'].values:
                    score = imp_df[imp_df['feature'] == feature]['importance'].iloc[0]
                    scores.append(score)
                    # Weight by number of samples in stratum
                    weight = self.strata_info.get(f"{scheme_name}_{stratum}", {}).get('n_samples', 1)
                    weights.append(weight)
            
            if scores:
                feature_scores[feature] = np.average(scores, weights=weights)
        
        # Create final importance DataFrame
        importance_df = pd.DataFrame(
            list(feature_scores.items()), 
            columns=['feature', 'importance']
        ).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def save_models(self, output_dir: str):
        """Save all stratified models and metadata."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save models
        for scheme_name, scheme_models in self.models.items():
            scheme_dir = f"{output_dir}/{scheme_name}"
            os.makedirs(scheme_dir, exist_ok=True)
            
            for stratum, model in scheme_models.items():
                joblib.dump(model, f"{scheme_dir}/model_{stratum}.pkl")
        
        # Save transformers
        joblib.dump(self.transformers, f"{output_dir}/transformers.pkl")
        
        # Save metadata
        metadata = {
            'strata_info': self.strata_info,
            'scores': self.scores,
            'best_params': self.best_params
        }
        joblib.dump(metadata, f"{output_dir}/metadata.pkl")
        
        print(f"All models saved to {output_dir}")
    
    def plot_stratified_performance(self):
        """Plot performance comparison across stratification schemes."""
        fig, axes = plt.subplots(1, len(self.scores), figsize=(5*len(self.scores), 6))
        if len(self.scores) == 1:
            axes = [axes]
        
        for idx, (scheme_name, scheme_scores) in enumerate(self.scores.items()):
            strata = list(scheme_scores.keys())
            scores = list(scheme_scores.values())
            
            axes[idx].bar(range(len(strata)), scores)
            axes[idx].set_xlabel('Stratum')
            axes[idx].set_ylabel('Weighted RMSE')
            axes[idx].set_title(f'{scheme_name} Performance')
            axes[idx].set_xticks(range(len(strata)))
            axes[idx].set_xticklabels(strata, rotation=45)
            
            # Add value labels
            for i, score in enumerate(scores):
                axes[idx].text(i, score + 0.01, f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()


def train_stratified_models(X_train_path: str, y_train: pd.Series,
                           X_test_path: str, outlet_type: pd.Series,
                           outlet_id: pd.Series, mrp_bins: pd.Series,
                           outlet_type_test: pd.Series, outlet_id_test: pd.Series,
                           mrp_bins_test: pd.Series, output_dir: str,
                           optimize_hyperparams: bool = True,
                           n_trials: int = 50) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Main function to train stratified models and generate predictions.
    
    Args:
        X_train_path: Path to encoded training features
        y_train: Training target values (original scale)
        X_test_path: Path to encoded test features
        outlet_type, outlet_id, mrp_bins: Training stratification columns
        outlet_type_test, outlet_id_test, mrp_bins_test: Test stratification columns
        output_dir: Directory to save models
        optimize_hyperparams: Whether to optimize hyperparameters
        n_trials: Number of optimization trials
        
    Returns:
        Tuple of (model_scores, test_predictions)
    """
    print("=== Stratified BigMart Model Training Pipeline ===")
    
    # Load encoded data
    print("Loading encoded features...")
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    
    # Initialize trainer
    trainer = StratifiedModelTrainer(optimize_hyperparams=optimize_hyperparams, n_trials=n_trials)
    
    # Train stratified models
    scores = trainer.train_stratified_models(
        X_train, y_train, outlet_type, outlet_id, mrp_bins
    )
    
    # Perform cross-validation
    cv_scores = trainer.cross_validate_stratified(
        X_train, y_train, outlet_type, outlet_id, mrp_bins
    )
    
    # Display results
    print("\n=== Stratified Model Performance Summary ===")
    for scheme_name, score in sorted(scores.items(), key=lambda x: x[1]):
        cv_info = ""
        if scheme_name in cv_scores:
            mean_cv, std_cv = cv_scores[scheme_name]
            cv_info = f" (CV: {mean_cv:.4f} ± {std_cv:.4f})"
        print(f"{scheme_name}: {score:.4f}{cv_info}")
    
    # Plot performance
    trainer.plot_stratified_performance()
    
    # Show feature importance
    for scheme_name in trainer.models.keys():
        importance_df = trainer.get_feature_importance(scheme_name)
        if not importance_df.empty:
            print(f"\nTop 10 features for {scheme_name}:")
            print(importance_df.head(10))
    
    # Generate test predictions using ensemble strategy
    print("\n=== Generating Test Predictions ===")
    test_predictions = trainer.predict_stratified(
        X_test, outlet_type_test, outlet_id_test, mrp_bins_test, 
        strategy='ensemble'
    )
    
    # Generate training predictions for analysis
    print("Generating training predictions...")
    train_predictions = trainer.predict_stratified(
        X_train, outlet_type, outlet_id, mrp_bins,
        strategy='ensemble'
    )
    
    # Save models
    print(f"Saving models to {output_dir}...")
    trainer.save_models(output_dir)
    
    return scores, test_predictions, train_predictions


if __name__ == "__main__":
    # Example usage
    import os
    
    # Setup paths
    data_dir = "/Users/whysocurious/Documents/MLDSAIProjects/SalesPred_Hackathon/data"
    model_dir = f"{data_dir}/models_stratified"
    os.makedirs(model_dir, exist_ok=True)
    
    # Load data
    train_df = pd.read_csv(f"{data_dir}/processed/train_features.csv")
    test_df = pd.read_csv(f"{data_dir}/processed/test_features.csv")
    
    # Prepare target and stratification columns
    y_train = train_df['Item_Outlet_Sales']  # Original scale
    outlet_type = train_df['Outlet_Type']
    outlet_id = train_df['Outlet_Identifier']
    mrp_bins = train_df['Item_MRP_Bins']
    
    outlet_type_test = test_df['Outlet_Type']
    outlet_id_test = test_df['Outlet_Identifier']
    mrp_bins_test = test_df['Item_MRP_Bins']
    
    # Train stratified models
    scores, predictions, train_pred = train_stratified_models(
        X_train_path=f"{data_dir}/processed/train_encoded.csv",
        y_train=y_train,
        X_test_path=f"{data_dir}/processed/test_encoded.csv",
        outlet_type=outlet_type,
        outlet_id=outlet_id,
        mrp_bins=mrp_bins,
        outlet_type_test=outlet_type_test,
        outlet_id_test=outlet_id_test,
        mrp_bins_test=mrp_bins_test,
        output_dir=model_dir,
        optimize_hyperparams=True,
        n_trials=10
    )    