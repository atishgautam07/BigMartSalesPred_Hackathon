import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional, Any
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


class ModelTrainer:
    """
    Handles model training, evaluation, and prediction for BigMart sales.
    Supports multiple algorithms and ensemble methods.
    """
    
    def __init__(self):
        self.models = {}
        self.scores = {}
        self.feature_importance = {}
        self.best_model = None
        self.ensemble_weights = None
        
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        stratify_col: Optional[pd.Series] = None,
                        val_size: float = 0.2) -> Dict[str, float]:
        """
        Train multiple models and evaluate performance.
        
        Args:
            X_train: Training features
            y_train: Training target
            stratify_col: Column to use for stratified split
            val_size: Validation set size
            
        Returns:
            Dictionary of model scores
        """
        print("=== Model Training Pipeline ===")
        
        # Create train-validation split
        X_tr, X_val, y_tr, y_val = self._create_validation_split(
            X_train, y_train, stratify_col, val_size
        )
        
        # Train different models
        self._train_baseline(X_tr, X_val, y_tr, y_val)
        self._train_linear_regression(X_tr, X_val, y_tr, y_val)
        self._train_random_forest(X_tr, X_val, y_tr, y_val)
        self._train_xgboost(X_tr, X_val, y_tr, y_val)
        self._train_lightgbm(X_tr, X_val, y_tr, y_val)
        
        # Select best model
        self._select_best_model()
        
        # Train ensemble if multiple good models
        if len([s for s in self.scores.values() if s < 1200]) > 2:
            self._train_ensemble(X_tr, X_val, y_tr, y_val)
        
        return self.scores
    
    def cross_validate(self, X_train: pd.DataFrame, y_train: pd.Series,
                      stratify_col: Optional[pd.Series] = None,
                      cv_folds: int = 5) -> Dict[str, Tuple[float, float]]:
        """
        Perform cross-validation for robust evaluation.
        
        Args:
            X_train: Training features
            y_train: Training target
            stratify_col: Column for stratification
            cv_folds: Number of CV folds
            
        Returns:
            Dictionary of model names to (mean_score, std_score)
        """
        print("\n=== Cross-Validation ===")
        
        cv_scores = {}
        
        # Only CV the best performing models
        top_models = ['XGBoost', 'LightGBM']
        
        for model_name in top_models:
            if model_name in self.models:
                print(f"Cross-validating {model_name}...")
                scores = self._cv_single_model(
                    self.models[model_name], 
                    X_train, y_train, 
                    stratify_col, cv_folds
                )
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                cv_scores[model_name] = (mean_score, std_score)
                print(f"{model_name} CV RMSE: {mean_score:.2f} (+/- {std_score:.2f})")
                
        return cv_scores
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from tree-based models.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if 'XGBoost' in self.models:
            model = self.models['XGBoost']
            importance = model.feature_importances_
            feature_names = model.get_booster().feature_names
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            self.feature_importance['XGBoost'] = importance_df
            
            return importance_df.head(top_n)
        
        return pd.DataFrame()
    
    def predict(self, X_test: pd.DataFrame, model_name: Optional[str] = None) -> np.ndarray:
        """
        Make predictions using specified or best model.
        
        Args:
            X_test: Test features
            model_name: Specific model to use, or None for best/ensemble
            
        Returns:
            Array of predictions
        """
        if model_name:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            return self.models[model_name].predict(X_test)
        
        # Use ensemble if available, otherwise best model
        if 'Ensemble' in self.models:
            return self._predict_ensemble(X_test)
        elif self.best_model:
            return self.models[self.best_model].predict(X_test)
        else:
            raise ValueError("No trained models available")
    
    def save_models(self, output_dir: str):
        """Save trained models to disk."""
        for name, model in self.models.items():
            if name != 'Baseline':  # Skip baseline
                joblib.dump(model, f"{output_dir}/{name.lower()}_model.pkl")
                
        # Save ensemble weights if available
        if self.ensemble_weights:
            joblib.dump(self.ensemble_weights, f"{output_dir}/ensemble_weights.pkl")
            
    def _create_validation_split(self, X: pd.DataFrame, y: pd.Series, 
                               stratify_col: Optional[pd.Series], 
                               val_size: float) -> Tuple:
        """Create stratified train-validation split."""
        print(f"Creating {int((1-val_size)*100)}/{int(val_size*100)} train/validation split...")
        
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=val_size, random_state=42,
            stratify=stratify_col
        )
        
        print(f"Training set: {X_tr.shape}")
        print(f"Validation set: {X_val.shape}")
        
        return X_tr, X_val, y_tr, y_val
    
    def _train_baseline(self, X_tr, X_val, y_tr, y_val):
        """Train baseline mean predictor."""
        print("\nTraining baseline model...")
        
        mean_pred = y_tr.mean()
        val_pred = np.full(len(y_val), mean_pred)
        
        rmse = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(val_pred)))
        self.models['Baseline'] = lambda x: np.full(len(x), mean_pred)
        self.scores['Baseline'] = rmse
        
        print(f"Baseline RMSE: {rmse:.2f}")
        
    def _train_linear_regression(self, X_tr, X_val, y_tr, y_val):
        """Train linear regression model."""
        print("Training Linear Regression...")
        
        model = LinearRegression()
        model.fit(X_tr, y_tr)
        val_pred = model.predict(X_val)
        
        rmse = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(val_pred)))
        self.models['Linear Regression'] = model
        self.scores['Linear Regression'] = rmse
        
        print(f"Linear Regression RMSE: {rmse:.2f}")
        
    def _train_random_forest(self, X_tr, X_val, y_tr, y_val):
        """Train Random Forest model."""
        print("Training Random Forest...")
        
        model = RandomForestRegressor(
            n_estimators=900,
            max_depth=15,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=3
        )
        model.fit(X_tr, y_tr)
        val_pred = model.predict(X_val)
        
        rmse = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(val_pred)))
        self.models['Random Forest'] = model
        self.scores['Random Forest'] = rmse
        
        print(f"Random Forest RMSE: {rmse:.2f}")
        
    def _train_xgboost(self, X_tr, X_val, y_tr, y_val):
        """Train XGBoost model."""
        print("Training XGBoost...")
        
        model = xgb.XGBRegressor(
            n_estimators=900,
            max_depth=12,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=3,
            early_stopping_rounds=50
        )
        
        # Use early stopping for better generalization
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        val_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(val_pred)))
        self.models['XGBoost'] = model
        self.scores['XGBoost'] = rmse
        
        print(f"XGBoost RMSE: {rmse:.2f}")
        
    def _train_lightgbm(self, X_tr, X_val, y_tr, y_val):
        """Train LightGBM model."""
        print("Training LightGBM...")
        
        model = lgb.LGBMRegressor(
            n_estimators=900,
            max_depth=12,
            learning_rate=0.05,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            random_state=42,
            verbose=-1
        )
        
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        val_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(val_pred)))
        self.models['LightGBM'] = model
        self.scores['LightGBM'] = rmse
        
        print(f"LightGBM RMSE: {rmse:.2f}")
        
    def _train_ensemble(self, X_tr, X_val, y_tr, y_val):
        """Train weighted ensemble of best models."""
        print("Training ensemble...")
        
        # Get predictions from top models
        ensemble_models = ['XGBoost', 'LightGBM', 'Random Forest']
        predictions = {}
        
        for model_name in ensemble_models:
            if model_name in self.models:
                predictions[model_name] = self.models[model_name].predict(X_val)
                
        # Find optimal weights using simple grid search
        best_score = float('inf')
        best_weights = None
        
        # Try different weight combinations
        for w1 in np.arange(0.3, 0.6, 0.1):
            for w2 in np.arange(0.3, 0.6, 0.1):
                w3 = 1 - w1 - w2
                if w3 >= 0.1:  # Ensure meaningful weight
                    weights = [w1, w2, w3]
                    ensemble_pred = sum(
                        w * predictions[m] 
                        for w, m in zip(weights, ensemble_models)
                        if m in predictions
                    )
                    score = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(ensemble_pred)))
                    
                    if score < best_score:
                        best_score = score
                        best_weights = weights
                        
        self.ensemble_weights = dict(zip(ensemble_models, best_weights))
        self.scores['Ensemble'] = best_score
        self.models['Ensemble'] = 'ensemble'  # Placeholder
        
        print(f"Ensemble RMSE: {best_score:.2f}")
        print(f"Weights: {self.ensemble_weights}")
        
    def _predict_ensemble(self, X_test: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions."""
        predictions = []
        
        for model_name, weight in self.ensemble_weights.items():
            if model_name in self.models:
                pred = self.models[model_name].predict(X_test)
                predictions.append(weight * pred)
                
        return sum(predictions)
    
    def _cv_single_model(self, model, X: pd.DataFrame, y: pd.Series,
                        stratify_col: Optional[pd.Series], 
                        cv_folds: int) -> List[float]:
        """Perform CV for a single model."""
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = []
        
        # Use stratify column if provided
        split_col = stratify_col if stratify_col is not None else np.zeros(len(X))
        
        for train_idx, val_idx in skf.split(X, split_col):
            X_fold_tr = X.iloc[train_idx]
            X_fold_val = X.iloc[val_idx]
            y_fold_tr = y.iloc[train_idx]
            y_fold_val = y.iloc[val_idx]
            
            # Clone model to avoid refitting
            model_clone = model.__class__(**model.get_params())
            
            # Handle early stopping for boosting models
            if isinstance(model_clone, (xgb.XGBRegressor, lgb.LGBMRegressor)):
                if isinstance(model_clone, xgb.XGBRegressor):
                    model_clone.fit(
                        X_fold_tr, y_fold_tr,
                        eval_set=[(X_fold_val, y_fold_val)],
                        # early_stopping_rounds=50,
                        verbose=False
                    )
                else:  # LightGBM
                    model_clone.fit(
                        X_fold_tr, y_fold_tr,
                        eval_set=[(X_fold_val, y_fold_val)],
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                    )
            else:
                model_clone.fit(X_fold_tr, y_fold_tr)
            
            fold_pred = model_clone.predict(X_fold_val)
            fold_rmse = np.sqrt(mean_squared_error(np.expm1(y_fold_val), np.expm1(fold_pred)))
            scores.append(fold_rmse)
            
        return scores
    
    def _select_best_model(self):
        """Select best performing model."""
        # Exclude baseline from selection
        model_scores = {k: v for k, v in self.scores.items() if k != 'Baseline'}
        self.best_model = min(model_scores, key=model_scores.get)
        
        print(f"\nBest model: {self.best_model} (RMSE: {self.scores[self.best_model]:.2f})")
    
    def plot_model_comparison(self):
        """Plot model performance comparison."""
        plt.figure(figsize=(10, 6))
        
        models = list(self.scores.keys())
        scores = list(self.scores.values())
        
        # Create bar plot
        bars = plt.bar(models, scores)
        
        # Color best model differently
        best_idx = models.index(self.best_model) if self.best_model else -1
        if best_idx >= 0:
            bars[best_idx].set_color('darkgreen')
            
        # Add value labels
        for i, (model, score) in enumerate(zip(models, scores)):
            plt.text(i, score + 10, f'{score:.1f}', ha='center', va='bottom')
            
        plt.xlabel('Model')
        plt.ylabel('RMSE')
        plt.title('Model Performance Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, top_n: int = 20):
        """Plot feature importance from best tree model."""
        importance_df = self.get_feature_importance(top_n)
        
        if importance_df.empty:
            print("No feature importance available")
            return
            
        plt.figure(figsize=(10, 8))
        
        # Reverse order for horizontal bar plot
        features = importance_df['feature'].iloc[::-1]
        importances = importance_df['importance'].iloc[::-1]
        
        plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importances (XGBoost)')
        plt.tight_layout()
        plt.show()


def train_models(X_train_path: str, 
                y_train: pd.Series,
                X_test_path: str,
                outlet_types: pd.Series,
                output_dir: str) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Main function to train models and generate predictions.
    
    Args:
        X_train_path: Path to encoded training features
        y_train: Training target values
        X_test_path: Path to encoded test features
        outlet_types: Outlet types for stratification
        output_dir: Directory to save models and predictions
        
    Returns:
        Tuple of (model_scores, test_predictions)
    """
    print("=== BigMart Model Training Pipeline ===")
    
    # Load encoded data
    print("Loading encoded features...")
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Train all models
    scores = trainer.train_all_models(X_train, y_train, outlet_types)
    
    # Perform cross-validation
    cv_scores = trainer.cross_validate(X_train, y_train, outlet_types)
    
    # Display results
    print("\n=== Model Performance Summary ===")
    for model_name, score in sorted(scores.items(), key=lambda x: x[1]):
        cv_info = ""
        if model_name in cv_scores:
            mean_cv, std_cv = cv_scores[model_name]
            cv_info = f" (CV: {mean_cv:.2f} ± {std_cv:.2f})"
        print(f"{model_name}: {score:.2f}{cv_info}")
    
    # Plot comparisons
    trainer.plot_model_comparison()
    trainer.plot_feature_importance()
    
    # Generate test predictions
    print("\n=== Generating Test Predictions ===")
    test_predictions = trainer.predict(X_test)
    
    # Save models
    print(f"Saving models to {output_dir}...")
    trainer.save_models(output_dir)
    
    return scores, test_predictions


if __name__ == "__main__":
    # Example usage
    import os
    
    # Setup paths
    data_dir = "/Users/whysocurious/Documents/MLDSAIProjects/BigMartSalesPred_Hackathon/data"
    model_dir = f"{data_dir}/models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Load target and stratification column
    train_df = pd.read_csv(f"{data_dir}/processed/train_features.csv")
    y_train = train_df['Item_Outlet_Sales']
    outlet_types = train_df['Outlet_Type']
    
    # Train models
    scores, predictions = train_models(
        X_train_path=f"{data_dir}/processed/train_encoded.csv",
        y_train=y_train,
        X_test_path=f"{data_dir}/processed/test_encoded.csv",
        outlet_types=outlet_types,
        output_dir=model_dir
    )
    
    # Create submission
    test_df = pd.read_csv(f"{data_dir}/processed/test_features.csv")
    submission = pd.DataFrame({
        'Item_Identifier': test_df['Item_Identifier'],
        'Outlet_Identifier': test_df['Outlet_Identifier'],
        'Item_Outlet_Sales': np.expm1(predictions)
    })
    
    submission.to_csv(f"{data_dir}/submission.csv", index=False)
    print("\nSubmission file created!")