import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class FeatureEncoder:
    """
    Handles all feature encoding and scaling operations for BigMart data.
    Maintains separation between fit and transform to prevent data leakage.
    """
    
    def __init__(self):
        self.encoders = {}
        self.scalers = {}
        self.encoding_maps = {}
        self.feature_lists = {}
        self._is_fitted = False
        
    def fit(self, train_df: pd.DataFrame, target_col: str = 'Item_Outlet_Sales') -> 'FeatureEncoder':
        """
        Learn encoding parameters from training data.
        
        Args:
            train_df: Training dataframe with all features
            target_col: Name of target column for target encoding
            
        Returns:
            self: Fitted FeatureEncoder instance
        """
        print("Fitting feature encoder...")
        
        # Identify feature types automatically
        self._identify_feature_types(train_df, target_col)
        
        # Fit different encoding strategies
        self._fit_target_encoding(train_df, target_col)
        self._fit_frequency_encoding(train_df)
        self._fit_label_encoding(train_df)
        self._fit_onehot_encoding(train_df)
        
        self._is_fitted = True
        print("Feature encoder fitted successfully!")
        return self
    
    def transform(self, df: pd.DataFrame, 
                 is_train: bool = True,
                 target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Apply learned encodings to dataframe.
        
        Args:
            df: Dataframe to transform
            is_train: Whether this is training data
            target_col: Target column name if available
            
        Returns:
            Encoded dataframe ready for modeling
        """
        if not self._is_fitted:
            raise ValueError("FeatureEncoder must be fitted before transform")
            
        print(f"Encoding {'training' if is_train else 'test'} features...")
        
        df_encoded = df.copy()
        
        # Apply encodings in sequence
        if is_train and target_col:
            df_encoded = self._apply_target_encoding_train(df_encoded, target_col)
        else:
            df_encoded = self._apply_target_encoding_test(df_encoded)
            
        df_encoded = self._apply_frequency_encoding(df_encoded)
        df_encoded = self._apply_label_encoding(df_encoded)
        
        # Get final feature matrix
        feature_matrix = self._prepare_feature_matrix(df_encoded)
        
        # Apply one-hot encoding
        feature_matrix = self._apply_onehot_encoding(feature_matrix)
        
        # Apply scaling
        feature_matrix = self._apply_scaling(feature_matrix)
        
        return feature_matrix
    
    def fit_transform(self, train_df: pd.DataFrame, 
                     target_col: str = 'Item_Outlet_Sales') -> pd.DataFrame:
        """Fit and transform training data in one step."""
        return self.fit(train_df, target_col).transform(train_df, True, target_col)
    
    def _identify_feature_types(self, df: pd.DataFrame, target_col: str):
        """Automatically identify different feature types."""
        # Define feature categories
        self.feature_lists['low_cardinality_cat'] = [
            'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'Item_Fat_Content'
        ]
        
        self.feature_lists['high_cardinality_cat'] = [
            'Item_Type', 'Item_Identifier', 'Outlet_Identifier'
        ]
        
        self.feature_lists['ordinal_cat'] = [
            'Item_MRP_Bins', 'Complement_Group'
        ]
        
        self.feature_lists['binary'] = [
            'Low_Fat_Flag', 'Is_Tier1_Supermarket', 'Large_Outlet_Premium_Item',
            'Drinks_In_Grocery', 'Above_Avg_Visibility_In_Category'
        ]
        
        # Identify numerical features dynamically
        potential_numerical = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target, binary, and identifier columns
        exclude_cols = [target_col] + self.feature_lists['binary']
        exclude_cols += ['Outlet_Establishment_Year']  # Original feature, not needed
        
        self.feature_lists['numerical'] = [
            col for col in potential_numerical 
            if col not in exclude_cols
        ]
        
        print(f"Identified {len(self.feature_lists['numerical'])} numerical features")
        print(f"Identified {len(self.feature_lists['binary'])} binary features")
        
    def _fit_target_encoding(self, train_df: pd.DataFrame, target_col: str):
        """Fit target encoding for high cardinality features using CV."""
        print("Fitting target encoding for Item_Type...")
        
        # Store CV fold means for Item_Type
        cv_folds = 5
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Calculate overall mean
        self.encoding_maps['overall_mean'] = train_df[target_col].mean()
        
        # Store fold-wise encodings for training
        self.encoding_maps['item_type_cv'] = {}
        
        for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(train_df, train_df['Outlet_Type'])
        ):
            fold_means = train_df.iloc[train_idx].groupby('Item_Type')[target_col].mean()
            self.encoding_maps['item_type_cv'][fold_idx] = fold_means
            
        # Store full training set means for test encoding
        self.encoding_maps['item_type_full'] = train_df.groupby('Item_Type')[target_col].mean()
        
    def _apply_target_encoding_train(self, train_df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Apply target encoding to training data using CV."""
        df = train_df.copy()
        
        # Initialize encoded column
        encoded_values = np.zeros(len(df))
        
        # Apply CV encoding
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(df, df['Outlet_Type'])
        ):
            # Use pre-computed fold means
            fold_means = self.encoding_maps['item_type_cv'][fold_idx]
            
            # Apply to validation fold
            encoded_values[val_idx] = df.iloc[val_idx]['Item_Type'].map(
                fold_means
            ).fillna(self.encoding_maps['overall_mean'])
            
        df['Item_Type_Encoded'] = encoded_values
        return df
    
    def _apply_target_encoding_test(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """Apply target encoding to test data."""
        df = test_df.copy()
        
        # Use full training set means
        df['Item_Type_Encoded'] = df['Item_Type'].map(
            self.encoding_maps['item_type_full']
        ).fillna(self.encoding_maps['overall_mean'])
        
        return df
    
    def _fit_frequency_encoding(self, train_df: pd.DataFrame):
        """Fit frequency encoding for identifier columns."""
        print("Fitting frequency encoding...")
        
        for col in ['Item_Identifier', 'Outlet_Identifier']:
            self.encoding_maps[f'{col}_freq'] = train_df[col].value_counts().to_dict()
            
    def _apply_frequency_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply frequency encoding."""
        df_encoded = df.copy()
        
        for col in ['Item_Identifier', 'Outlet_Identifier']:
            freq_map = self.encoding_maps[f'{col}_freq']
            df_encoded[f'{col}_Freq'] = df_encoded[col].map(freq_map).fillna(1)
            
        return df_encoded
    
    def _fit_label_encoding(self, train_df: pd.DataFrame):
        """Fit label encoders for ordinal features."""
        print("Fitting label encoding...")
        
        for col in self.feature_lists['ordinal_cat']:
            le = LabelEncoder()
            # Fit on unique values to handle unseen categories
            unique_vals = train_df[col].astype(str).unique()
            le.fit(unique_vals)
            self.encoders[f'{col}_label'] = le
            
    def _apply_label_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply label encoding."""
        df_encoded = df.copy()
        
        for col in self.feature_lists['ordinal_cat']:
            le = self.encoders[f'{col}_label']
            # Handle unseen categories gracefully
            col_str = df_encoded[col].astype(str)
            
            # Map unseen values to a default
            known_classes = set(le.classes_)
            col_str_mapped = col_str.apply(
                lambda x: x if x in known_classes else le.classes_[0]
            )
            
            df_encoded[f'{col}_Encoded'] = le.transform(col_str_mapped)
            
        return df_encoded
    
    def _fit_onehot_encoding(self, train_df: pd.DataFrame):
        """Fit one-hot encoder for categorical features."""
        print("Fitting one-hot encoding...")
        
        self.encoders['onehot'] = OneHotEncoder(
            sparse_output=False, 
            drop='first', 
            handle_unknown='ignore'
        )
        
        # Fit on low cardinality categorical features
        cat_features = train_df[self.feature_lists['low_cardinality_cat']]
        self.encoders['onehot'].fit(cat_features)
        
        # Store feature names
        self.encoding_maps['onehot_features'] = self.encoders['onehot'].get_feature_names_out(
            self.feature_lists['low_cardinality_cat']
        )
        
    def _prepare_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare the feature matrix with selected columns."""
        # Update numerical features with encoded ones
        numerical_cols = self.feature_lists['numerical'].copy()
        numerical_cols.extend([
            'Item_Type_Encoded', 
            'Item_Identifier_Freq', 
            'Outlet_Identifier_Freq',
            'Item_MRP_Bins_Encoded', 
            'Complement_Group_Encoded'
        ])
        
        # Combine all features
        feature_cols = (
            numerical_cols + 
            self.feature_lists['binary'] + 
            self.feature_lists['low_cardinality_cat']
        )
        
        # Filter to existing columns
        existing_cols = [col for col in feature_cols if col in df.columns]
        
        return df[existing_cols].copy()
    
    def _apply_onehot_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply one-hot encoding and combine with other features."""
        # Separate categorical features
        cat_cols = self.feature_lists['low_cardinality_cat']
        cat_data = df[cat_cols]
        other_data = df.drop(columns=cat_cols)
        
        # Apply one-hot encoding
        cat_encoded = self.encoders['onehot'].transform(cat_data)
        cat_encoded_df = pd.DataFrame(
            cat_encoded, 
            columns=self.encoding_maps['onehot_features'],
            index=df.index
        )
        
        # Combine
        return pd.concat([other_data, cat_encoded_df], axis=1)
    
    def _apply_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply standard scaling to numerical features."""
        # Identify numerical columns (exclude one-hot encoded)
        onehot_cols = self.encoding_maps['onehot_features']
        numerical_cols = [col for col in df.columns if col not in onehot_cols]
        
        # Initialize scaler if not exists
        if 'standard' not in self.scalers:
            print("Fitting standard scaler...")
            self.scalers['standard'] = StandardScaler()
            self.scalers['standard'].fit(df[numerical_cols])
        
        # Apply scaling
        df_scaled = df.copy()
        df_scaled[numerical_cols] = self.scalers['standard'].transform(df[numerical_cols])
        
        return df_scaled
    
    def get_feature_names(self) -> List[str]:
        """Get list of all features after encoding."""
        if not self._is_fitted:
            raise ValueError("FeatureEncoder must be fitted first")
            
        # Numerical + encoded features
        numerical_and_encoded = self.feature_lists['numerical'].copy()
        numerical_and_encoded.extend([
            'Item_Type_Encoded',
            'Item_Identifier_Freq',
            'Outlet_Identifier_Freq',
            'Item_MRP_Bins_Encoded',
            'Complement_Group_Encoded'
        ])
        
        # Add binary and one-hot encoded features
        all_features = (
            numerical_and_encoded + 
            self.feature_lists['binary'] +
            list(self.encoding_maps['onehot_features'])
        )
        
        return all_features


def encode_features(train_features_path: str, 
                   test_features_path: str,
                   output_train_path: str,
                   output_test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main function to encode features for modeling.
    
    Args:
        train_features_path: Path to training data with engineered features
        test_features_path: Path to test data with engineered features
        output_train_path: Path to save encoded training data
        output_test_path: Path to save encoded test data
        
    Returns:
        Tuple of (encoded_train, encoded_test) DataFrames
    """
    print("=== Feature Encoding Pipeline ===")
    
    # Load featured data
    print("Loading featured data...")
    train_df = pd.read_csv(train_features_path)
    test_df = pd.read_csv(test_features_path)
    
    # Initialize encoder
    encoder = FeatureEncoder()
    
    # Fit on train and transform both
    print("\nEncoding features...")
    X_train = encoder.fit_transform(train_df, 'Item_Outlet_Sales')
    X_test = encoder.transform(test_df, is_train=False)
    
    # Extract target if present
    y_train = None
    if 'Item_Outlet_Sales' in train_df.columns:
        y_train = train_df['Item_Outlet_Sales']
        
    # Save encoded data
    X_train.to_csv(output_train_path, index=False)
    X_test.to_csv(output_test_path, index=False)
    
    print(f"\nEncoded shapes - Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Total features after encoding: {len(encoder.get_feature_names())}")
    
    return X_train, X_test, y_train