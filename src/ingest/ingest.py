import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    A comprehensive data preprocessing pipeline for BigMart sales data.
    Handles missing value imputation, data cleaning, transformations, and segment preparation
    while preventing data leakage between train and test sets.
    """
    
    def __init__(self):
        self.outlet_size_mapping = None
        self.weight_imputation_mappings = None
        self.fat_content_mapping = None
        self.visibility_mapping = None
        self.mrp_bins_mapping = None  # New: for consistent MRP binning
        self.is_fitted = False
    
    def fit(self, train_data: pd.DataFrame) -> 'DataPreprocessor':
        """
        Fit the preprocessor on training data to learn transformation parameters.
        
        Args:
            train_data: Training dataset
            
        Returns:
            self: Fitted preprocessor instance
        """
        print("=== Fitting Data Preprocessor ===")
        
        # Create all transformation mappings from training data
        self.outlet_size_mapping = self._create_outlet_size_mapping(train_data)
        self.weight_imputation_mappings = self._create_weight_imputation_mappings(train_data)
        self.fat_content_mapping = self._create_fat_content_mapping()
        self.visibility_mapping = self._create_visibility_mapping(train_data)
        self.mrp_bins_mapping = self._create_mrp_bins_mapping(train_data)  # New
        
        self.is_fitted = True
        print("Preprocessor fitted successfully!")
        return self
    
    def transform(self, data: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
        """
        Transform data using fitted parameters.
        
        Args:
            data: Dataset to transform
            is_train: Whether this is training data (for target transformation)
            
        Returns:
            Transformed dataset with segment information
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        print(f"=== Transforming {'Training' if is_train else 'Test'} Data ===")
        
        # Create copy to avoid modifying original data
        data_clean = data.copy()
        
        # Apply transformations in sequence
        data_clean = self._impute_outlet_size(data_clean)
        data_clean = self._impute_item_weight(data_clean)
        data_clean = self._clean_fat_content(data_clean)
        data_clean = self._handle_visibility_zeros(data_clean)
        data_clean = self._create_mrp_bins(data_clean)  # New: create consistent MRP bins
        
        # Apply target transformation only for training data
        if is_train and 'Item_Outlet_Sales' in data_clean.columns:
            data_clean = self._transform_target_variable(data_clean)
        
        self._print_cleaning_summary(data_clean, is_train)
        return data_clean
    
    def fit_transform(self, train_data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit on training data and transform it in one step.
        
        Args:
            train_data: Training dataset
            
        Returns:
            Transformed training dataset
        """
        return self.fit(train_data).transform(train_data, is_train=True)
    
    def _create_mrp_bins_mapping(self, train_data: pd.DataFrame) -> Dict:
        """Create consistent MRP binning based on training data."""
        print("Creating MRP bins mapping...")
        
        # Calculate quantile-based bins for more balanced distribution
        mrp_values = train_data['Item_MRP'].dropna()
        
        # Define bins based on quantiles for better balance
        bins = [0, 
                mrp_values.quantile(0.2), 
                mrp_values.quantile(0.4), 
                mrp_values.quantile(0.6), 
                mrp_values.quantile(0.8), 
                mrp_values.max() + 1]
        
        labels = ['Very_Low', 'Low', 'Medium', 'High', 'Premium']
        
        # Store bins and labels for consistent application
        mrp_mapping = {
            'bins': bins,
            'labels': labels
        }
        
        print(f"MRP bins created: {[f'{labels[i]}: {bins[i]:.0f}-{bins[i+1]:.0f}' for i in range(len(labels))]}")
        return mrp_mapping
    
    def _create_mrp_bins(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create MRP bins using consistent mapping."""
        data_copy = data.copy()
        
        bins = self.mrp_bins_mapping['bins']
        labels = self.mrp_bins_mapping['labels']
        
        data_copy['Item_MRP_Bins'] = pd.cut(
            data_copy['Item_MRP'], 
            bins=bins, 
            labels=labels,
            include_lowest=True
        )
        
        # Fill any missing bins with 'Medium' as default
        data_copy['Item_MRP_Bins'] = data_copy['Item_MRP_Bins'].fillna('Medium')
        
        return data_copy
    
    def get_segment_info(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract segment information for model training.
        
        Args:
            data: Processed dataset
            
        Returns:
            DataFrame with segment columns
        """
        segment_info = pd.DataFrame({
            'Outlet_Type': data['Outlet_Type'],
            'Outlet_Identifier': data['Outlet_Identifier'],
            'Item_MRP_Bins': data['Item_MRP_Bins']
        })
        
        return segment_info
    
    def _create_outlet_size_mapping(self, train_data: pd.DataFrame) -> Dict[Tuple, str]:
        """Create outlet size imputation mapping from training data."""
        print("Creating outlet size imputation mapping...")
        
        outlet_size_mode = train_data.groupby(['Outlet_Location_Type', 'Outlet_Type'])['Outlet_Size'].apply(
            lambda x: x.mode()[0] if not x.mode().empty else 'Small'
        ).to_dict()
        
        print(f"Created {len(outlet_size_mode)} outlet size mappings")
        return outlet_size_mode
    
    def _create_weight_imputation_mappings(self, train_data: pd.DataFrame) -> Dict[str, Any]:
        """Create comprehensive item weight imputation mappings."""
        print("Creating item weight imputation mappings...")
        
        train_copy = train_data.copy()
        
        # Create median MRP mappings
        median_mrp_detailed = train_copy.pivot_table(
            index=["Item_Identifier", "Item_Type", "Outlet_Type"],
            values="Item_MRP", 
            aggfunc=np.nanmedian
        ).reset_index().rename(columns={"Item_MRP": "Median_Item_MRP"})
        
        median_mrp_general = train_copy.pivot_table(
            index=["Item_Type", "Outlet_Type"],
            values="Item_MRP", 
            aggfunc=np.nanmedian
        ).reset_index().rename(columns={"Item_MRP": "Median_Item_MRP_General"})
        
        # Merge MRP mappings to training data for weight calculation
        train_copy = train_copy.merge(median_mrp_detailed, 
                                    on=["Item_Identifier", "Item_Type", "Outlet_Type"], 
                                    how="left")
        train_copy = train_copy.merge(median_mrp_general, 
                                    on=["Item_Type", "Outlet_Type"], 
                                    how="left")
        
        # Create weight mappings at different granularity levels
        weight_mappings = self._build_weight_mappings(train_copy)
        
        # Calculate per-unit cost mapping using enhanced data
        per_unit_cost_mapping = self._calculate_per_unit_cost_mapping(train_copy, weight_mappings)
        
        mappings = {
            'median_mrp_detailed': median_mrp_detailed,
            'median_mrp_general': median_mrp_general,
            'per_unit_cost_mapping': per_unit_cost_mapping,
            **weight_mappings
        }
        
        print(f"Created comprehensive weight imputation system with {len(mappings)} mapping types")
        return mappings
    
    def _build_weight_mappings(self, train_data: pd.DataFrame) -> Dict[str, Dict]:
        """Build hierarchical weight mappings at different granularity levels."""
        weight_mappings = {}
        
        # Level 1: Most specific - Item + Outlet + MRP
        level1_stats = train_data.pivot_table(
            index=["Item_Identifier", "Outlet_Type", "Median_Item_MRP"],
            values="Item_Weight",
            aggfunc=np.nanmean
        ).reset_index()
        
        weight_mappings['level1'] = {}
        for _, row in level1_stats.iterrows():
            if not pd.isna(row['Item_Weight']):
                key = (row['Item_Identifier'], row['Outlet_Type'], row['Median_Item_MRP'])
                weight_mappings['level1'][key] = row['Item_Weight']
        
        # Level 2: Item + MRP
        level2_stats = train_data.pivot_table(
            index=["Item_Identifier", "Item_MRP"],
            values="Item_Weight",
            aggfunc=np.nanmean
        ).reset_index()
        
        weight_mappings['level2'] = {}
        for _, row in level2_stats.iterrows():
            if not pd.isna(row['Item_Weight']):
                key = (row['Item_Identifier'], row['Item_MRP'])
                weight_mappings['level2'][key] = row['Item_Weight']
        
        # Level 3: Item only
        level3_stats = train_data.pivot_table(
            index=["Item_Identifier"],
            values="Item_Weight",
            aggfunc=np.nanmean
        ).reset_index()
        
        weight_mappings['level3'] = {}
        for _, row in level3_stats.iterrows():
            if not pd.isna(row['Item_Weight']):
                weight_mappings['level3'][row['Item_Identifier']] = row['Item_Weight']
        
        # Level 4: Item Type + MRP
        level4_stats = train_data.pivot_table(
            index=["Item_Type", "Median_Item_MRP_General"],
            values="Item_Weight",
            aggfunc=np.nanmean
        ).reset_index()
        
        weight_mappings['level4'] = {}
        for _, row in level4_stats.iterrows():
            if not pd.isna(row['Item_Weight']):
                key = (row['Item_Type'], row['Median_Item_MRP_General'])
                weight_mappings['level4'][key] = row['Item_Weight']
        
        return weight_mappings
    
    def _calculate_per_unit_cost_mapping(self, train_data: pd.DataFrame, weight_mappings: Dict) -> Dict:
        """Calculate per-unit cost mapping after applying weight imputation."""
        train_enhanced = train_data.copy()
        
        # Apply weight imputation to get more complete weight data for cost calculation
        train_enhanced = self._apply_weight_mappings(train_enhanced, weight_mappings)
        
        # Calculate per-unit cost
        train_enhanced["per_unit_cost"] = train_enhanced["Item_Weight"] / train_enhanced["Item_MRP"]
        
        # Create cost mapping
        cost_stats = train_enhanced.pivot_table(
            index=["Item_Type", "Outlet_Type"],
            values="per_unit_cost",
            aggfunc=np.nanmean
        ).reset_index()
        
        per_unit_cost_mapping = {}
        for _, row in cost_stats.iterrows():
            if not pd.isna(row['per_unit_cost']):
                key = (row['Item_Type'], row['Outlet_Type'])
                per_unit_cost_mapping[key] = row['per_unit_cost']
        
        return per_unit_cost_mapping
    
    def _apply_weight_mappings(self, data: pd.DataFrame, weight_mappings: Dict) -> pd.DataFrame:
        """Apply hierarchical weight imputation mappings."""
        data_copy = data.copy()
        
        # Level 1: Item + Outlet + MRP
        missing_mask = data_copy['Item_Weight'].isnull()
        for idx in data_copy[missing_mask].index:
            row = data_copy.loc[idx]
            key = (row['Item_Identifier'], row['Outlet_Type'], row['Median_Item_MRP'])
            if key in weight_mappings['level1']:
                data_copy.loc[idx, 'Item_Weight'] = weight_mappings['level1'][key]
        
        # Level 2: Item + MRP
        missing_mask = data_copy['Item_Weight'].isnull()
        for idx in data_copy[missing_mask].index:
            row = data_copy.loc[idx]
            key = (row['Item_Identifier'], row['Item_MRP'])
            if key in weight_mappings['level2']:
                data_copy.loc[idx, 'Item_Weight'] = weight_mappings['level2'][key]
        
        # Level 3: Item only
        missing_mask = data_copy['Item_Weight'].isnull()
        for idx in data_copy[missing_mask].index:
            row = data_copy.loc[idx]
            if row['Item_Identifier'] in weight_mappings['level3']:
                data_copy.loc[idx, 'Item_Weight'] = weight_mappings['level3'][row['Item_Identifier']]
        
        # Level 4: Item Type + MRP
        missing_mask = data_copy['Item_Weight'].isnull()
        for idx in data_copy[missing_mask].index:
            row = data_copy.loc[idx]
            key = (row['Item_Type'], row['Median_Item_MRP_General'])
            if key in weight_mappings['level4']:
                data_copy.loc[idx, 'Item_Weight'] = weight_mappings['level4'][key]
        
        return data_copy
    
    def _create_fat_content_mapping(self) -> Dict[str, str]:
        """Create fat content standardization mapping."""
        return {
            'LF': 'Low Fat',
            'low fat': 'Low Fat', 
            'reg': 'Regular'
        }
    
    def _create_visibility_mapping(self, train_data: pd.DataFrame) -> Dict[Tuple, float]:
        """Create visibility imputation mapping for zero values."""
        print("Creating visibility imputation mapping...")
        
        # Calculate mean visibility by Item_Type and Outlet_Type for non-zero values
        visibility_stats = train_data[train_data['Item_Visibility'] > 0].groupby(
            ['Item_Type', 'Outlet_Type']
        )['Item_Visibility'].mean()
        
        visibility_mapping = visibility_stats.to_dict()
        print(f"Created {len(visibility_mapping)} visibility mappings")
        return visibility_mapping
    
    def _impute_outlet_size(self, data: pd.DataFrame) -> pd.DataFrame:
        """Impute outlet sizes using pre-computed mapping."""
        data_copy = data.copy()
        missing_before = data_copy['Outlet_Size'].isnull().sum()
        
        if missing_before > 0:
            print(f"Imputing {missing_before} missing outlet sizes...")
            
            data_copy['Outlet_Size'] = data_copy.apply(
                lambda row: self.outlet_size_mapping.get(
                    (row['Outlet_Location_Type'], row['Outlet_Type']), 'Small'
                ) if pd.isna(row['Outlet_Size']) else row['Outlet_Size'], 
                axis=1
            )
            
            missing_after = data_copy['Outlet_Size'].isnull().sum()
            print(f"Successfully imputed {missing_before - missing_after} outlet sizes")
        
        return data_copy
    
    def _impute_item_weight(self, data: pd.DataFrame) -> pd.DataFrame:
        """Impute item weights using hierarchical approach."""
        data_copy = data.copy()
        missing_before = data_copy['Item_Weight'].isnull().sum()
        
        if missing_before > 0:
            print(f"Imputing {missing_before} missing item weights...")
            
            # Add MRP mappings
            data_copy = data_copy.merge(
                self.weight_imputation_mappings['median_mrp_detailed'], 
                on=["Item_Identifier", "Item_Type", "Outlet_Type"], 
                how="left"
            )
            data_copy = data_copy.merge(
                self.weight_imputation_mappings['median_mrp_general'], 
                on=["Item_Type", "Outlet_Type"], 
                how="left"
            )
            
            # Apply hierarchical imputation
            weight_mappings = {k: v for k, v in self.weight_imputation_mappings.items() 
                             if k.startswith('level')}
            data_copy = self._apply_weight_mappings(data_copy, weight_mappings)
            
            # Final level: Per-unit cost approach
            missing_mask = data_copy['Item_Weight'].isnull()
            per_unit_cost_mapping = self.weight_imputation_mappings['per_unit_cost_mapping']
            
            for idx in data_copy[missing_mask].index:
                row = data_copy.loc[idx]
                key = (row['Item_Type'], row['Outlet_Type'])
                if key in per_unit_cost_mapping:
                    estimated_weight = row['Item_MRP'] * per_unit_cost_mapping[key]
                    data_copy.loc[idx, 'Item_Weight'] = estimated_weight
            
            # Clean up temporary columns
            temp_cols = ['Median_Item_MRP', 'Median_Item_MRP_General']
            data_copy.drop(columns=[col for col in temp_cols if col in data_copy.columns], 
                          inplace=True)
            
            missing_after = data_copy['Item_Weight'].isnull().sum()
            print(f"Successfully imputed {missing_before - missing_after} item weights")
        
        return data_copy
    
    def _clean_fat_content(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize fat content values."""
        data_copy = data.copy()
        
        # Standardize fat content values
        data_copy['Item_Fat_Content'] = data_copy['Item_Fat_Content'].replace(self.fat_content_mapping)
        
        # Create binary flag
        data_copy['Low_Fat_Flag'] = (data_copy['Item_Fat_Content'] == 'Low Fat').astype(int)
        
        return data_copy
    
    def _handle_visibility_zeros(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle zero visibility values using mean imputation."""
        data_copy = data.copy()
        zero_visibility_count = (data_copy['Item_Visibility'] == 0).sum()
        
        if zero_visibility_count > 0:
            print(f"Handling {zero_visibility_count} zero visibility values...")
            
            data_copy['Item_Visibility'] = data_copy.apply(
                lambda row: self.visibility_mapping.get(
                    (row['Item_Type'], row['Outlet_Type']), row['Item_Visibility']
                ) if row['Item_Visibility'] == 0 else row['Item_Visibility'], 
                axis=1
            )
            
            zero_after = (data_copy['Item_Visibility'] == 0).sum()
            print(f"Successfully handled {zero_visibility_count - zero_after} zero visibility values")
        
        return data_copy
    
    def _transform_target_variable(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply log transformation to target variable (kept for compatibility)."""
        data_copy = data.copy()
        
        if 'Item_Outlet_Sales' in data_copy.columns:
            print("Applying log transformation to target variable...")
            
            original_skewness = data_copy['Item_Outlet_Sales'].skew()
            data_copy['log_sales'] = np.log1p(data_copy['Item_Outlet_Sales'])
            log_skewness = data_copy['log_sales'].skew()

            data_copy.drop(columns=['log_sales'], inplace=True)             ##############
            
            print(f"Original sales skewness: {original_skewness:.3f}")
            print(f"Log sales skewness: {log_skewness:.3f}")
        
        return data_copy
    
    def _print_cleaning_summary(self, data: pd.DataFrame, is_train: bool):
        """Print summary of data cleaning results."""
        print(f"\n=== {'Training' if is_train else 'Test'} Data Cleaning Summary ===")
        
        missing_values = data.isnull().sum()
        if missing_values.sum() > 0:
            print("Remaining missing values:")
            print(missing_values[missing_values > 0])
        else:
            print("No missing values remaining!")
        
        print(f"Final dataset shape: {data.shape}")
        
        # Print segment distribution
        if 'Item_MRP_Bins' in data.columns:
            print("\nMRP Bins distribution:")
            print(data['Item_MRP_Bins'].value_counts())
        
        if 'Outlet_Type' in data.columns:
            print("\nOutlet Type distribution:")
            print(data['Outlet_Type'].value_counts())
        
        print("Data preprocessing completed successfully!")


def load_and_preprocess_data(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main function to load and preprocess BigMart sales data with segment preparation.
    
    Args:
        data_path: Path to data directory
        
    Returns:
        Tuple of (processed_train, processed_test) DataFrames
    """
    print("=== BigMart Data Preprocessing Pipeline ===")
    
    # Load raw data
    print("Loading raw data...")
    train = pd.read_csv(f"{data_path}/train_v9rqX0R.csv")
    test = pd.read_csv(f"{data_path}/test_AbJTz2l.csv")
    
    print(f"Loaded train data: {train.shape}")
    print(f"Loaded test data: {test.shape}")
    
    # Initialize and fit preprocessor
    preprocessor = DataPreprocessor()
    
    # Fit on training data and transform both datasets
    train_processed = preprocessor.fit_transform(train)
    test_processed = preprocessor.transform(test, is_train=False)
    
    return train_processed, test_processed


# Usage example
if __name__ == "__main__":
    # Set your data path
    data_path = "/Users/whysocurious/Documents/MLDSAIProjects/BigMartSalesPred_Hackathon/data/raw"
    
    # Process the data
    train_clean, test_clean = load_and_preprocess_data(data_path)
    
    print("\nPreprocessing completed!")
    print(f"Processed train shape: {train_clean.shape}")
    print(f"Processed test shape: {test_clean.shape}")
    
    # Example of extracting segment information
    preprocessor = DataPreprocessor()
    preprocessor.fit(train_clean)
    
    segment_info_train = preprocessor.get_segment_info(train_clean)
    segment_info_test = preprocessor.get_segment_info(test_clean)
    
    print(f"\nSegment info shapes:")
    print(f"Train segments: {segment_info_train.shape}")
    print(f"Test segments: {segment_info_test.shape}")