import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Feature engineering pipeline for BigMart sales prediction.
    Creates various features while preventing data leakage between train/test sets.
    Modified to support stratified modeling with proper MRP binning.
    """
    
    def __init__(self):
        # Store mappings learned from training data
        self.item_outlet_stats = None
        self.item_stats = None
        self.outlet_stats = None
        self.complement_groups = None
        self.outlet_location_counts = None
        self.outlet_premium_ratios = None
        self.mrp_bins_info = None  # NEW: Store MRP binning information
        self._is_fitted = False
        
    def fit(self, train_df: pd.DataFrame) -> 'FeatureEngineer':
        """
        Learn statistics and mappings from training data.
        
        Args:
            train_df: Training dataframe with target variable
            
        Returns:
            self: Fitted FeatureEngineer instance
        """
        print("Fitting feature engineering pipeline...")
        
        # NEW: Create MRP bins based on training data distribution
        self._create_mrp_bins(train_df)
        
        # Store statistics that will be used for both train and test
        self._calculate_item_outlet_statistics(train_df)
        self._calculate_outlet_statistics(train_df)
        self._calculate_outlet_characteristics(train_df)
        self._define_complement_groups()
        
        self._is_fitted = True
        print("Feature engineering pipeline fitted successfully!")
        return self
    
    def transform(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        Apply feature engineering transformations.
        
        Args:
            df: Dataframe to transform
            is_train: Whether this is training data
            
        Returns:
            Dataframe with engineered features
        """
        if not self._is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")
        
        print(f"Transforming {'training' if is_train else 'test'} data...")
        
        # Create copy to avoid modifying original
        df_features = df.copy()
        
        # Apply transformations in sequence
        df_features = self._create_basic_features(df_features)
        df_features = self._add_statistical_features(df_features)
        df_features = self._create_competition_features(df_features)
        df_features = self._create_assortment_features(df_features)
        df_features = self._create_interaction_features(df_features)
        df_features = self._create_complement_features(df_features)
        df_features = self._create_advanced_features(df_features)
        
        print(f"Created {df_features.shape[1] - df.shape[1]} new features")
        
        # NEW: Validate stratification columns exist
        self._validate_stratification_columns(df_features)
        
        return df_features
    
    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform training data in one step."""
        return self.fit(train_df).transform(train_df, is_train=True)
    
    def _create_mrp_bins(self, train_df: pd.DataFrame):
        """NEW: Create MRP bins based on training data quantiles for balanced distribution."""
        print("Creating MRP bins based on training data distribution...")
        
        mrp_values = train_df['Item_MRP']
        
        # Create bins using quantiles for balanced distribution across strata
        quantiles = [0, 0.25, 0.5, 0.75, 1.0]
        bin_edges = mrp_values.quantile(quantiles).tolist()
        bin_edges = [30.0, 50.0, 120.0, 175.0, 270.0]
        # Ensure unique edges (in case of duplicates)
        bin_edges = sorted(list(set(bin_edges)))
        
        # If we still have very few unique values, fall back to equal-width bins
        if len(bin_edges) < 3:
            bin_edges = [mrp_values.min(), 
                        mrp_values.min() + (mrp_values.max() - mrp_values.min()) / 3,
                        mrp_values.min() + 2 * (mrp_values.max() - mrp_values.min()) / 3,
                        mrp_values.max()]
        
        # Create meaningful labels
        if len(bin_edges) == 5:  # Quartile-based
            bin_labels = ['Low_Price', 'Medium_Low', 'Medium_High', 'High_Price']
        elif len(bin_edges) == 4:  # Tertile-based
            bin_labels = ['Low_Price', 'Medium_Price', 'High_Price']
        else:  # Fallback to original approach
            bin_labels = [f'MRP_Bin_{i}' for i in range(len(bin_edges)-1)]
        
        self.mrp_bins_info = {
            'edges': bin_edges,
            'labels': bin_labels
        }
        
        print(f"Created {len(bin_labels)} MRP bins with edges: {[f'{x:.1f}' for x in bin_edges]}")
        print(f"Bin labels: {bin_labels}")
        
        # Test the binning on training data to show distribution
        test_bins = pd.cut(mrp_values, bins=bin_edges, labels=bin_labels, include_lowest=True)
        print("Training data MRP bin distribution:")
        print(test_bins.value_counts().sort_index())
    
    def _calculate_item_outlet_statistics(self, train_df: pd.DataFrame):
        """Calculate item-outlet combination statistics from training data."""
        # MODIFIED: Remove sales-based statistics for segmented modeling
        self.item_outlet_stats = train_df.groupby(['Item_Type', 'Outlet_Identifier'])[
            'Item_Outlet_Sales'
        ].agg(['count']).reset_index()
        self.item_outlet_stats.columns = [
            'Item_Type', 'Outlet_Identifier', 
            'Item_Type_Outlet_Count'
        ]
    
    def _create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic derived features."""
        # Outlet age calculation
        df['Outlet_Age'] = 2013 - df['Outlet_Establishment_Year']
        df['Outlet_Age_Squared'] = df['Outlet_Age'] ** 2
        
        # MODIFIED: Use fitted MRP bins instead of hardcoded bins
        df['Item_MRP_Bins'] = pd.cut(
            df['Item_MRP'], 
            bins=self.mrp_bins_info['edges'], 
            labels=self.mrp_bins_info['labels'],
            include_lowest=True
        )
        
        # Handle any NaN values (items outside training range)
        df['Item_MRP_Bins'] = df['Item_MRP_Bins'].fillna(self.mrp_bins_info['labels'][0])
        
        # Visibility efficiency metric
        df['Price_Per_Unit_Visibility'] = df['Item_MRP'] / (df['Item_Visibility'] + 0.001)
        
        return df
        
    def _calculate_outlet_statistics(self, train_df: pd.DataFrame):
        """Calculate outlet-level statistics from training data."""
        self.outlet_stats = train_df.groupby('Outlet_Identifier').agg({
            'Item_Visibility': 'sum'
        }).reset_index()
        self.outlet_stats.columns = [
            'Outlet_Identifier', 
            'Outlet_Total_Visibility'
        ]
        
    def _calculate_outlet_characteristics(self, train_df: pd.DataFrame):
        """Calculate outlet characteristic mappings."""
        # Premium ratio per outlet
        self.outlet_premium_ratios = train_df.groupby('Outlet_Identifier').apply(
            lambda x: (x['Item_MRP'] > 150).mean()
        ).reset_index()
        self.outlet_premium_ratios.columns = ['Outlet_Identifier', 'Outlet_Premium_Ratio']
        
        # Outlet location counts
        self.outlet_location_counts = train_df.groupby(
            ['Outlet_Location_Type', 'Outlet_Type']
        ).size().reset_index(name='Outlet_Type_Count_In_Location')
        
    def _define_complement_groups(self):
        """Define complementary product groups based on domain knowledge."""
        self.complement_groups = {
            'Breakfast': ['Dairy', 'Breads', 'Breakfast'],
            'Snacks': ['Snack Foods', 'Soft Drinks'],
            'Household': ['Household', 'Health and Hygiene'],
            'Cooking': ['Fruits and Vegetables', 'Meat', 'Seafood']
        }
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add pre-calculated statistical features."""
        # Merge item-outlet statistics
        df = df.merge(self.item_outlet_stats, 
                     on=['Item_Type', 'Outlet_Identifier'], 
                     how='left')
        
        # Handle missing values for new combinations
        df['Item_Type_Outlet_Count'].fillna(1, inplace=True)
        
        # Merge outlet statistics
        df = df.merge(self.outlet_stats, on='Outlet_Identifier', how='left')
        visibility_mean = self.outlet_stats['Outlet_Total_Visibility'].mean()
        df['Outlet_Total_Visibility'].fillna(visibility_mean, inplace=True)
        
        return df
    
    def _create_competition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create competition and market positioning features."""
        # Items in same category - direct competition measure
        df['Items_In_Same_Category'] = df.groupby(
            ['Outlet_Identifier', 'Item_Type']
        )['Item_Identifier'].transform('count')
        
        # Category visibility share - shelf space competition
        category_visibility = df.groupby(
            ['Outlet_Identifier', 'Item_Type']
        )['Item_Visibility'].transform('sum')
        df['Category_Visibility_Share'] = df['Item_Visibility'] / (category_visibility + 0.001)
        
        # Price competition features
        df['Item_Price_Rank_In_Category'] = df.groupby(
            ['Outlet_Identifier', 'Item_Type']
        )['Item_MRP'].rank(method='dense')
        
        # Calculate cheaper alternatives efficiently
        df = self._calculate_cheaper_alternatives(df)
        
        # Relative positioning metrics
        category_avg_price = df.groupby(
            ['Outlet_Identifier', 'Item_Type']
        )['Item_MRP'].transform('mean')
        df['Price_Ratio_To_Category_Avg'] = df['Item_MRP'] / category_avg_price
        
        category_avg_visibility = df.groupby(
            ['Outlet_Identifier', 'Item_Type']
        )['Item_Visibility'].transform('mean')
        df['Visibility_Ratio_To_Category_Avg'] = df['Item_Visibility'] / (category_avg_visibility + 0.001)
        
        return df
    
    def _calculate_cheaper_alternatives(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate number of cheaper alternatives in same category."""
        # Group by outlet and item type
        grouped = df.groupby(['Outlet_Identifier', 'Item_Type'])
        
        # For each group, count cheaper items
        cheaper_counts = []
        for (outlet, item_type), group in grouped:
            prices = group['Item_MRP'].values
            counts = np.array([(prices < price).sum() for price in prices])
            cheaper_counts.extend(counts)
        
        df['Cheaper_Alternatives_Count'] = cheaper_counts
        return df
    
    def _create_assortment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create outlet assortment and diversity features."""
        # Product diversity
        df['Unique_Categories_In_Outlet'] = df.groupby(
            'Outlet_Identifier'
        )['Item_Type'].transform('nunique')
        
        # Merge pre-calculated outlet characteristics
        df = df.merge(self.outlet_premium_ratios, on='Outlet_Identifier', how='left')
        df['Outlet_Premium_Ratio'].fillna(
            self.outlet_premium_ratios['Outlet_Premium_Ratio'].mean(), 
            inplace=True
        )
        
        # Low fat ratio - health consciousness indicator
        df['Outlet_Low_Fat_Ratio'] = df.groupby(
            'Outlet_Identifier'
        )['Low_Fat_Flag'].transform('mean')
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between different attributes."""
        # Strategic location-type combinations
        df['Is_Tier1_Supermarket'] = (
            (df['Outlet_Location_Type'] == 'Tier 1') & 
            (df['Outlet_Type'].str.contains('Supermarket'))
        ).astype(int)
        
        # MODIFIED: Use dynamic MRP bins instead of hardcoded 'Premium'
        # Check if any bin label suggests premium/high-end
        premium_bins = [label for label in self.mrp_bins_info['labels'] 
                       if 'high' in label.lower() or 'premium' in label.lower()]
        
        if premium_bins:
            premium_condition = df['Item_MRP_Bins'].isin(premium_bins)
        else:
            # Fallback: use the highest MRP bin
            premium_condition = df['Item_MRP_Bins'] == self.mrp_bins_info['labels'][-1]
        
        df['Large_Outlet_Premium_Item'] = (
            (df['Outlet_Size'] == 'Large') & premium_condition
        ).astype(int)
        
        # Category-outlet type mismatch
        drinks_categories = ['Soft Drinks', 'Dairy']
        df['Drinks_In_Grocery'] = (
            (df['Item_Type'].isin(drinks_categories)) & 
            (df['Outlet_Type'] == 'Grocery Store')
        ).astype(int)
        
        return df
    
    def _create_complement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on complementary product relationships."""
        # Map items to complement groups
        item_to_group = {}
        for group, items in self.complement_groups.items():
            for item in items:
                item_to_group[item] = group
        
        df['Complement_Group'] = df['Item_Type'].map(item_to_group).fillna('Other')
        
        # Complement group statistics
        df['Complement_Group_Items_Count'] = df.groupby(
            ['Outlet_Identifier', 'Complement_Group']
        )['Item_Identifier'].transform('count')
        
        df['Complement_Group_Visibility'] = df.groupby(
            ['Outlet_Identifier', 'Complement_Group']
        )['Item_Visibility'].transform('sum')
        
        return df
    
    def _create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced domain-specific features."""
        # Weight to price ratio - value perception
        df['Item_Weight_To_MRP_Ratio'] = df['Item_Weight'] / (df['Item_MRP'] + 1)
        
        # Marketing efficiency metrics
        df['Visibility_Per_Dollar'] = df['Item_Visibility'] / (df['Item_MRP'] + 1)
        
        # Above average visibility flag
        df['Above_Avg_Visibility_In_Category'] = (
            df['Item_Visibility'] > 
            df.groupby(['Outlet_Identifier', 'Item_Type'])['Item_Visibility'].transform('mean')
        ).astype(int)
        
        # Market penetration
        df = df.merge(
            self.outlet_location_counts, 
            on=['Outlet_Location_Type', 'Outlet_Type'], 
            how='left'
        )
        
        # Enhanced cannibalization features
        df = self._create_cannibalization_features(df)
        
        return df
    
    def _create_cannibalization_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced cannibalization and substitution features."""
        # Price gap to nearest competitor
        def calculate_min_price_gap(group):
            gaps = []
            prices = group['Item_MRP'].values
            for i, price in enumerate(prices):
                other_prices = np.delete(prices, i)
                if len(other_prices) > 0:
                    min_gap = np.min(np.abs(other_prices - price))
                else:
                    min_gap = 0
                gaps.append(min_gap)
            return pd.Series(gaps, index=group.index)
        
        price_gaps = df.groupby(['Outlet_Identifier', 'Item_Type']).apply(
            calculate_min_price_gap
        )
        df['Price_Gap_To_Nearest_Competitor'] = price_gaps.values
        
        # Substitution intensity - weighted by price similarity
        df['Substitution_Intensity'] = df.apply(
            lambda row: row['Items_In_Same_Category'] * 
            np.exp(-row['Price_Gap_To_Nearest_Competitor'] / 50),
            axis=1
        )
        
        # Price band competition - items within 10% price range
        def count_price_band_competitors(group):
            counts = []
            for _, row in group.iterrows():
                price = row['Item_MRP']
                band_count = ((group['Item_MRP'] >= price * 0.9) & 
                            (group['Item_MRP'] <= price * 1.1)).sum() - 1
                counts.append(max(0, band_count))
            return pd.Series(counts, index=group.index)
        
        band_counts = df.groupby(['Outlet_Identifier', 'Item_Type']).apply(
            count_price_band_competitors
        )
        df['Price_Band_Competitors'] = band_counts.values
        
        return df
    
    def _validate_stratification_columns(self, df: pd.DataFrame):
        """NEW: Validate that required stratification columns exist and are properly formatted."""
        required_cols = ['Outlet_Type', 'Outlet_Identifier', 'Item_MRP_Bins']
        
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required stratification column: {col}")
        
        # Check for reasonable number of unique values
        print(f"Stratification column validation:")
        for col in required_cols:
            unique_count = df[col].nunique()
            print(f"  {col}: {unique_count} unique values")
            
            if col == 'Item_MRP_Bins':
                print(f"    MRP bin distribution: {df[col].value_counts().to_dict()}")
    
    def get_feature_names(self) -> List[str]:
        """Get list of all created feature names."""
        basic_features = [
            'Outlet_Age', 'Outlet_Age_Squared', 'Item_MRP_Bins', 
            'Price_Per_Unit_Visibility'
        ]
        
        statistical_features = [
            'Item_Type_Outlet_Count',
            'Outlet_Total_Visibility'
        ]
        
        competition_features = [
            'Items_In_Same_Category', 'Category_Visibility_Share',
            'Item_Price_Rank_In_Category', 'Cheaper_Alternatives_Count',
            'Price_Ratio_To_Category_Avg', 'Visibility_Ratio_To_Category_Avg'
        ]
        
        assortment_features = [
            'Unique_Categories_In_Outlet', 'Outlet_Premium_Ratio',
            'Outlet_Low_Fat_Ratio'
        ]
        
        interaction_features = [
            'Is_Tier1_Supermarket', 'Large_Outlet_Premium_Item',
            'Drinks_In_Grocery'
        ]
        
        complement_features = [
            'Complement_Group', 'Complement_Group_Items_Count',
            'Complement_Group_Visibility'
        ]
        
        advanced_features = [
            'Item_Weight_To_MRP_Ratio', 'Visibility_Per_Dollar',
            'Above_Avg_Visibility_In_Category', 'Outlet_Type_Count_In_Location',
            'Price_Gap_To_Nearest_Competitor', 'Substitution_Intensity',
            'Price_Band_Competitors'
        ]
        
        all_features = (basic_features + statistical_features + 
                       competition_features + assortment_features + 
                       interaction_features + complement_features + 
                       advanced_features)
        
        return all_features
    
    def get_segment_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        NEW: Extract segment information for model training.
        Add this method to the FeatureEngineer class.
        
        Args:
            df: Processed dataset
            
        Returns:
            DataFrame with segment columns
        """
        if not self._is_fitted:
            raise ValueError("FeatureEngineer must be fitted before extracting segment info")
        
        # Ensure MRP bins are created if not already present
        if 'Item_MRP_Bins' not in df.columns:
            df = df.copy()
            df['Item_MRP_Bins'] = pd.cut(
                df['Item_MRP'], 
                bins=self.mrp_bins_info['edges'], 
                labels=self.mrp_bins_info['labels'],
                include_lowest=True
            )
            df['Item_MRP_Bins'] = df['Item_MRP_Bins'].fillna(self.mrp_bins_info['labels'][0])
        
        segment_info = pd.DataFrame({
            'Outlet_Type': df['Outlet_Type'],
            'Outlet_Identifier': df['Outlet_Identifier'],
            'Item_MRP_Bins': df['Item_MRP_Bins']
        })
        
        return segment_info


def create_features(train_path: str, test_path: str, 
                   output_train_path: str, output_test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main function to create features for train and test sets.
    MODIFIED for segmented modeling compatibility.
    
    Args:
        train_path: Path to cleaned training data
        test_path: Path to cleaned test data
        output_train_path: Path to save featured training data
        output_test_path: Path to save featured test data
        
    Returns:
        Tuple of (featured_train, featured_test) DataFrames
    """
    print("=== BigMart Feature Engineering Pipeline ===")
    
    # Load data
    print("Loading cleaned data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Fit on train and transform both
    print("\nCreating features...")
    train_features = engineer.fit_transform(train_df)
    
    # MODIFIED: Handle target variable properly for stratified modeling
    if 'Item_Outlet_Sales' in train_df.columns:
        # Keep original target for stratified modeling (no log transformation here)
        train_features['Item_Outlet_Sales'] = train_df['Item_Outlet_Sales']
    
    # Handle log_sales if it exists (from data preprocessing)
    if 'log_sales' in train_df.columns:
        train_features['log_sales'] = train_df['log_sales']
    
    test_features = engineer.transform(test_df, is_train=False)
    
    # Save featured datasets
    train_features.to_csv(output_train_path, index=False)
    test_features.to_csv(output_test_path, index=False)
    
    print(f"\nFinal shapes - Train: {train_features.shape}, Test: {test_features.shape}")
    print(f"Created {len(engineer.get_feature_names())} new features")
    
    # NEW: Show stratification readiness
    print(f"\nStratification readiness check:")
    print(f"✓ Outlet Types: {train_features['Outlet_Type'].nunique()} unique")
    print(f"✓ Outlet Identifiers: {train_features['Outlet_Identifier'].nunique()} unique")
    print(f"✓ MRP Bins: {train_features['Item_MRP_Bins'].nunique()} unique")
    
    print("\nFeature engineering completed successfully!")
    
    # Display feature importance hints
    if 'Item_Outlet_Sales' in train_features.columns:
        print("\nTop correlated features with sales:")
        correlations = train_features.select_dtypes(include=[np.number]).corr()['Item_Outlet_Sales']
        print(correlations.abs().sort_values(ascending=False).head(10))
    
    return train_features, test_features

if __name__ == "__main__":
    
    data_path = "/Users/whysocurious/Documents/MLDSAIProjects/BigMartSalesPred_Hackathon/data"
    
    train_features, test_features = create_features(
        train_path=f"{data_path}/processed/train_cleaned.csv",
        test_path=f"{data_path}/processed/test_cleaned.csv",
        output_train_path=f"{data_path}/processed/train_features.csv",
        output_test_path=f"{data_path}/processed/test_features.csv"
    )