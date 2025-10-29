"""
Script 2: Data Preprocessing
============================
This script prepares your data for clustering analysis.

Key tasks:
1. Handle categorical variables (one-hot encoding)
2. Identify features for clustering vs. target variables
3. Handle missing values if any
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def identify_column_types(df):
    """
    Identify which columns are features (for clustering) vs targets vs IDs.
    
    This is like organizing your SAS variables into different categories.
    """
    
    # Define your columns based on the project
    # ADJUST THESE BASED ON YOUR ACTUAL COLUMN NAMES
    
    # ID columns (customer identifiers - don't use for analysis)
    id_columns = ['customer_id', 'account_number']  # Adjust as needed
    
    # Target variables (outcomes we're measuring)
    target_columns = [
        'recovery_percent',
        'rpc_flag',  # 1 if customer had RPC, 0 otherwise
        'payment_flag'  # 1 if customer made payment, 0 otherwise
    ]
    
    # Feature columns (what we use to create segments)
    feature_columns = [
        'test_control',
        'collateral_status',
        'nsol_flag',
        'balance_bin',
        'last_rpc_months',
        'last_payment_months',
        'time_since_chargeoff_months'
    ]
    
    # Find which ones actually exist in your data
    available_features = [col for col in feature_columns if col in df.columns]
    available_targets = [col for col in target_columns if col in df.columns]
    available_ids = [col for col in id_columns if col in df.columns]
    
    print("\n📋 Column Classification:")
    print(f"  ID columns: {available_ids}")
    print(f"  Feature columns: {available_features}")
    print(f"  Target columns: {available_targets}")
    
    return {
        'id': available_ids,
        'features': available_features,
        'targets': available_targets
    }


def handle_missing_values(df, strategy='drop'):
    """
    Handle missing values in the dataset.
    
    Parameters:
    -----------
    strategy : str
        'drop' - remove rows with missing values
        'mode' - fill with most common value (for categorical)
        'median' - fill with median (for numeric)
    """
    
    missing_before = df.isnull().sum().sum()
    
    if missing_before == 0:
        print("\n✓ No missing values found!")
        return df
    
    print(f"\n⚠️  Found {missing_before} missing values")
    print("\nMissing values by column:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    
    if strategy == 'drop':
        df_clean = df.dropna()
        rows_removed = len(df) - len(df_clean)
        print(f"\n✓ Removed {rows_removed} rows with missing values")
        return df_clean
    
    elif strategy == 'mode':
        # Fill with most common value
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                mode_value = df[col].mode()[0]
                df[col].fillna(mode_value, inplace=True)
                print(f"  Filled {col} with mode: {mode_value}")
        return df
    
    return df


def create_one_hot_encoding(df, feature_columns):
    """
    Convert categorical variables to one-hot encoded format.
    
    This is like creating dummy variables in SAS:
    In SAS you might write: CLASS variable_name / PARAM=GLM;
    In Python, we use pd.get_dummies()
    
    Parameters:
    -----------
    df : DataFrame
        Your dataset
    feature_columns : list
        List of column names to encode
    
    Returns:
    --------
    df_encoded : DataFrame
        Dataset with one-hot encoded features
    encoding_map : dict
        Dictionary showing which new columns came from which original columns
    """
    
    print("\n" + "="*60)
    print("ONE-HOT ENCODING")
    print("="*60)
    
    # Separate features to encode from the rest of the data
    df_features = df[feature_columns].copy()
    df_other = df.drop(columns=feature_columns)
    
    # Perform one-hot encoding
    # drop_first=False means keep all categories (we want all for clustering)
    df_encoded = pd.get_dummies(df_features, drop_first=False, dtype=int)
    
    # Combine back with other columns
    df_final = pd.concat([df_other, df_encoded], axis=1)
    
    # Create a mapping to track which columns came from where
    encoding_map = {}
    for original_col in feature_columns:
        # Find all new columns that start with the original column name
        new_cols = [col for col in df_encoded.columns if col.startswith(f"{original_col}_")]
        encoding_map[original_col] = new_cols
    
    print("\n📊 Encoding Summary:")
    for original, new_cols in encoding_map.items():
        print(f"\n  {original} →")
        for new_col in new_cols:
            print(f"    • {new_col}")
    
    print(f"\n✓ Original features: {len(feature_columns)}")
    print(f"✓ Encoded features: {len(df_encoded.columns)}")
    
    return df_final, encoding_map


def prepare_clustering_data(df, column_info, target_variable):
    """
    Prepare the final dataset for clustering.
    
    Parameters:
    -----------
    df : DataFrame
        Your full dataset with encoded features
    column_info : dict
        Dictionary with 'id', 'features', and 'targets' keys
    target_variable : str
        Which target variable we're analyzing (e.g., 'recovery_percent')
    
    Returns:
    --------
    X : DataFrame
        Features for clustering (one-hot encoded)
    y : Series
        Target variable values
    customer_ids : Series
        Customer identifiers
    """
    
    print(f"\n🎯 Preparing data for target: {target_variable}")
    
    # Get encoded feature columns (all columns with underscores from encoding)
    # This is a bit of pattern matching to find the encoded columns
    all_columns = df.columns.tolist()
    encoded_features = [col for col in all_columns 
                       if any(feat in col for feat in column_info['features'])]
    
    # Filter to only rows where target variable is not null
    df_clean = df[df[target_variable].notna()].copy()
    
    print(f"  Rows with valid target: {len(df_clean)}")
    print(f"  Features for clustering: {len(encoded_features)}")
    
    # X = features, y = target, ids = customer identifiers
    X = df_clean[encoded_features]
    y = df_clean[target_variable]
    customer_ids = df_clean[column_info['id'][0]] if column_info['id'] else df_clean.index
    
    print(f"  Final X shape: {X.shape}")
    print(f"  Final y shape: {y.shape}")
    
    return X, y, customer_ids


# Main execution
if __name__ == "__main__":
    
    print("="*60)
    print("DATA PREPROCESSING PIPELINE")
    print("="*60)
    
    # Load the data from previous step
    df = pd.read_csv('output/01_loaded_data.csv')
    print(f"✓ Loaded data: {df.shape}")
    
    # Step 1: Identify column types
    column_info = identify_column_types(df)
    
    # Step 2: Handle missing values
    df = handle_missing_values(df, strategy='drop')
    
    # Step 3: One-hot encode categorical features
    df_encoded, encoding_map = create_one_hot_encoding(df, column_info['features'])
    
    # Step 4: Save preprocessed data
    df_encoded.to_csv('output/02_preprocessed_data.csv', index=False)
    print("\n✓ Preprocessed data saved to 'output/02_preprocessed_data.csv'")
    
    # Step 5: Save encoding map for reference
    import json
    with open('output/02_encoding_map.json', 'w') as f:
        json.dump(encoding_map, f, indent=2)
    print("✓ Encoding map saved to 'output/02_encoding_map.json'")
    
    # Step 6: Prepare data for each target (as examples)
    print("\n" + "="*60)
    print("PREPARING DATA FOR EACH TARGET VARIABLE")
    print("="*60)
    
    for target in column_info['targets']:
        if target in df_encoded.columns:
            X, y, ids = prepare_clustering_data(df_encoded, column_info, target)
            
            # Save prepared data for this target
            output_file = f'output/02_prepared_{target}.csv'
            prepared_df = pd.DataFrame(X)
            prepared_df['target'] = y.values
            prepared_df['customer_id'] = ids.values
            prepared_df.to_csv(output_file, index=False)
            print(f"  ✓ Saved prepared data for {target}")
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)
