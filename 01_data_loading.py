"""
Script 1: Data Loading and Initial Validation
==============================================
This script loads your Excel data and performs basic checks.

Think of this as the equivalent of:
- SAS: PROC IMPORT + PROC CONTENTS
- SQL: SELECT * FROM table LIMIT 10
"""

# Import libraries (like loading SAS procedures)
import pandas as pd  # pd is just a nickname (alias) for pandas
import numpy as np

def load_campaign_data(file_path):
    """
    Load the campaign data from Excel file.
    
    Parameters:
    -----------
    file_path : str
        Path to your Excel file (e.g., 'data/campaign_data.xlsx')
    
    Returns:
    --------
    df : DataFrame
        Your data loaded into a pandas DataFrame
    """
    
    # Read Excel file - like opening a table in SAS or SQL
    # This reads the first sheet by default
    df = pd.read_excel(file_path, sheet_name='Campaign Data')
    
    print("✓ Data loaded successfully!")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    
    return df


def initial_data_validation(df):
    """
    Perform initial data quality checks.
    This is like running PROC FREQ and PROC MEANS in SAS.
    """
    
    print("\n" + "="*60)
    print("INITIAL DATA VALIDATION")
    print("="*60)
    
    # 1. Show first few rows (like SELECT TOP 5 in SQL)
    print("\n📊 First 5 rows:")
    print(df.head())
    
    # 2. Show data types (like PROC CONTENTS in SAS)
    print("\n📋 Column Data Types:")
    print(df.dtypes)
    
    # 3. Check for missing values (like counting NULLs in SQL)
    print("\n🔍 Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values found!")
    
    # 4. Basic statistics (like PROC MEANS in SAS)
    print("\n📈 Summary Statistics for Numeric Columns:")
    print(df.describe())
    
    # 5. Check unique values for categorical columns
    print("\n🏷️  Unique Values in Categorical Columns:")
    categorical_cols = df.select_dtypes(include=['object', 'bool']).columns
    for col in categorical_cols:
        unique_count = df[col].nunique()
        print(f"  {col}: {unique_count} unique values")
        if unique_count <= 10:  # Show values if there aren't too many
            print(f"    Values: {df[col].unique()}")
    
    # 6. Check test/control distribution
    if 'test_control' in df.columns:
        print("\n📊 Test/Control Distribution:")
        print(df['test_control'].value_counts())
        print(f"\nPercentages:")
        print(df['test_control'].value_counts(normalize=True) * 100)
    
    return df


# Main execution block
if __name__ == "__main__":
    # This block only runs when you execute this file directly
    # (not when you import it into another script)
    
    # Set your file path
    file_path = 'data/campaign_data.xlsx'
    
    # Load data
    df = load_campaign_data(file_path)
    
    # Validate data
    df = initial_data_validation(df)
    
    # Save the loaded data for next steps (optional but recommended)
    # This creates a checkpoint you can return to
    df.to_csv('output/01_loaded_data.csv', index=False)
    print("\n✓ Data saved to 'output/01_loaded_data.csv'")
