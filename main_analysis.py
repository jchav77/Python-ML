"""
MASTER ANALYSIS SCRIPT
======================
This script runs the complete analysis pipeline for all three target variables.

Usage: python main_analysis.py

This will:
1. Load and validate data
2. Preprocess features
3. Run clustering analysis for each target
4. Test statistical significance for each target
5. Generate summary report
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

# Import our custom modules
import sys
sys.path.append('.')

print("="*80)
print(" " * 20 + "CAMPAIGN SEGMENTATION ANALYSIS")
print("="*80)
print(f"\nAnalysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Configuration
DATA_FILE = 'data/campaign_data.xlsx'
TARGET_VARIABLES = {
    'recovery_percent': 'continuous',
    'rpc_flag': 'proportion',
    'payment_flag': 'proportion'
}
SIGNIFICANCE_LEVEL = 0.10  # 90% confidence

# Create output directories if they don't exist
os.makedirs('output/figures', exist_ok=True)
os.makedirs('output/results', exist_ok=True)

print("\n" + "="*80)
print("STEP 1: DATA LOADING")
print("="*80)

# Run data loading
exec(open('01_data_loading.py').read())

print("\n" + "="*80)
print("STEP 2: DATA PREPROCESSING")  
print("="*80)

# Run preprocessing
exec(open('02_preprocessing.py').read())

print("\n" + "="*80)
print("STEP 3: CLUSTERING & SIGNIFICANCE TESTING")
print("="*80)

# For each target variable
results_summary = []

for target_var, metric_type in TARGET_VARIABLES.items():
    print(f"\n{'='*80}")
    print(f"ANALYZING: {target_var} ({metric_type})")
    print(f"{'='*80}")
    
    try:
        # Check if prepared data exists
        input_file = f'output/02_prepared_{target_var}.csv'
        if not os.path.exists(input_file):
            print(f"⚠️  Skipping {target_var} - prepared data not found")
            continue
        
        # Run clustering
        print(f"\n🔍 Running clustering for {target_var}...")
        exec(f"""
TARGET_VARIABLE = '{target_var}'
exec(open('03_clustering.py').read())
""")
        
        # Run significance testing  
        print(f"\n📊 Running significance tests for {target_var}...")
        exec(f"""
TARGET_VARIABLE = '{target_var}'
METRIC_TYPE = '{metric_type}'
ALPHA = {SIGNIFICANCE_LEVEL}
exec(open('04_significance_testing.py').read())
""")
        
        # Load results
        sig_results = pd.read_csv(f'output/04_significance_results_{target_var}.csv')
        
        # Count significant clusters
        testable = sig_results['is_testable'].sum()
        significant = sig_results[sig_results['is_testable'] == True]['is_significant'].sum()
        
        results_summary.append({
            'target_variable': target_var,
            'metric_type': metric_type,
            'total_clusters': len(sig_results),
            'testable_clusters': testable,
            'significant_clusters': significant,
            'success_rate': f"{significant/testable*100:.1f}%" if testable > 0 else "N/A"
        })
        
        print(f"✓ {target_var} analysis complete")
        
    except Exception as e:
        print(f"❌ Error analyzing {target_var}: {str(e)}")
        results_summary.append({
            'target_variable': target_var,
            'metric_type': metric_type,
            'total_clusters': 0,
            'testable_clusters': 0,
            'significant_clusters': 0,
            'success_rate': 'ERROR'
        })

print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

summary_df = pd.DataFrame(results_summary)
print("\n", summary_df.to_string(index=False))

# Save summary
summary_df.to_csv('output/00_analysis_summary.csv', index=False)
print("\n✓ Summary saved to 'output/00_analysis_summary.csv'")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nNext steps:")
print("1. Review the summary above")
print("2. Check the 'output/figures/' folder for visualizations")
print("3. Review significant clusters in '04_significant_clusters_*.csv' files")
print("4. Dive deeper into cluster characteristics in '03_cluster_profiles_*.csv' files")
print("\n" + "="*80)
