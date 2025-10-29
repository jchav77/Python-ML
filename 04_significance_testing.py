"""
Script 4: Statistical Significance Testing
==========================================
This script tests whether test vs control differences are statistically significant
within each cluster.

Tests used:
- Two-proportion z-test: For binary outcomes (RPC flag, payment flag)
- Two-sample t-test: For continuous outcomes (recovery percent)
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns


def check_sample_size_requirements(test_data, control_data, metric_type='proportion'):
    """
    Check if sample sizes are adequate for statistical testing.
    
    Parameters:
    -----------
    test_data : Series or array
        Test group data
    control_data : Series or array
        Control group data
    metric_type : str
        'proportion' or 'continuous'
    
    Returns:
    --------
    is_adequate : bool
        Whether sample size requirements are met
    reason : str
        Explanation of the decision
    """
    
    n_test = len(test_data)
    n_control = len(control_data)
    
    if metric_type == 'proportion':
        # For proportions: need at least 10 successes and 10 failures in each group
        test_successes = test_data.sum()
        test_failures = n_test - test_successes
        control_successes = control_data.sum()
        control_failures = n_control - control_successes
        
        if min(test_successes, test_failures, control_successes, control_failures) < 10:
            return False, "Not enough successes/failures in one or both groups (need ≥10 each)"
        
    elif metric_type == 'continuous':
        # For continuous: generally want at least 30 in each group
        # But can go lower if data is normally distributed
        if min(n_test, n_control) < 10:
            return False, "Sample size too small (need ≥10 in each group)"
        elif min(n_test, n_control) < 30:
            return True, f"Small sample (test={n_test}, control={n_control}), results should be interpreted cautiously"
    
    return True, f"Adequate sample size (test={n_test}, control={n_control})"


def two_proportion_z_test(test_data, control_data, alpha=0.10):
    """
    Perform two-proportion z-test.
    
    Use this for binary outcomes like:
    - Did customer have RPC? (yes=1, no=0)
    - Did customer make payment? (yes=1, no=0)
    
    Null hypothesis: proportion_test = proportion_control
    Alternative hypothesis: proportion_test ≠ proportion_control
    
    Parameters:
    -----------
    test_data : Series or array
        Binary data for test group (0s and 1s)
    control_data : Series or array
        Binary data for control group (0s and 1s)
    alpha : float
        Significance level (0.10 = 90% confidence, 0.05 = 95% confidence)
    
    Returns:
    --------
    results : dict
        Dictionary with test results
    """
    
    # Calculate proportions
    n_test = len(test_data)
    n_control = len(control_data)
    
    p_test = test_data.mean()
    p_control = control_data.mean()
    
    # Calculate pooled proportion (used under null hypothesis)
    p_pooled = (test_data.sum() + control_data.sum()) / (n_test + n_control)
    
    # Calculate standard error
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_test + 1/n_control))
    
    # Calculate z-statistic
    z_stat = (p_test - p_control) / se if se > 0 else 0
    
    # Calculate p-value (two-tailed test)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    # Determine significance
    is_significant = p_value < alpha
    
    # Calculate confidence interval for difference
    se_diff = np.sqrt(p_test*(1-p_test)/n_test + p_control*(1-p_control)/n_control)
    z_critical = stats.norm.ppf(1 - alpha/2)
    ci_lower = (p_test - p_control) - z_critical * se_diff
    ci_upper = (p_test - p_control) + z_critical * se_diff
    
    return {
        'test_proportion': p_test,
        'control_proportion': p_control,
        'difference': p_test - p_control,
        'difference_pct': (p_test - p_control) * 100,
        'z_statistic': z_stat,
        'p_value': p_value,
        'is_significant': is_significant,
        'confidence_level': (1-alpha)*100,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_test': n_test,
        'n_control': n_control
    }


def two_sample_t_test(test_data, control_data, alpha=0.10):
    """
    Perform two-sample t-test.
    
    Use this for continuous outcomes like:
    - Average recovery percentage
    - Average payment amount
    
    Null hypothesis: mean_test = mean_control
    Alternative hypothesis: mean_test ≠ mean_control
    
    Parameters:
    -----------
    test_data : Series or array
        Continuous data for test group
    control_data : Series or array
        Continuous data for control group
    alpha : float
        Significance level
    
    Returns:
    --------
    results : dict
        Dictionary with test results
    """
    
    # Calculate means and standard deviations
    mean_test = test_data.mean()
    mean_control = control_data.mean()
    std_test = test_data.std()
    std_control = control_data.std()
    n_test = len(test_data)
    n_control = len(control_data)
    
    # Perform t-test (using Welch's t-test which doesn't assume equal variances)
    t_stat, p_value = ttest_ind(test_data, control_data, equal_var=False)
    
    # Determine significance
    is_significant = p_value < alpha
    
    # Calculate confidence interval for difference
    se_diff = np.sqrt(std_test**2/n_test + std_control**2/n_control)
    df = n_test + n_control - 2  # Simplified degrees of freedom
    t_critical = stats.t.ppf(1 - alpha/2, df)
    ci_lower = (mean_test - mean_control) - t_critical * se_diff
    ci_upper = (mean_test - mean_control) + t_critical * se_diff
    
    return {
        'test_mean': mean_test,
        'control_mean': mean_control,
        'test_std': std_test,
        'control_std': std_control,
        'difference': mean_test - mean_control,
        't_statistic': t_stat,
        'p_value': p_value,
        'is_significant': is_significant,
        'confidence_level': (1-alpha)*100,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_test': n_test,
        'n_control': n_control
    }


def test_cluster_significance(df, cluster_id, target_variable, metric_type, alpha=0.10):
    """
    Test significance for a specific cluster.
    
    Parameters:
    -----------
    df : DataFrame
        Data with cluster labels and test/control flags
    cluster_id : int
        Which cluster to test
    target_variable : str
        Name of target variable column
    metric_type : str
        'proportion' or 'continuous'
    alpha : float
        Significance level
    
    Returns:
    --------
    results : dict
        Test results, or None if sample size inadequate
    """
    
    # Filter to this cluster
    cluster_data = df[df['cluster'] == cluster_id].copy()
    
    # Split into test and control
    # Assuming you have a column indicating test/control
    # Adjust column name as needed
    test_data = cluster_data[cluster_data['test_control'] == 'test'][target_variable]
    control_data = cluster_data[cluster_data['test_control'] == 'control'][target_variable]
    
    # Check sample size
    is_adequate, reason = check_sample_size_requirements(test_data, control_data, metric_type)
    
    if not is_adequate:
        return {
            'cluster': cluster_id,
            'n_test': len(test_data),
            'n_control': len(control_data),
            'is_testable': False,
            'reason': reason
        }
    
    # Perform appropriate test
    if metric_type == 'proportion':
        test_results = two_proportion_z_test(test_data, control_data, alpha)
    else:  # continuous
        test_results = two_sample_t_test(test_data, control_data, alpha)
    
    test_results['cluster'] = cluster_id
    test_results['is_testable'] = True
    test_results['reason'] = reason
    
    return test_results


def run_all_cluster_tests(df, target_variable, metric_type, alpha=0.10):
    """
    Run significance tests for all clusters.
    
    Parameters:
    -----------
    df : DataFrame
        Data with cluster labels
    target_variable : str
        Name of target variable
    metric_type : str
        'proportion' or 'continuous'
    alpha : float
        Significance level
    
    Returns:
    --------
    results_df : DataFrame
        Summary of all test results
    """
    
    print("\n" + "="*60)
    print(f"TESTING SIGNIFICANCE FOR ALL CLUSTERS")
    print(f"Target: {target_variable} | Type: {metric_type}")
    print(f"Confidence Level: {(1-alpha)*100}%")
    print("="*60)
    
    clusters = sorted(df['cluster'].unique())
    results_list = []
    
    for cluster_id in clusters:
        print(f"\nTesting Cluster {cluster_id}...")
        results = test_cluster_significance(df, cluster_id, target_variable, metric_type, alpha)
        results_list.append(results)
        
        if results['is_testable']:
            if metric_type == 'proportion':
                print(f"  Test: {results['test_proportion']:.1%} | Control: {results['control_proportion']:.1%}")
                print(f"  Difference: {results['difference_pct']:.2f} percentage points")
            else:
                print(f"  Test: {results['test_mean']:.4f} | Control: {results['control_mean']:.4f}")
                print(f"  Difference: {results['difference']:.4f}")
            
            print(f"  P-value: {results['p_value']:.4f}")
            print(f"  Significant: {'YES ✓' if results['is_significant'] else 'NO'}")
        else:
            print(f"  ⚠️  Cannot test: {results['reason']}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results_list)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    testable = results_df['is_testable'].sum()
    significant = results_df[results_df['is_testable'] == True]['is_significant'].sum()
    
    print(f"Total clusters: {len(results_df)}")
    print(f"Testable clusters: {testable}")
    print(f"Significant results: {significant}")
    print(f"Success rate: {significant/testable*100:.1f}% of testable clusters" if testable > 0 else "N/A")
    
    return results_df


def create_significance_visualization(results_df, target_variable, output_path):
    """
    Create a visualization of test results across clusters.
    """
    
    # Filter to testable clusters
    testable = results_df[results_df['is_testable'] == True].copy()
    
    if len(testable) == 0:
        print("⚠️  No testable clusters to visualize")
        return
    
    # Determine metric type from columns
    if 'test_proportion' in testable.columns:
        metric_type = 'proportion'
        y_col = 'difference_pct'
        y_label = 'Difference (percentage points)'
    else:
        metric_type = 'continuous'
        y_col = 'difference'
        y_label = 'Difference in Mean'
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Effect size by cluster
    colors = ['green' if sig else 'red' for sig in testable['is_significant']]
    ax1.bar(testable['cluster'], testable[y_col], color=colors, alpha=0.6)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('Cluster')
    ax1.set_ylabel(y_label)
    ax1.set_title(f'Effect Size by Cluster\n{target_variable}')
    ax1.grid(True, alpha=0.3)
    
    # Add significance markers
    for idx, row in testable.iterrows():
        marker = '**' if row['p_value'] < 0.05 else '*' if row['p_value'] < 0.10 else ''
        if marker:
            ax1.text(row['cluster'], row[y_col], marker, 
                    ha='center', va='bottom' if row[y_col] > 0 else 'top')
    
    # Plot 2: P-values by cluster
    colors = ['green' if sig else 'red' for sig in testable['is_significant']]
    ax2.bar(testable['cluster'], testable['p_value'], color=colors, alpha=0.6)
    ax2.axhline(y=0.10, color='orange', linestyle='--', label='p=0.10 (90% conf)')
    ax2.axhline(y=0.05, color='red', linestyle='--', label='p=0.05 (95% conf)')
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('P-value')
    ax2.set_title(f'P-values by Cluster\n{target_variable}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {output_path}")
    plt.close()


# Main execution
if __name__ == "__main__":
    
    # Configuration
    TARGET_VARIABLE = 'recovery_percent'  # Change as needed
    METRIC_TYPE = 'continuous'  # 'proportion' or 'continuous'
    ALPHA = 0.10  # Significance level (0.10 = 90% confidence)
    
    print("="*60)
    print(f"STATISTICAL SIGNIFICANCE ANALYSIS")
    print(f"Target: {TARGET_VARIABLE}")
    print("="*60)
    
    # Load clustered data
    input_file = f'output/03_clustered_{TARGET_VARIABLE}.csv'
    df = pd.read_csv(input_file)
    
    # Make sure we have the test_control column
    # You may need to merge this from your original data
    # For now, assuming it's already in the clustered data
    
    print(f"✓ Loaded {len(df)} customers")
    print(f"  Clusters: {df['cluster'].nunique()}")
    
    # Run significance tests
    results_df = run_all_cluster_tests(df, 'target', METRIC_TYPE, ALPHA)
    
    # Save results
    output_file = f'output/04_significance_results_{TARGET_VARIABLE}.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to {output_file}")
    
    # Create visualization
    create_significance_visualization(
        results_df, 
        TARGET_VARIABLE,
        f'output/figures/04_significance_{TARGET_VARIABLE}.png'
    )
    
    # Filter to significant clusters only
    significant_clusters = results_df[
        (results_df['is_testable'] == True) & 
        (results_df['is_significant'] == True)
    ]
    
    if len(significant_clusters) > 0:
        print("\n" + "="*60)
        print("SIGNIFICANT CLUSTERS TO INVESTIGATE")
        print("="*60)
        print("\nThese are the clusters where test significantly differs from control:")
        print(significant_clusters[['cluster', 'difference', 'p_value', 'n_test', 'n_control']])
        
        significant_output = f'output/04_significant_clusters_{TARGET_VARIABLE}.csv'
        significant_clusters.to_csv(significant_output, index=False)
        print(f"\n✓ Significant clusters saved to {significant_output}")
    else:
        print("\n⚠️  No significant differences found in any cluster")
    
    print("\n" + "="*60)
    print("SIGNIFICANCE TESTING COMPLETE!")
    print("="*60)
