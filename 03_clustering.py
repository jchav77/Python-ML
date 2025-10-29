"""
Script 3: K-means Clustering Analysis
=====================================
This script performs clustering analysis to find natural customer segments.

What we're doing:
1. Test different numbers of clusters (K = 2 to 15)
2. Use two methods to find optimal K:
   - Elbow Method: Look for diminishing returns
   - Silhouette Score: Measure cluster separation
3. Fit the final model with optimal K
4. Assign cluster labels to each customer
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for prettier plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def standardize_features(X):
    """
    Standardize features to have mean=0 and std=1.
    
    Why? K-means is sensitive to scale. If one feature ranges from 0-1
    and another from 0-1000, the algorithm will think the second is more important.
    
    This is like converting everything to z-scores in statistics.
    
    Parameters:
    -----------
    X : DataFrame
        Feature matrix
    
    Returns:
    --------
    X_scaled : array
        Standardized features
    scaler : StandardScaler
        The scaler object (save this to transform new data later)
    """
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("✓ Features standardized")
    print(f"  Original mean: {X.mean().mean():.2f}")
    print(f"  Scaled mean: {X_scaled.mean():.2f}")
    print(f"  Scaled std: {X_scaled.std():.2f}")
    
    return X_scaled, scaler


def find_optimal_clusters_elbow(X_scaled, max_k=15):
    """
    Use the Elbow Method to find optimal number of clusters.
    
    The "elbow" is where the line bends - diminishing returns kick in.
    
    Parameters:
    -----------
    X_scaled : array
        Standardized feature matrix
    max_k : int
        Maximum number of clusters to test
    
    Returns:
    --------
    inertias : list
        Within-cluster sum of squares for each K
    k_range : list
        Range of K values tested
    """
    
    print("\n" + "="*60)
    print("ELBOW METHOD - Finding Optimal K")
    print("="*60)
    
    inertias = []
    k_range = range(2, max_k + 1)
    
    print("\nTesting different cluster counts...")
    for k in k_range:
        # Fit K-means with k clusters
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        
        # Inertia = sum of squared distances to nearest cluster center
        # Lower is better (tighter clusters)
        inertias.append(kmeans.inertia_)
        print(f"  K={k:2d}: Inertia = {kmeans.inertia_:,.0f}")
    
    return inertias, list(k_range)


def plot_elbow_method(inertias, k_range, output_path):
    """
    Create elbow plot to visualize optimal K.
    """
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (K)', fontsize=12)
    plt.ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
    plt.title('Elbow Method - Finding Optimal K', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Annotate each point
    for k, inertia in zip(k_range, inertias):
        plt.annotate(f'{inertia:,.0f}', 
                    xy=(k, inertia), 
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Elbow plot saved to {output_path}")
    plt.close()


def find_optimal_clusters_silhouette(X_scaled, max_k=15):
    """
    Use Silhouette Score to find optimal number of clusters.
    
    Silhouette Score measures:
    - How similar each point is to its own cluster (cohesion)
    - How different each point is from other clusters (separation)
    
    Score ranges from -1 to 1:
    - Near +1: Point is far from other clusters (good!)
    - Near 0: Point is on the border between clusters (unclear)
    - Near -1: Point is in the wrong cluster (bad!)
    
    Parameters:
    -----------
    X_scaled : array
        Standardized feature matrix
    max_k : int
        Maximum number of clusters to test
    
    Returns:
    --------
    silhouette_scores : list
        Average silhouette score for each K
    k_range : list
        Range of K values tested
    """
    
    print("\n" + "="*60)
    print("SILHOUETTE METHOD - Finding Optimal K")
    print("="*60)
    
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    
    print("\nCalculating silhouette scores...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Calculate average silhouette score
        score = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores.append(score)
        print(f"  K={k:2d}: Silhouette Score = {score:.4f}")
    
    # Find K with highest score
    best_k = k_range[silhouette_scores.index(max(silhouette_scores))]
    print(f"\n✓ Best K by Silhouette Score: {best_k}")
    
    return silhouette_scores, list(k_range)


def plot_silhouette_method(silhouette_scores, k_range, output_path):
    """
    Create silhouette score plot.
    """
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (K)', fontsize=12)
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.title('Silhouette Method - Finding Optimal K', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Mark the best K
    best_k = k_range[silhouette_scores.index(max(silhouette_scores))]
    best_score = max(silhouette_scores)
    plt.axvline(best_k, color='red', linestyle='--', alpha=0.5, label=f'Best K = {best_k}')
    plt.plot(best_k, best_score, 'r*', markersize=20)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Silhouette plot saved to {output_path}")
    plt.close()


def fit_final_kmeans(X_scaled, optimal_k):
    """
    Fit the final K-means model with the chosen K.
    
    Parameters:
    -----------
    X_scaled : array
        Standardized feature matrix
    optimal_k : int
        Number of clusters to use
    
    Returns:
    --------
    kmeans : KMeans
        Fitted K-means model
    cluster_labels : array
        Cluster assignment for each customer
    """
    
    print("\n" + "="*60)
    print(f"FITTING FINAL MODEL with K={optimal_k}")
    print("="*60)
    
    # Fit the model
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Print cluster sizes
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print("\n📊 Cluster Sizes:")
    for cluster_id, count in zip(unique, counts):
        pct = (count / len(cluster_labels)) * 100
        print(f"  Cluster {cluster_id}: {count:,} customers ({pct:.1f}%)")
    
    return kmeans, cluster_labels


def analyze_cluster_characteristics(X, y, cluster_labels, feature_names):
    """
    Analyze what makes each cluster unique.
    
    This creates a profile of each cluster by showing:
    1. Average target variable value per cluster
    2. Most common features in each cluster
    
    Parameters:
    -----------
    X : DataFrame
        Original features (before standardization)
    y : Series
        Target variable
    cluster_labels : array
        Cluster assignments
    feature_names : list
        Names of features
    
    Returns:
    --------
    cluster_profiles : DataFrame
        Summary of each cluster
    """
    
    print("\n" + "="*60)
    print("CLUSTER CHARACTERISTICS")
    print("="*60)
    
    # Create a DataFrame with everything
    df_analysis = X.copy()
    df_analysis['cluster'] = cluster_labels
    df_analysis['target'] = y.values
    
    # Calculate statistics per cluster
    cluster_profiles = df_analysis.groupby('cluster').agg({
        'target': ['mean', 'std', 'count']
    }).round(4)
    
    print("\n📊 Target Variable by Cluster:")
    print(cluster_profiles)
    
    # For each cluster, find the most distinctive features
    print("\n🔍 Most Distinctive Features per Cluster:")
    print("(Features with highest % in this cluster vs others)")
    
    for cluster_id in range(len(cluster_profiles)):
        print(f"\n  Cluster {cluster_id}:")
        cluster_data = df_analysis[df_analysis['cluster'] == cluster_id]
        
        # For binary features, calculate the percentage that have each feature
        feature_pcts = cluster_data[feature_names].mean()
        
        # Find features that are most common in this cluster
        # (at least 50% have this feature)
        distinctive_features = feature_pcts[feature_pcts > 0.5].sort_values(ascending=False)
        
        if len(distinctive_features) > 0:
            for feat, pct in distinctive_features.head(10).items():
                print(f"    • {feat}: {pct*100:.1f}%")
        else:
            print("    • No highly distinctive features")
    
    return cluster_profiles, df_analysis


# Main execution
if __name__ == "__main__":
    
    # Choose which target variable to analyze
    # RUN THIS SCRIPT THREE TIMES, once for each target
    TARGET_VARIABLE = 'recovery_percent'  # Change to 'rpc_flag' or 'payment_flag'
    
    print("="*60)
    print(f"CLUSTERING ANALYSIS FOR: {TARGET_VARIABLE}")
    print("="*60)
    
    # Load prepared data
    input_file = f'output/02_prepared_{TARGET_VARIABLE}.csv'
    df = pd.read_csv(input_file)
    
    # Separate features and target
    X = df.drop(columns=['target', 'customer_id'])
    y = df['target']
    customer_ids = df['customer_id']
    
    print(f"\n✓ Loaded data: {X.shape[0]} customers, {X.shape[1]} features")
    
    # Step 1: Standardize features
    X_scaled, scaler = standardize_features(X)
    
    # Step 2: Find optimal K using Elbow Method
    inertias, k_range = find_optimal_clusters_elbow(X_scaled, max_k=15)
    plot_elbow_method(inertias, k_range, 
                     f'output/figures/03_elbow_{TARGET_VARIABLE}.png')
    
    # Step 3: Find optimal K using Silhouette Method
    silhouette_scores, k_range = find_optimal_clusters_silhouette(X_scaled, max_k=15)
    plot_silhouette_method(silhouette_scores, k_range,
                          f'output/figures/03_silhouette_{TARGET_VARIABLE}.png')
    
    # Step 4: Choose optimal K (you'll review the plots and decide)
    # For now, let's use the K with highest silhouette score
    optimal_k = k_range[silhouette_scores.index(max(silhouette_scores))]
    
    print("\n" + "="*60)
    print("DECISION POINT")
    print("="*60)
    print("\nReview the plots:")
    print("1. Elbow plot - look for the 'elbow' where improvement slows")
    print("2. Silhouette plot - look for the highest score")
    print(f"\nRecommended K (highest silhouette): {optimal_k}")
    print("\nIf you want to use a different K, modify optimal_k variable below.")
    print("="*60)
    
    # Fit final model
    kmeans, cluster_labels = fit_final_kmeans(X_scaled, optimal_k)
    
    # Analyze clusters
    cluster_profiles, df_analysis = analyze_cluster_characteristics(
        X, y, cluster_labels, X.columns.tolist()
    )
    
    # Save results
    df_analysis.to_csv(f'output/03_clustered_{TARGET_VARIABLE}.csv', index=False)
    print(f"\n✓ Clustered data saved to 'output/03_clustered_{TARGET_VARIABLE}.csv'")
    
    cluster_profiles.to_csv(f'output/03_cluster_profiles_{TARGET_VARIABLE}.csv')
    print(f"✓ Cluster profiles saved")
    
    print("\n" + "="*60)
    print("CLUSTERING COMPLETE!")
    print("="*60)
    print("\nNext step: Run statistical significance tests on each cluster")
