import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("HDBSCAN not installed. Install with: pip install hdbscan")

def find_optimal_eps(data, min_samples=5, percentile=90):
    """
    Find optimal eps value for DBSCAN using k-distance graph method
    """
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(data)
    distances, indices = neighbors_fit.kneighbors(data)
    
    distances = np.sort(distances[:, -1], axis=0)
    eps = np.percentile(distances, percentile)
    
    return eps

def cluster_data(data, method='auto', scale=True, visualize=True, **kwargs):
    """
    Main clustering function that automatically clusters your data
    
    Parameters:
    -----------
    data : array-like or DataFrame
        Input data to cluster
    method : str, default='auto'
        Clustering method: 'auto', 'dbscan', 'hdbscan'
    scale : bool, default=True
        Whether to standardize the data
    visualize : bool, default=True
        Whether to create visualization plots
    **kwargs : additional parameters for clustering algorithms
    
    Returns:
    --------
    dict : Dictionary containing:
        - 'labels': cluster labels
        - 'n_clusters': number of clusters found
        - 'n_noise': number of noise points
        - 'metrics': clustering quality metrics
        - 'model': fitted clustering model
        - 'data_scaled': scaled data (if scale=True)
    """
    
    # Convert to numpy array if needed
    if isinstance(data, pd.DataFrame):
        data = data.values
    data = np.array(data)
    
    # Scale data if requested
    if scale:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
    else:
        data_scaled = data.copy()
    
    # Determine method
    if method == 'auto':
        method = 'hdbscan' if HDBSCAN_AVAILABLE else 'dbscan'
    
    # Perform clustering
    if method == 'hdbscan':
        labels, model = perform_hdbscan(data_scaled, **kwargs)
    else:
        labels, model = perform_dbscan(data_scaled, **kwargs)
    
    # Calculate metrics
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    metrics = {}
    if n_clusters > 1:
        mask = labels != -1
        if mask.sum() > 0:
            try:
                metrics['silhouette'] = silhouette_score(data_scaled[mask], labels[mask])
                metrics['davies_bouldin'] = davies_bouldin_score(data_scaled[mask], labels[mask])
            except:
                metrics['silhouette'] = None
                metrics['davies_bouldin'] = None
    
    # Visualize if requested
    if visualize:
        visualize_clusters(data_scaled, labels, method)
    
    # Prepare results
    results = {
        'labels': labels,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'noise_ratio': n_noise / len(labels),
        'metrics': metrics,
        'model': model,
        'data_scaled': data_scaled if scale else None,
        'method': method
    }
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"CLUSTERING RESULTS ({method.upper()})")
    print(f"{'='*50}")
    print(f"Number of clusters: {n_clusters}")
    print(f"Number of noise points: {n_noise} ({results['noise_ratio']:.1%})")
    if metrics:
        if metrics.get('silhouette'):
            print(f"Silhouette Score: {metrics['silhouette']:.3f}")
        if metrics.get('davies_bouldin'):
            print(f"Davies-Bouldin Score: {metrics['davies_bouldin']:.3f}")
    print(f"{'='*50}\n")
    
    return results

def perform_dbscan(data, eps=None, min_samples=None, **kwargs):
    """
    Perform DBSCAN clustering with automatic parameter tuning
    """
    # Auto-tune parameters if not provided
    if min_samples is None:
        min_samples = max(5, int(np.log(len(data))))
    
    if eps is None:
        eps = find_optimal_eps(data, min_samples)
        print(f"Auto-selected eps: {eps:.3f}")
    
    # Perform clustering
    model = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
    labels = model.fit_predict(data)
    
    return labels, model

def perform_hdbscan(data, min_cluster_size=None, min_samples=None, **kwargs):
    """
    Perform HDBSCAN clustering with automatic parameter tuning
    """
    if not HDBSCAN_AVAILABLE:
        print("HDBSCAN not available, falling back to DBSCAN")
        return perform_dbscan(data)
    
    # Auto-tune parameters if not provided
    if min_cluster_size is None:
        min_cluster_size = max(5, int(0.01 * len(data)))
    
    if min_samples is None:
        min_samples = min_cluster_size
    
    # Perform clustering
    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        **kwargs
    )
    labels = model.fit_predict(data)
    
    return labels, model

def visualize_clusters(data, labels, method=''):
    """
    Create visualization of clustering results
    """
    n_features = data.shape[1]
    
    # Determine visualization strategy
    if n_features == 2:
        plot_data = data
    else:
        # Use PCA for dimensionality reduction
        pca = PCA(n_components=2)
        plot_data = pca.fit_transform(data)
        print(f"Data reduced to 2D using PCA (explained variance: {pca.explained_variance_ratio_.sum():.1%})")
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Scatter plot with clusters
    ax1 = axes[0]
    scatter = ax1.scatter(
        plot_data[:, 0], 
        plot_data[:, 1], 
        c=labels, 
        cmap='viridis',
        s=50,
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )
    
    # Highlight noise points
    noise_mask = labels == -1
    if noise_mask.any():
        ax1.scatter(
            plot_data[noise_mask, 0],
            plot_data[noise_mask, 1],
            c='red',
            marker='x',
            s=50,
            label='Noise',
            alpha=0.8
        )
    
    ax1.set_xlabel('Feature 1' if n_features == 2 else 'PC 1')
    ax1.set_ylabel('Feature 2' if n_features == 2 else 'PC 2')
    ax1.set_title(f'{method.upper()} Clustering Results')
    ax1.legend()
    plt.colorbar(scatter, ax=ax1, label='Cluster ID')
    
    # Plot 2: Cluster size distribution
    ax2 = axes[1]
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
    
    cluster_sizes = [np.sum(labels == i) for i in unique_labels]
    if cluster_sizes:
        bars = ax2.bar(range(len(cluster_sizes)), cluster_sizes)
        ax2.set_xlabel('Cluster ID')
        ax2.set_ylabel('Number of Points')
        ax2.set_title('Cluster Size Distribution')
        ax2.set_xticks(range(len(cluster_sizes)))
        
        # Color bars
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.viridis(i / len(bars)))
    
    # Add noise bar if present
    if noise_mask.any():
        ax2.bar(len(cluster_sizes), noise_mask.sum(), color='red', label='Noise')
        ax2.legend()
    
    plt.tight_layout()
    plt.show()

def compare_methods(data, scale=True, visualize=True):
    """
    Compare DBSCAN and HDBSCAN on the same dataset
    """
    results = {}
    
    # Run DBSCAN
    print("\n" + "="*50)
    print("Running DBSCAN...")
    print("="*50)
    results['dbscan'] = cluster_data(data, method='dbscan', scale=scale, visualize=visualize)
    
    # Run HDBSCAN if available
    if HDBSCAN_AVAILABLE:
        print("\n" + "="*50)
        print("Running HDBSCAN...")
        print("="*50)
        results['hdbscan'] = cluster_data(data, method='hdbscan', scale=scale, visualize=visualize)
    
    # Compare results
    print("\n" + "="*50)
    print("COMPARISON SUMMARY")
    print("="*50)
    
    comparison_df = pd.DataFrame({
        'Method': [],
        'Clusters': [],
        'Noise Points': [],
        'Noise %': [],
        'Silhouette': [],
        'Davies-Bouldin': []
    })
    
    for method, result in results.items():
        row = {
            'Method': method.upper(),
            'Clusters': result['n_clusters'],
            'Noise Points': result['n_noise'],
            'Noise %': f"{result['noise_ratio']:.1%}",
            'Silhouette': f"{result['metrics'].get('silhouette', 0):.3f}" if result['metrics'].get('silhouette') else 'N/A',
            'Davies-Bouldin': f"{result['metrics'].get('davies_bouldin', 0):.3f}" if result['metrics'].get('davies_bouldin') else 'N/A'
        }
        comparison_df = pd.concat([comparison_df, pd.DataFrame([row])], ignore_index=True)
    
    print(comparison_df.to_string(index=False))
    print("="*50)
    
    return results

# Example usage and testing
if __name__ == "__main__":
    # Generate sample data for testing
    from sklearn.datasets import make_blobs, make_moons
    
    print("EXAMPLE 1: Blob-like clusters")
    print("-"*50)
    X1, _ = make_blobs(n_samples=300, centers=4, n_features=2, random_state=42, cluster_std=0.5)
    noise = np.random.uniform(-6, 6, (20, 2))
    X1 = np.vstack([X1, noise])
    
    # Single method clustering
    result1 = cluster_data(X1, method='auto')
    
    print("\n\nEXAMPLE 2: Non-convex clusters (moons)")
    print("-"*50)
    X2, _ = make_moons(n_samples=200, noise=0.05, random_state=42)
    
    # Compare both methods
    comparison = compare_methods(X2)
    
    print("\n\nEXAMPLE 3: Quick usage with your own data")
    print("-"*50)
    print("To use with your own data, simply call:")
    print("  result = cluster_data(your_data)")
    print("\nOr for comparison:")
    print("  results = compare_methods(your_data)")
    print("\nThe function will automatically:")
    print("  - Scale your data")
    print("  - Find optimal parameters")
    print("  - Cluster the data")
    print("  - Visualize results")
    print("  - Return cluster labels and metrics")