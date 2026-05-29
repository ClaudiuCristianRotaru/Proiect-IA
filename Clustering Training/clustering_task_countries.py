import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

def load_and_preprocess_countries(path):
    df = pd.read_csv(path)
    df = df.dropna().drop_duplicates()
    features = ['child_mort', 'exports', 'health', 'imports', 'income', 'inflation', 'life_expec', 'total_fer', 'gdpp']
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return df, X_scaled

def print_cluster_groups(df):
    print("--- K-MEANS GROUPS ---")
    for cluster_id, group in df.groupby('kmeans_cluster'):
        countries = group['country'].tolist()
        print(f"Cluster {cluster_id} (Total: {len(countries)}):")
        print(", ".join(countries))
        print()

    print("--- DBSCAN GROUPS ---")
    for cluster_id, group in df.groupby('dbscan_cluster'):
        countries = group['country'].tolist()
        if cluster_id == -1:
            print(f"Noise / Outliers (Total: {len(countries)}):")
        else:
            print(f"Cluster {cluster_id} (Total: {len(countries)}):")
        print(", ".join(countries))
        print()

def run_country_clustering(df, X_scaled, kmeans_k, dbscan_eps, dbscan_min_samples):
    kmeans = KMeans(n_clusters=kmeans_k, random_state=42, n_init=10)
    df['kmeans_cluster'] = kmeans.fit_predict(X_scaled)
    
    dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    df['dbscan_cluster'] = dbscan.fit_predict(X_scaled)
    
    print_cluster_groups(df)
    
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    df['PC1'] = X_pca[:, 0]
    df['PC2'] = X_pca[:, 1]
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    
    axes[0].scatter(df['PC1'], df['PC2'], c=df['kmeans_cluster'], cmap='viridis', alpha=0.6, edgecolors='k', s=50)
    axes[0].set_title(f"K-Means Clustering (k={kmeans_k})", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Principal Component 1 (PC1)")
    axes[0].set_ylabel("Principal Component 2 (PC2)")
    axes[0].grid(True, linestyle='--', alpha=0.3)
    
    axes[1].scatter(df['PC1'], df['PC2'], c=df['dbscan_cluster'], cmap='plasma', alpha=0.6, edgecolors='k', s=50)
    axes[1].set_title(f"DBSCAN Clustering (eps={dbscan_eps}, min_samples={dbscan_min_samples})", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Principal Component 1 (PC1)")
    axes[1].set_ylabel("Principal Component 2 (PC2)")
    axes[1].grid(True, linestyle='--', alpha=0.3)
    
    for i in range(len(df)):
        country_name = df['country'].iloc[i]
        x_val = df['PC1'].iloc[i]
        y_val = df['PC2'].iloc[i]
        
        axes[0].text(x_val + 0.12, y_val, country_name, fontsize=6.5, alpha=0.85)
        axes[1].text(x_val + 0.12, y_val, country_name, fontsize=6.5, alpha=0.85)
            
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    target_kmeans_k = 4
    target_dbscan_eps = 1.22
    target_dbscan_min_samples = 3
    
    df, X_scaled = load_and_preprocess_countries('data/countries.csv')
    run_country_clustering(df, X_scaled, target_kmeans_k, target_dbscan_eps, target_dbscan_min_samples)