import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN

def load_and_preprocess_hrd(path):
    df = pd.read_csv(path)
    df = df.dropna().drop_duplicates()
    
    features = ['Temperature (K)', 'Absolute magnitude(Mv)']
    X_raw = df[features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    return df, X_scaled

def run_hrd_clustering(df, X_scaled):
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['kmeans_cluster'] = kmeans.fit_predict(X_scaled)
    
    dbscan = DBSCAN(eps=0.35, min_samples=4)
    df['dbscan_cluster'] = dbscan.fit_predict(X_scaled)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    axes[0].scatter(df['Temperature (K)'], df['Absolute magnitude(Mv)'], c=df['kmeans_cluster'], cmap='rainbow', alpha=0.7, edgecolors='k', s=40)
    axes[0].set_title("K-Means: Geometric Stellar Families", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Temperature (K)")
    axes[0].set_ylabel("Absolute Magnitude (Mv)")
    axes[0].grid(True, linestyle='--', alpha=0.3)
    axes[0].invert_xaxis()
    axes[0].invert_yaxis()
    
    db_colors = ['black' if l == -1 else plt.cm.Set1(l) for l in df['dbscan_cluster']]
    axes[1].scatter(df['Temperature (K)'], df['Absolute magnitude(Mv)'], c=db_colors, alpha=0.7, edgecolors='k', s=40)
    axes[1].set_title("DBSCAN: Density Sequences & Noise", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Temperature (K)")
    axes[1].set_ylabel("Absolute Magnitude (Mv)")
    axes[1].grid(True, linestyle='--', alpha=0.3)
    axes[1].invert_xaxis()  
    axes[1].invert_yaxis()  
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    df, X_scaled = load_and_preprocess_hrd('data/hrd.csv')
    run_hrd_clustering(df, X_scaled)