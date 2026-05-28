import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch

# 1. LOAD AND INITIAL CLEANING
df = pd.read_csv('data/star_classification.csv', encoding='ISO-8859-1')

print(len(df))
# Drop rows without CustomerID and remove canceled orders (Quantity < 0)
# df = df.dropna(subset=['CustomerID'])
df = df[df['class'] == "STAR"]

print(df)
print(len(df))

print(df)
print(len(df))

# 2. Clean the Data: Remove extreme telescope error outliers
# Magnitudes should be within a normal physical range (e.g., between 0 and 30)
filters = ['u', 'g', 'r', 'i', 'z']
for f in filters:
    df = df[(df[f] > 0) & (df[f] < 30)]

# 3. Feature Engineering: Create the Color Index Columns
df['u_g'] = df['u'] - df['g']
df['g_r'] = df['g'] - df['r']
df['r_i'] = df['r'] - df['i']
df['i_z'] = df['i'] - df['z']

# 4. Drop the junk columns and the raw filters to prevent redundancy
columns_to_drop = [
    'obj_ID', 'run_ID', 'rerun_ID', 'cam_col', 'field_ID', 
    'spec_obj_ID', 'plate', 'MJD', 'fiber_ID', 'class',
    'u', 'g', 'i', 'z' # Keeping 'r' to use as the brightness Y-axis for the HR plot
]
clean_stars = df.drop(columns=columns_to_drop, axis=1)

# Inspect your mathematically optimized dataset
print("Engineered Columns for ML/Plotting:\n", clean_stars.columns.tolist())
print("\nFinal Shape (Clean Rows, Perfect Columns):", clean_stars.shape)

clustering_sample = clean_stars.sample(n=3000, random_state=42).copy()

# Select our physical clustering axes (The color indices + brightness)
features_for_clustering = ['u_g', 'g_r', 'r_i', 'i_z', 'r']
X = clustering_sample[features_for_clustering]

# =====================================================================
# 3. FEATURE SCALING (Mandatory for distance-based models)
# =====================================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =====================================================================
# 4. ALGORITHM 1: K-MEANS CLUSTERING
# =====================================================================
print("Running K-Means Clustering...")
# Setting n_clusters=3 to look for major stellar populations (e.g., Dwarfs, Main Sequence, Giants)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

# =====================================================================
# 5. ALGORITHM 2: AGGLOMERATIVE HIERARCHICAL CLUSTERING
# =====================================================================
print("Running Agglomerative Hierarchical Clustering...")
ward_cluster = AgglomerativeClustering(n_clusters=3, linkage='ward')
hierarchical_labels = ward_cluster.fit_predict(X_scaled)

# =====================================================================
# 6. COMPARATIVE EVALUATION (The "Comparatii si Interpretari" Rubric)
# =====================================================================
print("\n--- Model Evaluation Metrics ---")
kmeans_sil = silhouette_score(X_scaled, kmeans_labels)
hier_sil = silhouette_score(X_scaled, hierarchical_labels)

print(f"K-Means Silhouette Score:         {kmeans_sil:.4f}")
print(f"Hierarchical Silhouette Score:    {hier_sil:.4f}")

# Save the predictions back into our sample dataframe for plotting
clustering_sample['KMeans_Cluster'] = kmeans_labels

# =====================================================================
# 7. VISUALIZATION: PROJECTING CLUSTERS ONTO THE H-R DIAGRAM
# =====================================================================
print("Generating Hertzsprung-Russell Cluster Projection...")
plt.figure(figsize=(11, 8))

# Scatter plot using our physical axes: Temperature Proxy (g_r) vs Brightness (r)
# Color-coded by the unsupervised K-Means cluster assignments
scatter = plt.scatter(clustering_sample['g_r'], 
                      clustering_sample['r'], 
                      c=clustering_sample['KMeans_Cluster'], 
                      cmap='viridis', 
                      s=4, 
                      alpha=0.8)

# Invert the vertical axis because lower telescope magnitude means brighter star
plt.gca().invert_yaxis()

# Documenting and styling the plot
plt.title("Unsupervised K-Means Clusters Mapped onto the Hertzsprung-Russell Space", fontsize=14)
plt.xlabel("Surface Temperature Proxy / Color Index (g - r)", fontsize=12)
plt.ylabel("Apparent Brightness / r-band Magnitude (Inverted)", fontsize=12)
plt.colorbar(scatter, label='Assigned Cluster ID')
plt.grid(True, linestyle='--', alpha=0.3)

plt.show()