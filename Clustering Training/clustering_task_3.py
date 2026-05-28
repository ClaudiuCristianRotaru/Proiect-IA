import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# =====================================================================
# 1. LOAD AND PREPROCESS SATELLITE DATA
# =====================================================================
print("Loading Hipparcos Satellite Archive...")
# Replace with your local file path (typically 'hipparcos-voidmain.csv')
df = pd.read_csv('./data/hipparcos.csv')

# Isolate only the three core pillars and drop missing rows
core_data = df[['Vmag', 'Plx', 'B-V']].dropna().copy()

# Clean Data: Parallax must be positive to calculate logarithmic distance
core_data = core_data[core_data['Plx'] > 0]

# =====================================================================
# 2. FEATURE ENGINEERING: CALCULATING ABSOLUTE MAGNITUDE
# =====================================================================
# Applying the astronomical distance modulus formula via parallax vector
core_data['Absolute_Mag'] = core_data['Vmag'] + 5 * np.log10(core_data['Plx']) - 10

print(f"Cleaned dataset contains {len(core_data)} stars ready for processing.")

# Save a clean sample to protect memory during Hierarchical clustering execution
clustering_sample = core_data.sample(n=3000, random_state=42).copy()

# =====================================================================
# 3. MATRIX SELECTION AND SCALING
# =====================================================================
# X-axis proxy (B-V) and Y-axis proxy (Absolute_Mag) form our clustering space
features = ['B-V', 'Absolute_Mag']
X = clustering_sample[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =====================================================================
# 4. RUN UNSUPERVISED CLUSTERING (K-Means vs Hierarchical)
# =====================================================================
print("Running K-Means...")
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

print("Running Agglomerative Hierarchical...")
ward = AgglomerativeClustering(n_clusters=3, linkage='ward')
hier_labels = ward.fit_predict(X_scaled)

# Store results for validation checks
clustering_sample['KMeans_Cluster'] = kmeans_labels

# Calculate performance metrics for report defense
km_sil = silhouette_score(X_scaled, kmeans_labels)
hi_sil = silhouette_score(X_scaled, hier_labels)
print(f"\nK-Means Silhouette Score:         {km_sil:.4f}")
print(f"Hierarchical Silhouette Score:    {hi_sil:.4f}")

# =====================================================================
# 5. GENERATING THE TEXTBOOK HR DIAGRAM
# =====================================================================
plt.figure(figsize=(10, 8))

# Color code using our AI cluster groups
scatter = plt.scatter(clustering_sample['B-V'], 
                      clustering_sample['Absolute_Mag'], 
                      c=clustering_sample['KMeans_Cluster'], 
                      cmap='plasma', 
                      s=5, 
                      alpha=0.7)

# Crucial Inversion: negative/low absolute magnitudes mean BRIGHTER stars
plt.gca().invert_yaxis()

# Note: B-V naturally scales from blue (left) to red (right), no X inversion needed!
plt.title("Hipparcos Satellite HR Diagram (Clustered via Unsupervised K-Means)", fontsize=13)
plt.xlabel("Stellar Color Profile / Temperature Proxy (B-V)", fontsize=11)
plt.ylabel("True Intrinsic Brightness / Absolute Magnitude (Inverted)", fontsize=11)
plt.colorbar(scatter, label='AI Isolated Structural Cluster')
plt.grid(True, linestyle='--', alpha=0.3)

plt.show()