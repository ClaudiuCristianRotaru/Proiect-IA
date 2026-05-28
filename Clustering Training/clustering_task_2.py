import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Load the dataset
df = pd.read_csv('./data/hrd.csv') # Replace with your local file path

# 2. Fix the Star Color text typos
df['Star color'] = df['Star color'].str.lower().str.replace('-', ' ').str.strip()

# 3. Create Log columns for the ML clustering distance math
df['log_L'] = np.log10(df['Luminosity(L/Lo)'])
df['log_T'] = np.log10(df['Temperature (K)'])

# 4. Separate features for K-Means (Using Absolute Magnitude and Log columns)
features = ['log_T', 'log_L', 'Radius(R/Ro)', 'Absolute magnitude(Mv)']
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Run K-Means to find the stellar populations
kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# =====================================================================
# THE TEXTBOOK HR DIAGRAM PLOT
# =====================================================================
plt.figure(figsize=(10, 8))

# Plot True Temperature vs Absolute Magnitude, color-coded by AI clusters
scatter = plt.scatter(df['Temperature (K)'], 
                      df['Absolute magnitude(Mv)'], 
                      c=df['Cluster'], 
                      cmap='rainbow', 
                      s=30, 
                      edgecolor='black', 
                      alpha=0.8)

# --- THE CRUCIAL TEXTBOOK INVERSIONS ---
plt.gca().invert_xaxis()  # Puts blistering hot stars on the LEFT
plt.gca().invert_yaxis()  # Puts ultra-bright negative magnitudes on TOP
# ---------------------------------------

# Use a logarithmic scale for the temperature ticks to space them out evenly
plt.xscale('log')
plt.xticks([2000, 4000, 7000, 10000, 20000, 40000], ['2k', '4k', '7k', '10k', '20k', '40k'])

plt.title("Textbook-Perfect HR Diagram (Clustered by K-Means)", fontsize=14)
plt.xlabel("Temperature (Kelvin) -> Hotter on the Left (Log Scale)", fontsize=12)
plt.ylabel("Absolute Magnitude (Mv) -> Brighter at the Top (Inverted)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.colorbar(scatter, label='AI Isolated Stellar Population')

plt.show()