import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# =====================================================================
# 1. DATA PREPARATION (Demographics & Test Preparation Only)
# =====================================================================
print("Loading and preparing student dataset...")
# Make sure the file path matches your local environment setup
df = pd.read_csv('data/students.csv')

# Step A: Binary Mapping for 2-option columns
df['gender'] = df['gender'].map({'male': 1, 'female': 0})
df['lunch'] = df['lunch'].map({'standard': 1, 'free/reduced': 0})
df['test preparation course'] = df['test preparation course'].map({'completed': 1, 'none': 0})

# Step B: One-Hot Encoding for multi-option categories
# drop_first=True eliminates the dummy variable trap for regression stability
df_encoded = pd.get_dummies(df, columns=['race/ethnicity', 'parental level of education'], drop_first=True)

# Step C: Separate X (Inputs) and y (The 3 Simultaneous Grade Targets)
# We drop all three scores from X so the model only relies on demographic cues
X = df_encoded.drop(['math score', 'reading score', 'writing score'], axis=1)
y = df_encoded[['math score', 'reading score', 'writing score']]

# Step D: Train/Test Split (80% training baseline, 20% validation exam)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Features shape: {X_train.shape} | Targets shape: {y_train.shape}")

# =====================================================================
# 2. MULTI-OUTPUT MODEL TRAINING
# =====================================================================
print("\nTraining Multi-Output Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

print("Training Multi-Output Random Forest Regressor...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Generate 2D prediction arrays for the validation dataset
lr_preds = lr_model.predict(X_test)
rf_preds = rf_model.predict(X_test)

# =====================================================================
# 3. VISUALIZATION: GENERATING THE CORRECTED 3x2 EVALUATION GRID
# =====================================================================
print("\nGenerating clean evaluation plots for all 3 grades...")

# Establish a grid matrix containing 3 rows (subjects) and 2 columns (models)
fig, axes = plt.subplots(3, 2, figsize=(14, 18))

targets = ['math score', 'reading score', 'writing score']

# FIXED: Replaced non-standard color names with verified Matplotlib color specs
colors_lr = ['royalblue', 'dodgerblue', 'cornflowerblue']
colors_rf = ['darkgreen', 'forestgreen', 'mediumseagreen']

for i, col_name in enumerate(targets):
    # Isolate the true ground-truth grades for this specific loop iteration
    actuals = y_test[col_name]
    
    # Establish a perfect prediction reference boundary line
    perfect_line = np.linspace(min(actuals), max(actuals), 100)
    
    # -----------------------------------------------------------------
    # COLUMN 1: LINEAR REGRESSION SUBPLOTS (Left Side)
    # -----------------------------------------------------------------
    ax_lr = axes[i, 0]
    r2_lr = r2_score(actuals, lr_preds[:, i])
    mae_lr = mean_absolute_error(actuals, lr_preds[:, i])
    
    ax_lr.scatter(actuals, lr_preds[:, i], color=colors_lr[i], alpha=0.6, edgecolors='k', s=35)
    ax_lr.plot(perfect_line, perfect_line, color='crimson', linestyle='--', linewidth=2, label='Perfect Prediction')
    ax_lr.set_title(f"Linear Reg: {col_name.title()} (R² = {r2_lr:.2f})", fontsize=11, fontweight='bold')
    ax_lr.set_xlabel(f"Actual {col_name.title()}")
    ax_lr.set_ylabel(f"Predicted {col_name.title()}")
    ax_lr.grid(True, linestyle='--', alpha=0.4)
    ax_lr.legend()
    
    # Print numerical readouts to terminal console for validation checks
    print(f"[{col_name.upper()}] Linear Reg -> R²: {r2_lr:.4f} | MAE: {mae_lr:.2f} points")
    
    # -----------------------------------------------------------------
    # COLUMN 2: RANDOM FOREST SUBPLOTS (Right Side)
    # -----------------------------------------------------------------
    ax_rf = axes[i, 1]
    r2_rf = r2_score(actuals, rf_preds[:, i])
    mae_rf = mean_absolute_error(actuals, rf_preds[:, i])
    
    ax_rf.scatter(actuals, rf_preds[:, i], color=colors_rf[i], alpha=0.6, edgecolors='k', s=35)
    ax_rf.plot(perfect_line, perfect_line, color='crimson', linestyle='--', linewidth=2, label='Perfect Prediction')
    ax_rf.set_title(f"Random Forest: {col_name.title()} (R² = {r2_rf:.2f})", fontsize=11, fontweight='bold')
    ax_rf.set_xlabel(f"Actual {col_name.title()}")
    ax_rf.set_ylabel(f"Predicted {col_name.title()}")
    ax_rf.grid(True, linestyle='--', alpha=0.4)
    ax_rf.legend()
    
    print(f"[{col_name.upper()}] Random Forest -> R²: {r2_rf:.4f} | MAE: {mae_rf:.2f} points")

# Polish layout bounds and output canvas frame
plt.tight_layout()
plt.suptitle("Multi-Output Performance Profiles: Demographics-Only Predictions", y=1.01, fontsize=15, fontweight='bold')
plt.show()