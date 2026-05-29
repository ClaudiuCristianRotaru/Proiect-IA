import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_and_encode_data(path):
    df = pd.read_csv(path)
    df = df.dropna().drop_duplicates()
    
    df['gender'] = df['gender'].map({'male': 1, 'female': 0})
    df['lunch'] = df['lunch'].map({'standard': 1, 'free/reduced': 0})
    df['test preparation course'] = df['test preparation course'].map({'completed': 1, 'none': 0})
    
    df_encoded = pd.get_dummies(df, columns=['race/ethnicity', 'parental level of education'], drop_first=True)
    return df, df_encoded

def prepare_splits(df, df_encoded, option):
    if option == 4:
        df_encoded['average score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)
        X = df_encoded.drop(['math score', 'reading score', 'writing score', 'average score'], axis=1)
        target_col = 'average score'
    else:
        subjects = {1: 'math score', 2: 'reading score', 3: 'writing score'}
        target_col = subjects[option]
        X = df_encoded.drop([target_col], axis=1)
        
    y = df_encoded[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if option != 4:
        num_features = [s for s in ['math score', 'reading score', 'writing score'] if s != target_col]
        scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        X_train_scaled[num_features] = scaler.fit_transform(X_train[num_features])
        X_test_scaled[num_features] = scaler.transform(X_test[num_features])
        return X_train_scaled, X_test_scaled, y_train, y_test, target_col
    
    return X_train, X_test, y_train, y_test, target_col

def train_and_evaluate(X_train, X_test, y_train, y_test, target_col):
    output_folder = './models'
    os.makedirs(output_folder, exist_ok=True)

    lr_model = LinearRegression(fit_intercept=True)
    lr_model.fit(X_train, y_train)

    filename = f"linear_regression_{target_col.replace(' ','_').lower()}.joblib"
    save_path = os.path.join(output_folder, filename)
    joblib.dump(lr_model, save_path)
    print(f"{lr_model} model successfully saved to disk!")
    
    rf_model = RandomForestRegressor(
        n_estimators=200, 
        max_depth=12, 
        min_samples_split=10, 
        random_state=42
    )
    rf_model.fit(X_train, y_train)

    filename = f"random_forest_{target_col.replace(' ','_').lower()}.joblib"
    save_path = os.path.join(output_folder, filename)
    joblib.dump(rf_model, save_path)
    print(f"{rf_model} model successfully saved to disk!")
    
    lr_preds = lr_model.predict(X_test)
    rf_preds = rf_model.predict(X_test)
    
    print(f"\nGenerating evaluation plots for {target_col.title()}...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    perfect_line = np.linspace(min(y_test), max(y_test), 100)
    
    r2_lr = r2_score(y_test, lr_preds)
    mae_lr = mean_absolute_error(y_test, lr_preds)
    rmse_lr = np.sqrt(mean_squared_error(y_test, lr_preds))
    
    axes[0].scatter(y_test, lr_preds, color='royalblue', alpha=0.6, edgecolors='k', s=35)
    axes[0].plot(perfect_line, perfect_line, color='crimson', linestyle='--', linewidth=2, label='Perfect Prediction')
    axes[0].set_title(f"Linear Reg: {target_col.title()}\n(R² = {r2_lr:.4f} | MAE: {mae_lr:.2f} | RMSE = {rmse_lr:.2f})", fontsize=11, fontweight='bold')
    axes[0].set_xlabel(f"Actual {target_col.title()}")
    axes[0].set_ylabel(f"Predicted {target_col.title()}")
    axes[0].grid(True, linestyle='--', alpha=0.4)
    axes[0].legend()
    print(f"[{target_col.upper()}] Linear Reg -> R²: {r2_lr:.4f} | MAE: {mae_lr:.2f} | RMSE: {rmse_lr:.2f}")
    
    r2_rf = r2_score(y_test, rf_preds)
    mae_rf = mean_absolute_error(y_test, rf_preds)
    rmse_rf = np.sqrt(mean_squared_error(y_test, rf_preds))
    
    axes[1].scatter(y_test, rf_preds, color='darkgreen', alpha=0.6, edgecolors='k', s=35)
    axes[1].plot(perfect_line, perfect_line, color='crimson', linestyle='--', linewidth=2, label='Perfect Prediction')
    axes[1].set_title(f"Random Forest: {target_col.title()}\n(R² = {r2_rf:.4f} | MAE: {mae_rf:.2f} | RMSE = {rmse_rf:.2f})", fontsize=11, fontweight='bold')
    axes[1].set_xlabel(f"Actual {target_col.title()}")
    axes[1].set_ylabel(f"Predicted {target_col.title()}")
    axes[1].grid(True, linestyle='--', alpha=0.4)
    axes[1].legend()
    print(f"[{target_col.upper()}] Random Forest -> R²: {r2_rf:.4f} | MAE: {mae_rf:.2f} | RMSE: {rmse_rf:.2f}")
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    df, df_encoded = load_and_encode_data('data/students.csv')
    for option in range(1,5):
        X_train, X_test, y_train, y_test, target_col = prepare_splits(df, df_encoded, option)
        train_and_evaluate(X_train, X_test, y_train, y_test, target_col)