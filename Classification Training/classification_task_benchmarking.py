import pandas as pd
import numpy as np
import time
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def load_and_prepare_data():
    df = pd.read_csv('./data/heart.csv')

    df = df[(df['Cholesterol'] > 0) & (df['RestingBP'] > 0)]

    df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})
    df['ExerciseAngina'] = df['ExerciseAngina'].map({'Y': 1, 'N': 0})

    df_encoded = pd.get_dummies(df, columns=['ChestPainType', 'RestingECG', 'ST_Slope'], drop_first=True)

    X = df_encoded.drop('HeartDisease', axis=1)
    y = df_encoded['HeartDisease']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    num_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_and_prepare_data()

rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
knn_model = KNeighborsClassifier(n_neighbors=29)
svm_model = SVC(C=0.1, kernel='rbf', random_state=42)

print("="*90)
print("                  RUNNING EFFICIENCY BENCHMARK                  ")
print("="*90)

# Training speed testing
# Random Forest
start_time = time.time()
rf_model.fit(X_train, y_train)
rf_train_time = time.time() - start_time

# KNN
start_time = time.time()
knn_model.fit(X_train, y_train)
knn_train_time = time.time() - start_time

# SVM
start_time = time.time()
svm_model.fit(X_train, y_train)
svm_train_time = time.time() - start_time

# Prediction speed testing
loops = 1000

# Random Forest
start_time = time.time()
for _ in range(loops):
    rf_model.predict(X_test)
rf_predict_time = time.time() - start_time

# KNN
start_time = time.time()
for _ in range(loops):
    knn_model.predict(X_test)
knn_predict_time = time.time() - start_time

# SVM
start_time = time.time()
for _ in range(loops):
    svm_model.predict(X_test)
svm_predict_time = time.time() - start_time

# Memory footprint testing
rf_size_kb = len(pickle.dumps(rf_model)) / 1024
knn_size_kb = len(pickle.dumps(knn_model)) / 1024
svm_size_kb = len(pickle.dumps(svm_model)) / 1024

# Results
rf_size_str = f"{rf_size_kb:.2f} KB"
knn_size_str = f"{knn_size_kb:.2f} KB"
svm_size_str = f"{svm_size_kb:.2f} KB"

rf_train_str = f"{rf_train_time*1000:.2f} ms"
knn_train_str = f"{knn_train_time*1000:.2f} ms"
svm_train_str = f"{svm_train_time*1000:.2f} ms"

rf_pred_str = f"{rf_predict_time:.4f} sec"
knn_pred_str = f"{knn_predict_time:.4f} sec"
svm_pred_str = f"{svm_predict_time:.4f} sec"

print(f"{'Metric':<25}{'Random Forest (n=200)':<25}{'KNN (k=29)':<20}{'SVM (C=0.1)':<20}")
print("-" * 90)
print(f"{'Model File Size':<25}{rf_size_str:<25}{knn_size_str:<20}{svm_size_str:<20}")
print(f"{'Training Time':<25}{rf_train_str:<25}{knn_train_str:<20}{svm_train_str:<20}")
print(f"{'Inference Time (1k loops)':<25}{rf_pred_str:<25}{knn_pred_str:<20}{svm_pred_str:<20}")
print("="*90)