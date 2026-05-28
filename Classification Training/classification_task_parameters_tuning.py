import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def load_and_prepare_data():
    df = pd.read_csv('./data/heart.csv')
    
    df = df[df['Cholesterol'] > 0]
    df = df[df['RestingBP'] > 0]
        
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

print("="*60)
print("                 TUNING PARAMS                ")
print("="*60)

X_train, X_test, y_train, y_test = load_and_prepare_data()

# Random Forest (n_estimators)
print("\n--- [1/3] Random Forest Estimator Scale Sweep ---")
n_values = [1, 3, 5, 7, 10, 20, 50, 100, 200, 500, 1000, 3000]
rf_train_accs = []
rf_test_accs = []

print(f"{'Num Trees':<12}{'Train Acc':<14}{'Test Acc':<12}")

for n in n_values:
    model = RandomForestClassifier(n_estimators=n, random_state=42)
    model.fit(X_train, y_train)
    tr_acc = accuracy_score(y_train, model.predict(X_train))
    te_acc = accuracy_score(y_test, model.predict(X_test))
    
    rf_train_accs.append(tr_acc)
    rf_test_accs.append(te_acc)
    print(f"{n:<12}{tr_acc*100:.1f}%{'':<7}{te_acc*100:.1f}%")

# Generate Random Forest Plot
plt.figure(figsize=(8, 5))
plt.plot(n_values, rf_train_accs, marker='o', linestyle='-', color='darkgreen', label='Train Accuracy', linewidth=2)
plt.plot(n_values, rf_test_accs, marker='s', linestyle='--', color='forestgreen', label='Test Accuracy', linewidth=2)
plt.xscale('log')
plt.xticks(n_values, labels=[str(n) for n in n_values])
plt.title('Random Forest: Tuning Number of Trees (n_estimators)', fontsize=12, fontweight='bold')
plt.xlabel('Number of Trees (Log Scale)', fontsize=10)
plt.ylabel('Accuracy Score', fontsize=10)
plt.ylim(0.6, 1.05)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
    
# KNN (n_neighbors)
print("\n--- [2/3] K-Nearest Neighbors Spatial Closeness Sweep ---")
k_values = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 100]
knn_train_accs = []
knn_test_accs = []

print(f"{'K Neighbors':<12}{'Train Acc':<14}{'Test Acc':<12}")

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    tr_acc = accuracy_score(y_train, model.predict(X_train))
    te_acc = accuracy_score(y_test, model.predict(X_test))
    
    knn_train_accs.append(tr_acc)
    knn_test_accs.append(te_acc)
    print(f"{k:<12}{tr_acc*100:.1f}%{'':<7}{te_acc*100:.1f}%")

# Generate KNN Plot
plt.figure(figsize=(8, 5))
plt.plot(k_values, knn_train_accs, marker='o', linestyle='-', color='darkorange', label='Train Accuracy', linewidth=2)
plt.plot(k_values, knn_test_accs, marker='s', linestyle='--', color='navy', label='Test Accuracy', linewidth=2)
plt.xticks(k_values)
plt.title('KNN: Tuning Neighborhood Size (k)', fontsize=12, fontweight='bold')
plt.xlabel('Number of Neighbors (k)', fontsize=10)
plt.ylabel('Accuracy Score', fontsize=10)
plt.ylim(0.6, 1.05)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
    
# SVM (C)
print("\n--- [3/3] Support Vector Machine (RBF) Regularization Sweep ---")
c_values = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 2000.0, 5000.0]
svm_train_accs = []
svm_test_accs = []

print(f"{'C Value':<12}{'Train Acc':<14}{'Test Acc':<12}")

for c in c_values:
    model = SVC(C=c, kernel='rbf', random_state=42)
    model.fit(X_train, y_train)
    tr_acc = accuracy_score(y_train, model.predict(X_train))
    te_acc = accuracy_score(y_test, model.predict(X_test))
    
    svm_train_accs.append(tr_acc)
    svm_test_accs.append(te_acc)
    print(f"{c:<12}{tr_acc*100:.1f}%{'':<7}{te_acc*100:.1f}%")

# Generate SVM Plot
plt.figure(figsize=(8, 5))
plt.plot(c_values, svm_train_accs, marker='o', linestyle='-', color='crimson', label='Train Accuracy', linewidth=2)
plt.plot(c_values, svm_test_accs, marker='s', linestyle='--', color='royalblue', label='Test Accuracy', linewidth=2)
plt.xscale('log')
plt.xticks(c_values, labels=[str(c) for c in c_values])
plt.title('SVM (RBF Kernel): Tuning Regularization Parameter (C)', fontsize=12, fontweight='bold')
plt.xlabel('Regularization Penalty Parameter C (Log Scale)', fontsize=10)
plt.ylabel('Accuracy Score', fontsize=10)
plt.ylim(0.3, 1.05)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

print("\n")
print("Done!")