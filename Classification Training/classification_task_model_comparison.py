import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

print("Loading and preprocessing heart disease clinical data...")
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


models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Naive Bayes': GaussianNB(),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Support Vector Machine': SVC(kernel='rbf', probability=True, random_state=42),
}

accuracies = {}

print("\n==================================================")
print("     COMPREHENSIVE COURSE PERFORMANCE REPORTS     ")
print("==================================================")

for name, model in models.items():
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    accuracies[name] = acc
    
    print(f"\n>>> Method: {name}")
    print(f"    Overall Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    print("    Detailed Classification Matrix:")
    print(classification_report(y_test, preds, target_names=['Normal (0)', 'Heart Disease (1)']))
    print("-" * 55)

print("\nGenerating final course-wide algorithmic comparison chart...")
plt.figure(figsize=(13, 6))

colors = ['royalblue', 'orange', 'forestgreen', 'mediumpurple', 'crimson', 'teal', 'darkgoldenrod']
bars = plt.bar(accuracies.keys(), accuracies.values(), color=colors, edgecolor='black', width=0.55)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval*100:.1f}%", 
             ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.title("Heart Disease Detection", fontsize=14, fontweight='bold')
plt.ylabel("Accuracy Score (Test Set)", fontsize=11)
plt.ylim(0, 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.3)

plt.xticks(rotation=15, ha='right', fontweight='semibold')
plt.tight_layout()

plt.show()