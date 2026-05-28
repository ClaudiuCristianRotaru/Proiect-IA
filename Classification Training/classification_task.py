import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
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

def main():
    used_models = {
        'Random Forest (n=200)': RandomForestClassifier(n_estimators=200, random_state=42),
        'K-Nearest Neighbors (k=29)': KNeighborsClassifier(n_neighbors=29),
        'Support Vector Machine (C=0.1)': SVC(C=0.1, kernel='rbf', random_state=42)
    }
    
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    
    print("="*60)
    print("     IN-DEPTH EVALUATION:     ")
    print("="*60)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    axes = axes.flatten()
    
    output_folder = './models'
    os.makedirs(output_folder, exist_ok=True)
    
    for idx, (name, model) in enumerate(used_models.items()):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        filename = f'{name.replace(' ','_').lower()}.joblib'
        save_path = os.path.join(output_folder, filename)
        joblib.dump(model, save_path)
        print(f"{name} model successfully saved to disk!")
        
        acc = accuracy_score(y_test, preds)
        cm = confusion_matrix(y_test, preds)
        
        print("\n" + f" FINAL METRICS: {name.upper()} ".center(55, "-"))
        print(f"Overall Accuracy: {acc*100:.2f}%")
        print("\nDetailed Performance Matrix:")
        print(classification_report(y_test, preds, target_names=['Healthy (0)', 'Heart Disease (1)']))
        print("-" * 55)
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Healthy (0)', 'Heart Disease (1)'])
        disp.plot(ax=axes[idx], cmap='Blues', values_format='d', colorbar=False)
        
        axes[idx].set_title(name, fontsize=12, fontweight='bold', pad=10)
        axes[idx].set_xlabel("Predicted Label", fontsize=10)
        axes[idx].set_ylabel("True Label", fontsize=10)
        
    plt.suptitle("Confusion Matrix Evaluation", fontsize=14, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()