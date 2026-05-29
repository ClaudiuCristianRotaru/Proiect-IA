import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

def load_and_prepare_student_data():
    df = pd.read_csv('data/students.csv')
    df = df.dropna().drop_duplicates()
    
    string_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
    for col in string_cols:
        df[col] = df[col].astype(str).str.strip()
        
    df_encoded = pd.get_dummies(df, columns=string_cols, drop_first=True)
    
    X = df_encoded.drop('math score', axis=1)
    y = df_encoded['math score']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    num_features = ['reading score', 'writing score']
    scaler = StandardScaler()
    X_train[num_features] = scaler.fit_transform(X_train[num_features])
    X_test[num_features] = scaler.transform(X_test[num_features])
    
    return X_train, X_test, y_train, y_test

def main():
    X_train, X_test, y_train, y_test = load_and_prepare_student_data()
    
    param_grid = {
        'n_estimators': [50, 100, 200, 500, 1000],
        'max_depth': [2, 4, 8, 12, 20],
        'min_samples_split' : [10] # change this > 2
    }
    
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    cv_results = pd.DataFrame(grid_search.cv_results_)
    scores_matrix = cv_results.pivot_table(index='param_max_depth', columns='param_n_estimators', values='mean_test_score', aggfunc='mean')
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(scores_matrix, annot=True, fmt=".4f", cmap="YlGnBu", cbar_kws={'label': 'Mean Validation R²'})
    plt.title('Random Forest Hyperparameter Tuning Grid Search', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Number of Estimators (n_estimators)', fontsize=10)
    plt.ylabel('Maximum Depth (max_depth)', fontsize=10)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()