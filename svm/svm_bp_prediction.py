import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

np.random.seed(42)


def load_and_preprocess_data(file_path):
    file_path = 'Dataset/ultimate dataset.csv'
    df = pd.read_csv(file_path)
    categorical_cols = [
        'Gender', 'Pregnancy', 'Smoking_Status', 
        'Physical_Activity_Level', 'Chronic_kidney_disease',
        'Adrenal_and_thyroid_disorders', 'Family_History', 'Diabetes'
    ]
    numerical_cols = [
        'Level_of_Hemoglobin', 'Genetic_Pedigree_Coefficient', 'Age',
        'BMI', 'Salt_Intake', 'Alcohol_Intake', 'Stress_Level',
        'Cholesterol', 'Sleep_Duration', 'Heart_Rate', 'Glucose',
        'Height', 'Weight'
    ]
    categories = [
        ['Male', 'Female'],
        ['Yes', 'No'],
        ['Never', 'Former', 'Current'],
        ['Sedentary', 'Light', 'Moderate', 'Active'],
        ['Yes', 'No'],
        ['Yes', 'No'],
        ['Yes', 'No'],
        ['Yes', 'No']
    ]
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(categories=categories, handle_unknown='ignore'), categorical_cols)
        ])
    X = df.drop('Blood_Pressure_Abnormality', axis=1)
    y = df['Blood_Pressure_Abnormality']
    for col in categorical_cols:
        X[col] = X[col].astype(str).str.strip()
    return X, y, preprocessor, numerical_cols, categorical_cols

def train_svm(X_train, y_train):
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf'],
        'class_weight': ['balanced']
    }
    grid = GridSearchCV(SVC(probability=True, random_state=42), param_grid, refit=True, cv=5, scoring='f1')
    grid.fit(X_train, y_train)
    print("Best parameters:", grid.best_params_)
    return grid.best_estimator_

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print("\nModel Performance Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title('SVM Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('svm_confusion_matrix.png')
    plt.close()
    return accuracy, precision, recall, f1

def plot_svm_weights(model, feature_names):
    if hasattr(model, 'coef_'):
        weights = model.coef_[0]
        feature_weights = pd.DataFrame({
            'Feature': feature_names,
            'Weight': weights
        })
        feature_weights = feature_weights.sort_values('Weight', key=abs, ascending=False)
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Weight', y='Feature', data=feature_weights)
        plt.title('SVM Feature Weights (Linear Kernel)')
        plt.xlabel('Weight')
        plt.tight_layout()
        plt.savefig('svm_feature_weights.png')
        plt.close()

def main():
    try:
        file_path = 'Dataset/ultimate dataset.csv'
        X, y, preprocessor, numerical_cols, categorical_cols = load_and_preprocess_data(file_path)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        model = train_svm(X_train_processed, y_train)
        evaluate_model(model, X_test_processed, y_test)
        cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
        feature_names = numerical_cols + list(cat_features)
        plot_svm_weights(model, feature_names)
        model_data = {
            'model': model,
            'preprocessor': preprocessor,
            'feature_names': feature_names
        }
        joblib.dump(model_data, 'svm_model.bin')
        print("\nModel saved successfully as 'svm_model.bin'")
    except FileNotFoundError:
        print("Error: Dataset file not found. Please ensure the data file exists in the correct location.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 