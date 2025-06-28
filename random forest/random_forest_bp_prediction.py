import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Set random seed for reproducibility
np.random.seed(42)

def load_and_preprocess_data(file_path):
    # Load the data
    df = pd.read_csv(file_path)
    
    # Define categorical and numerical columns
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

    # Specify all possible categories for each categorical feature
    categories = [
        ['Male', 'Female'],                # Gender
        ['Yes', 'No'],                     # Pregnancy
        ['Never', 'Former', 'Current'],    # Smoking_Status
        ['Sedentary', 'Light', 'Moderate', 'Active'],  # Physical_Activity_Level
        ['Yes', 'No'],                     # Chronic_kidney_disease
        ['Yes', 'No'],                     # Adrenal_and_thyroid_disorders
        ['Yes', 'No'],                     # Family_History
        ['Yes', 'No']                      # Diabetes
    ]

    # Create preprocessing pipeline with explicit categories and handle_unknown='ignore'
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(categories=categories, handle_unknown='ignore'), categorical_cols)
        ])
    
    # Prepare features and target
    X = df.drop('Blood_Pressure_Abnormality', axis=1)
    y = df['Blood_Pressure_Abnormality']
    
    # Ensure categorical columns are strings and strip whitespace
    for col in categorical_cols:
        X[col] = X[col].astype(str).str.strip()
    
    return X, y, preprocessor, numerical_cols, categorical_cols

def train_random_forest(X_train, y_train):
    # Initialize Random Forest classifier with optimized parameters
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    
    # Train the model
    rf_model.fit(X_train, y_train)
    return rf_model

def evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Print metrics
    print("\nModel Performance Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('random forest/confusion_matrix_rf.png')
    plt.close()
    
    return accuracy, precision, recall, f1

def plot_feature_importance(model, feature_names):
    # Get feature importance
    importances = model.feature_importances_
    
    # Create DataFrame for better visualization
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('random forest/feature_importance_rf.png')
    plt.close()

def main():
    try:
        # Load and preprocess data
        X, y, preprocessor, numerical_cols, categorical_cols = load_and_preprocess_data('Dataset/ultimate dataset.csv')
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Fit and transform the training data
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Train the model
        model = train_random_forest(X_train_processed, y_train)
        
        # Evaluate the model
        evaluate_model(model, X_test_processed, y_test)
        
        # Get feature names after preprocessing
        cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
        feature_names = numerical_cols + list(cat_features)
        
        # Plot feature importance
        plot_feature_importance(model, feature_names)
        
        # Save the model and preprocessor
        model_data = {
            'model': model,
            'preprocessor': preprocessor,
            'feature_names': feature_names
        }
        joblib.dump(model_data, 'model.bin')
        print("\nModel saved successfully as 'model.bin'")
        
    except FileNotFoundError:
        print("Error: Dataset file not found. Please ensure the data file exists in the correct location.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 