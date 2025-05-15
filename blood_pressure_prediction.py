import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import xgboost as xgb
from sklearn.metrics import roc_curve, auc

# Load the dataset
file_path = 'Dataset/ultimate dataset.csv'
data = pd.read_csv(file_path)

# Display basic info and check for missing values
print("Dataset Info:")
print(data.info())
print("\nMissing Values:")
print(data.isnull().sum())

# Compute correlation matrix
correlation_matrix = data.corr()

# Visualize correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# Display top correlations with the target
target = 'Blood_Pressure_Abnormality'
top_correlations = correlation_matrix[target].sort_values(ascending=False)
print("\nTop Correlations with Blood_Pressure_Abnormality:")
print(top_correlations)

# Define categorical and numerical columns
categorical_cols = ['Gender', 'Pregnancy', 'Smoking_Status', 'Physical_Activity_Level', 'Chronic_kidney_disease', 'Adrenal_and_thyroid_disorders', 'Family_History', 'Diabetes']
numerical_cols = ['Level_of_Hemoglobin', 'Genetic_Pedigree_Coefficient', 'Age', 'BMI', 'Salt_Intake', 'Alcohol_Intake', 'Stress_Level', 'Cholesterol', 'Sleep_Duration', 'Heart_Rate', 'Glucose', 'Height', 'Weight']

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ])

# Prepare features and target
X = data.drop('Blood_Pressure_Abnormality', axis=1)
y = data['Blood_Pressure_Abnormality']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit and transform the training data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Preprocessing completed. Training set shape:", X_train_processed.shape)
print("Test set shape:", X_test_processed.shape)

# Train Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_processed, y_train)
y_pred_rf = rf.predict(X_test_processed)

# Train SVM
svm = SVC(kernel='linear', probability=True, random_state=42)
svm.fit(X_train_processed, y_train)
y_pred_svm = svm.predict(X_test_processed)

# Evaluation function
def evaluate_model(y_true, y_pred, model_name):
    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_true, y_pred))
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return acc, prec, rec, f1

# Evaluate both models
rf_metrics = evaluate_model(y_test, y_pred_rf, "Random Forest")
svm_metrics = evaluate_model(y_test, y_pred_svm, "SVM")

# Confusion matrices
cm_rf = confusion_matrix(y_test, y_pred_rf)
cm_svm = confusion_matrix(y_test, y_pred_svm)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.subplot(1, 2, 2)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Greens')
plt.title('SVM Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrices.png')
plt.close()

# Feature importances (Random Forest)
importances = rf.feature_importances_
# Get feature names after preprocessing
cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
feature_names = numerical_cols + list(cat_features)

# Bar chart of feature importances
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.title('Feature Importances (Random Forest)')
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), np.array(feature_names)[indices], rotation=90)
plt.tight_layout()
plt.savefig('feature_importances_rf.png')
plt.close()

# SVM weights (if linear kernel)
if hasattr(svm, 'coef_'):
    svm_weights = svm.coef_[0]
    print("\nSVM Feature Weights:")
    for name, weight in zip(feature_names, svm_weights):
        print(f"{name}: {weight:.4f}")

# Print feature importances
print("\nRandom Forest Feature Importances:")
for name, importance in zip(feature_names, importances):
    print(f"{name}: {importance:.4f}")

# Model recommendation
print("\nModel Comparison:")
print(f"Random Forest - Accuracy: {rf_metrics[0]:.3f}, Precision: {rf_metrics[1]:.3f}, Recall: {rf_metrics[2]:.3f}, F1: {rf_metrics[3]:.3f}")
print(f"SVM           - Accuracy: {svm_metrics[0]:.3f}, Precision: {svm_metrics[1]:.3f}, Recall: {svm_metrics[2]:.3f}, F1: {svm_metrics[3]:.3f}")

if rf_metrics[3] > svm_metrics[3]:
    print("\nRecommendation: Random Forest performs better based on F1-score.")
elif svm_metrics[3] > rf_metrics[3]:
    print("\nRecommendation: SVM performs better based on F1-score.")
else:
    print("\nRecommendation: Both models perform equally well based on F1-score.")

# Set random seed for reproducibility
np.random.seed(42)

# Load and prepare the data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    # Separate features and target
    X = df.drop('Blood_Pressure_Abnormality', axis=1)
    y = df['Blood_Pressure_Abnormality']
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def train_model(X_train, y_train):
    # Initialize XGBoost classifier
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    # Train the model
    xgb_model.fit(X_train, y_train)
    return xgb_model

def evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
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
    plt.show()
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

def plot_feature_importance(model, feature_names):
    # Get feature importance
    importance = model.feature_importances_
    
    # Create DataFrame for better visualization
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.show()

def main():
    # Load data
    try:
        df = load_data('Dataset/ultimate dataset.csv')
    except FileNotFoundError:
        print("Error: blood_pressure_data.csv not found. Please ensure the data file exists.")
        return
    
    # Preprocess data
    X, y = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)
    
    # Plot feature importance
    feature_names = df.drop('Blood_Pressure_Abnormality', axis=1).columns
    plot_feature_importance(model, feature_names)

if __name__ == "__main__":
    main() 