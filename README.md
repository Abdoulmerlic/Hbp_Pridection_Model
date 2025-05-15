# Blood Pressure Abnormality Prediction

This project implements machine learning models to predict blood pressure abnormalities based on various health and lifestyle factors. The system uses multiple algorithms including Random Forest, SVM, and XGBoost to provide accurate predictions and insights into the factors affecting blood pressure.

## Features

- Multiple machine learning models (Random Forest, SVM, XGBoost)
- Comprehensive feature analysis and visualization
- Model performance comparison
- Feature importance analysis
- Confusion matrix visualization
- ROC curve analysis

## Dataset

The project uses a dataset containing various health-related features including:
- Demographic information (Age, Gender, Height, Weight)
- Health metrics (BMI, Heart Rate, Cholesterol, Glucose)
- Lifestyle factors (Smoking Status, Physical Activity Level, Sleep Duration)
- Medical history (Chronic kidney disease, Adrenal and thyroid disorders, Family History)
- Other factors (Stress Level, Salt Intake, Alcohol Intake)

## Requirements

The project requires the following Python packages:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost

You can install all required packages using:
```bash
pip install -r requirements.txt
```

## Project Structure

```
├── blood_pressure_prediction.py    # Main prediction script
├── Dataset/                       # Contains the dataset
├── requirements.txt               # Project dependencies
├── feature_importances_rf.png     # Random Forest feature importance visualization
├── confusion_matrices.png         # Model confusion matrices
└── correlation_matrix.png         # Feature correlation matrix
```

## Usage

1. Ensure all required packages are installed
2. Place your dataset in the `Dataset` folder
3. Run the main script:
```bash
python blood_pressure_prediction.py
```

## Model Performance

The project compares multiple models and provides:
- Accuracy scores
- Precision and recall metrics
- F1-scores
- Confusion matrices
- ROC curves
- Feature importance analysis

## Visualizations

The project generates several visualizations:
- Correlation matrix heatmap
- Feature importance plots
- Confusion matrices
- ROC curves

## Contributing

Feel free to submit issues and enhancement requests! 