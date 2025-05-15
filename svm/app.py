from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model and preprocessor
try:
    model_data = joblib.load('svm_model.bin')
    model = model_data['model']
    preprocessor = model_data['preprocessor']
    feature_names = model_data['feature_names']
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None
    preprocessor = None
    feature_names = None

categorical_features = [
    'Gender', 'Pregnancy', 'Smoking_Status', 
    'Physical_Activity_Level', 'Chronic_kidney_disease',
    'Adrenal_and_thyroid_disorders', 'Family_History', 'Diabetes'
]
numerical_features = [
    'Level_of_Hemoglobin', 'Genetic_Pedigree_Coefficient', 'Age',
    'BMI', 'Salt_Intake', 'Alcohol_Intake', 'Stress_Level',
    'Cholesterol', 'Sleep_Duration', 'Heart_Rate', 'Glucose',
    'Height', 'Weight'
]

@app.route('/')
def home():
    return render_template('index.html', 
                         numerical_features=numerical_features,
                         categorical_features=categorical_features)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or preprocessor is None:
        return jsonify({'error': 'Model not loaded properly'}), 500
    try:
        data = request.get_json()
        input_data = pd.DataFrame([data])
        processed_data = preprocessor.transform(input_data)
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0][1]
        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability),
            'message': 'High risk of blood pressure abnormality' if prediction == 1 else 'Low risk of blood pressure abnormality'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True) 