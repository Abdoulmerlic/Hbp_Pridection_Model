from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model and preprocessor
try:
    model_data = joblib.load('model.bin')
    model = model_data['model']
    preprocessor = model_data['preprocessor']
    feature_names = model_data['feature_names']
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None
    preprocessor = None
    feature_names = None

# Define feature categories
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
    return render_template('landing.html')

@app.route('/predictor')
def predictor():
    return render_template('index.html', 
                         numerical_features=numerical_features,
                         categorical_features=categorical_features)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or preprocessor is None:
        return jsonify({'error': 'Model not loaded properly'}), 500

    try:
        # Get data from request
        data = request.get_json()
        
        # Create a DataFrame with the input data
        input_data = pd.DataFrame([data])
        
        # Preprocess the input data
        processed_data = preprocessor.transform(input_data)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0][1]
        
        # Convert probability to percentage
        probability_percentage = probability * 100
        
        # Determine result based on percentage thresholds
        if probability_percentage > 51:
            result = "True"
            risk_level = "High Risk"
            message = "High risk of blood pressure abnormality detected"
            alert_class = "alert-danger"
            meter_position = 85  # Position for high risk on meter
        elif probability_percentage < 49:
            result = "False"
            risk_level = "Low Risk"
            message = "Low risk of blood pressure abnormality"
            alert_class = "alert-success"
            meter_position = 15  # Position for low risk on meter
        else:
            result = "Neutral"
            risk_level = "Neutral Risk"
            message = "Moderate risk - further monitoring recommended"
            alert_class = "alert-warning"
            meter_position = 50  # Position for neutral risk on meter
        
        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability),
            'probability_percentage': round(probability_percentage, 2),
            'result': result,
            'risk_level': risk_level,
            'message': message,
            'alert_class': alert_class,
            'meter_position': meter_position
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True) 