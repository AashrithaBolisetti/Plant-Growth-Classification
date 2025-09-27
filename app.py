import joblib
import pandas as pd
from flask import Flask, request, jsonify
import numpy as np
import os # Included for robustness, though not strictly needed here

# --- 1. Initialize Flask App and Load Model ---
app = Flask(__name__)

# Define the model path
MODEL_PATH = 'plant_growth_clf_model.pkl'

# Load the trained model artifact once when the app starts
try:
    # Use os.path.join for robust path handling in different OS environments
    clf = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    # If the model fails to load, the deployment will fail, which is correct behavior.
    print(f"FATAL ERROR: Could not load the model from {MODEL_PATH}. Error: {e}")
    clf = None

# --- 2. Define Encoding Maps and Feature Order (CRUCIAL) ---

# These mappings must exactly reflect the LabelEncoder fits during training.
# Based on the original data and the LabelEncoder fit:
LABEL_ENCODING_MAPS = {
    # Assuming standard alphabetical LabelEncoder fit on the three soil types in the data:
    # clay(0), loam(1), sandy(2)
    'Soil_Type': {'clay': 0, 'loam': 1, 'sandy': 2},
    # Water_Frequency: bi-weekly(0), daily(1), weekly(2)
    'Water_Frequency': {'bi-weekly': 0, 'daily': 1, 'weekly': 2}
}

# This is the exact feature order used to train the Random Forest model.
# The input DataFrame for prediction MUST match this order.
TRAINING_FEATURES_ORDER = [
    # Continuous (Original)
    'Sunlight_Hours',
    
    # Continuous (Extracted/Rounded)
    'Temperature_Extracted',
    'Humidity_Extracted',
    
    # Label Encoded
    'Soil_Type', 
    'Water_Frequency', 
    
    # One-Hot Encoded (The encoded columns are created based on the alphabetical order of categories)
    'Fertilizer_Type_none',
    'Fertilizer_Type_organic'
]

# --- 3. Preprocessing Function ---
def preprocess_input(data: dict) -> pd.DataFrame:
    """
    Takes raw JSON input, performs the exact transformations used during training,
    and returns a DataFrame with the features in the correct order.
    """
    # Create a DataFrame from the input data (assuming a single plant data sample)
    df_raw = pd.DataFrame([data])

    # 1. Feature Extraction/Rounding (Simulating Temperature_Extracted and Humidity_Extracted)
    # The model was trained on Temperature/Humidity rounded to 2 decimal places.
    df_raw['Temperature_Extracted'] = round(df_raw['Temperature'], 2)
    df_raw['Humidity_Extracted'] = round(df_raw['Humidity'], 2)
    
    # 2. Label Encoding (Soil_Type and Water_Frequency)
    # Use the pre-defined maps to convert strings to integers.
    df_raw['Soil_Type'] = df_raw['Soil_Type'].map(LABEL_ENCODING_MAPS['Soil_Type'])
    df_raw['Water_Frequency'] = df_raw['Water_Frequency'].map(LABEL_ENCODING_MAPS['Water_Frequency'])

    # 3. One-Hot Encoding (Fertilizer_Type)
    # Initialize the two required dummy columns to 0.
    df_raw['Fertilizer_Type_none'] = 0
    df_raw['Fertilizer_Type_organic'] = 0
    
    # Set the appropriate dummy column to 1 based on the input 'Fertilizer_Type'
    fertilizer = df_raw['Fertilizer_Type'].iloc[0]
    if fertilizer == 'none':
        df_raw['Fertilizer_Type_none'] = 1
    elif fertilizer == 'organic':
        df_raw['Fertilizer_Type_organic'] = 1
    # 'chemical' remains all zeros (implied by drop_first=True in training)

    # 4. Final Feature Selection and Ordering
    # Select only the features the model expects and ensure they are in the correct order.
    X_processed = df_raw[TRAINING_FEATURES_ORDER]
    
    return X_processed


# --- 4. Prediction API Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    if clf is None:
        return jsonify({'error': 'Internal server error: Model not available.'}), 500

    try:
        # Get JSON data from the request
        json_data = request.get_json(force=True)

        # 1. Preprocess the data
        X_final = preprocess_input(json_data)
        
        # Validate input for missing/unmappable categories
        if X_final.isnull().values.any():
            return jsonify({'error': 'Invalid categorical value provided. Check Soil_Type or Water_Frequency.'}), 400

        # 2. Make prediction
        prediction = clf.predict(X_final)
        
        # 3. Format output
        milestone_result = 'Milestone Reached (1)' if prediction[0] == 1 else 'No Milestone Reached (0)'

        return jsonify({
            'prediction_code': int(prediction[0]),
            'result_description': milestone_result,
            'features_used': X_final.iloc[0].to_dict() # Show the features sent to the model for verification
        })

    except Exception as e:
        # Catch any unexpected errors during processing
        return jsonify({'error': f'An error occurred during prediction: {e}'}), 500

# --- 5. Removed Flask Development Server ---
# The code block 'if __name__ == "__main__": app.run(debug=True)' 
# MUST be removed/commented out for deployment on Streamlit Cloud (or production environments).
# The platform handles running the web application.
