import joblib
import pandas as pd
from flask import Flask, request, jsonify
import numpy as np

# --- 1. Initialize Flask App and Load Model ---
app = Flask(__name__)

# Load the trained model artifact
try:
    MODEL_PATH = 'plant_growth_clf_model.pkl'
    clf = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}. Check file path.")
    clf = None
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    clf = None

# --- 2. Define Encoding Maps (Crucial for consistent preprocessing) ---
# These mappings must be derived from the LabelEncoder fits during training.
# Based on the original data and the LabelEncoder fit:
# Soil_Type: clay(0), loam(1), sandy(2)
# Water_Frequency: bi-weekly(0), daily(1), weekly(2)
LABEL_ENCODING_MAPS = {
    'Soil_Type': {'clay': 0, 'loam': 1, 'sandy': 2},
    'Water_Frequency': {'bi-weekly': 0, 'daily': 1, 'weekly': 2}
}

# This is the exact feature order used to train the model. 
# The input DataFrame MUST match this order for prediction.
TRAINING_FEATURES_ORDER = [
    'Sunlight_Hours',
    'Temperature_Extracted',
    'Humidity_Extracted',
    'Soil_Type', 
    'Water_Frequency', 
    'Fertilizer_Type_none',
    'Fertilizer_Type_organic'
]


# --- 3. Preprocessing Function ---
def preprocess_input(data: dict) -> pd.DataFrame:
    """
    Takes raw JSON input, performs the exact transformations used during training,
    and returns a DataFrame with the features in the correct order.
    """
    # Create a DataFrame from the input data (assuming single plant data)
    df_raw = pd.DataFrame([data])

    # 1. Feature Extraction/Rounding (Simulation of previous steps)
    # The model was trained on Temperature/Humidity rounded to 2 decimal places.
    df_raw['Temperature_Extracted'] = round(df_raw['Temperature'], 2)
    df_raw['Humidity_Extracted'] = round(df_raw['Humidity'], 2)
    
    # 2. Label Encoding (Soil_Type and Water_Frequency)
    df_raw['Soil_Type'] = df_raw['Soil_Type'].map(LABEL_ENCODING_MAPS['Soil_Type'])
    df_raw['Water_Frequency'] = df_raw['Water_Frequency'].map(LABEL_ENCODING_MAPS['Water_Frequency'])

    # 3. One-Hot Encoding (Fertilizer_Type)
    # Create the two dummy columns and set to 0.
    df_raw['Fertilizer_Type_none'] = 0
    df_raw['Fertilizer_Type_organic'] = 0
    
    # Set the appropriate dummy column to 1 based on the input
    fertilizer = df_raw['Fertilizer_Type'].iloc[0]
    if fertilizer == 'none':
        df_raw['Fertilizer_Type_none'] = 1
    elif fertilizer == 'organic':
        df_raw['Fertilizer_Type_organic'] = 1
    # 'chemical' remains all zeros (implied by drop_first=True)

    # 4. Select final features and ensure correct order
    # Drop original columns used only for calculation/mapping
    X_processed = df_raw[TRAINING_FEATURES_ORDER]
    
    return X_processed


# --- 4. Prediction API Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    if clf is None:
        return jsonify({'error': 'Model not loaded.'}), 500

    try:
        # Get JSON data from the request
        json_data = request.get_json(force=True)

        # 1. Preprocess the data
        X_final = preprocess_input(json_data)
        
        # Check for NaN values resulting from bad input mapping
        if X_final.isnull().values.any():
            return jsonify({'error': 'Invalid categorical value provided.'}), 400

        # 2. Make prediction
        prediction = clf.predict(X_final)
        
        # 3. Format output
        # Convert the prediction (0 or 1) back to a meaningful string
        milestone_result = 'Milestone Reached' if prediction[0] == 1 else 'No Milestone Reached'

        return jsonify({
            'prediction': int(prediction[0]),
            'result': milestone_result
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# To run the app locally:
if __name__ == '__main__':
    # When deploying, a production server (like Gunicorn) will run this.
    app.run(debug=True)
