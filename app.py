import joblib
import pandas as pd
import streamlit as st
import numpy as np

# --- 1. Initialize Flask App and Load Model ---
# (No changes here)
MODEL_PATH = 'plant_growth_clf_model.pkl'

try:
    clf = joblib.load(MODEL_PATH)
    st.success("Machine Learning Model Loaded Successfully!")
except Exception as e:
    st.error(f"FATAL ERROR: Could not load the model. Error: {e}")
    clf = None
    
# --- 2. Define Encoding Maps and Feature Order (CRUCIAL UPDATE) ---

# These mappings are derived from the unique values in the original CSV
# and the result of LabelEncoder's fit_transform (usually alphabetical).

LABEL_ENCODING_MAPS = {
    # Unique values in Soil_Type: ['clay', 'loam', 'sandy']
    'Soil_Type': {'clay': 0, 'loam': 1, 'sandy': 2},
    # Unique values in Water_Frequency: ['bi-weekly', 'daily', 'weekly']
    'Water_Frequency': {'bi-weekly': 0, 'daily': 1, 'weekly': 2}
    # Note: Fertilizer_Type is handled by One-Hot Encoding later
}

# This is the exact feature order the model expects. We ensure this order explicitly.
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
def preprocess_input(input_data: dict) -> pd.DataFrame:
    """
    Performs the exact transformations used during training on a single input.
    """
    df_raw = pd.DataFrame([input_data])

    # 1. Feature Extraction/Rounding
    df_raw['Temperature_Extracted'] = round(df_raw['Temperature'], 2)
    df_raw['Humidity_Extracted'] = round(df_raw['Humidity'], 2)
    
    # 2. Label Encoding (Soil_Type and Water_Frequency)
    # .map() will return NaN if the key is not found, which we check for later.
    df_raw['Soil_Type'] = df_raw['Soil_Type'].map(LABEL_ENCODING_MAPS['Soil_Type'])
    df_raw['Water_Frequency'] = df_raw['Water_Frequency'].map(LABEL_ENCODING_MAPS['Water_Frequency'])

    # 3. One-Hot Encoding (Fertilizer_Type)
    # Initialize all One-Hot columns to 0. This is necessary because pd.get_dummies 
    # might not create all columns if a category is missing in the input.
    df_raw['Fertilizer_Type_none'] = 0
    df_raw['Fertilizer_Type_organic'] = 0
    
    # Safely get the fertilizer type
    fertilizer = df_raw['Fertilizer_Type'].iloc[0]
    
    # Set the appropriate dummy column to 1
    if fertilizer == 'none':
        df_raw['Fertilizer_Type_none'] = 1
    elif fertilizer == 'organic':
        df_raw['Fertilizer_Type_organic'] = 1
    # 'chemical' results in both columns being 0 (which is correct for drop_first=True)


    # 4. Final Feature Selection and Ordering
    
    # Ensure all required features are present before selecting the final set.
    # We must ensure Temperature/Humidity are floats for the model.
    X_processed = df_raw[[
        'Sunlight_Hours', 'Temperature_Extracted', 'Humidity_Extracted', 
        'Soil_Type', 'Water_Frequency', 'Fertilizer_Type_none', 
        'Fertilizer_Type_organic'
    ]].astype({
        'Sunlight_Hours': float,
        'Temperature_Extracted': float,
        'Humidity_Extracted': float,
        'Soil_Type': float,
        'Water_Frequency': float,
        'Fertilizer_Type_none': float,
        'Fertilizer_Type_organic': float,
    })
    
    # Re-order the columns explicitly to match the model's training order
    X_processed = X_processed[TRAINING_FEATURES_ORDER]
    
    return X_processed


# --- 4. Streamlit Application Interface ---
st.title("🌱 Plant Growth Milestone Predictor")
st.markdown("Enter the environmental conditions and soil parameters to predict if the plant will reach its growth milestone.")

# ... (The rest of the Streamlit interface remains the same) ...

if clf is not None:
    # Create input fields for the user
    with st.form("input_form"):
        st.header("Environmental Data")
        
        # Numeric Inputs (Use the same ranges as defined previously)
        sunlight_hours = st.slider("Sunlight Hours (h)", min_value=1.0, max_value=10.0, value=6.0, step=0.1)
        temperature = st.slider("Temperature (°C)", min_value=15.0, max_value=35.0, value=25.0, step=0.1)
        humidity = st.slider("Humidity (%)", min_value=40.0, max_value=80.0, value=60.0, step=0.1)

        st.header("Plant Parameters")

        # Categorical Inputs (Using keys from the refined maps)
        soil_type = st.selectbox("Soil Type", list(LABEL_ENCODING_MAPS['Soil_Type'].keys()))
        water_frequency = st.selectbox("Water Frequency", list(LABEL_ENCODING_MAPS['Water_Frequency'].keys()))
        fertilizer_type = st.selectbox("Fertilizer Type", ['chemical', 'organic', 'none'])

        submitted = st.form_submit_button("Predict Growth Milestone")

    if submitted:
        input_data = {
            'Sunlight_Hours': sunlight_hours,
            'Temperature': temperature,
            'Humidity': humidity,
            'Soil_Type': soil_type,
            'Water_Frequency': water_frequency,
            'Fertilizer_Type': fertilizer_type
        }
        
        # 4. Run Prediction
        try:
            X_final = preprocess_input(input_data)
            
            # Check for NaN values resulting from bad input mapping (e.g., if a user manually typed a bad value)
            if X_final.isnull().values.any():
                st.error("Invalid categorical value detected. Please select values only from the dropdowns.")
            else:
                prediction = clf.predict(X_final)[0]
                
                if prediction == 1:
                    st.success(f"## Milestone Prediction: Milestone Reached (Code 1) 🎉")
                    st.balloons()
                else:
                    st.warning(f"## Milestone Prediction: No Milestone Reached (Code 0) 🙁")

                with st.expander("Show Features Used for Prediction"):
                    st.json(X_final.iloc[0].to_dict())

        except Exception as e:
            st.exception(f"An unexpected error occurred during prediction. Check your input data: {e}")
