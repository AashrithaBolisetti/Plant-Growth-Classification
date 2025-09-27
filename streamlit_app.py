import joblib
import pandas as pd
import streamlit as st
import numpy as np

# --- 1. Load Model and Define Constants ---

# Define the model path
MODEL_PATH = 'plant_growth_clf_model.pkl'

# Load the trained model artifact
try:
    clf = joblib.load(MODEL_PATH)
    st.success("Machine Learning Model Loaded Successfully!")
except Exception as e:
    st.error(f"FATAL ERROR: Could not load the model. Error: {e}")
    clf = None
    
# These mappings must exactly reflect the LabelEncoder fits during training.
# Soil_Type: clay(0), loam(1), sandy(2)
# Water_Frequency: bi-weekly(0), daily(1), weekly(2)
LABEL_ENCODING_MAPS = {
    'Soil_Type': {'clay': 0, 'loam': 1, 'sandy': 2},
    'Water_Frequency': {'bi-weekly': 0, 'daily': 1, 'weekly': 2}
}

# The exact feature order used to train the Random Forest model.
TRAINING_FEATURES_ORDER = [
    'Sunlight_Hours',
    'Temperature_Extracted',
    'Humidity_Extracted',
    'Soil_Type', 
    'Water_Frequency', 
    'Fertilizer_Type_none',
    'Fertilizer_Type_organic'
]


# --- 2. Preprocessing Function ---
def preprocess_input(input_data: dict) -> pd.DataFrame:
    """
    Performs the exact transformations used during training on a single input.
    """
    df_raw = pd.DataFrame([input_data])

    # 1. Feature Extraction/Rounding
    df_raw['Temperature_Extracted'] = round(df_raw['Temperature'], 2)
    df_raw['Humidity_Extracted'] = round(df_raw['Humidity'], 2)
    
    # 2. Label Encoding (Soil_Type and Water_Frequency)
    df_raw['Soil_Type'] = df_raw['Soil_Type'].map(LABEL_ENCODING_MAPS['Soil_Type'])
    df_raw['Water_Frequency'] = df_raw['Water_Frequency'].map(LABEL_ENCODING_MAPS['Water_Frequency'])

    # 3. One-Hot Encoding (Fertilizer_Type)
    df_raw['Fertilizer_Type_none'] = 0
    df_raw['Fertilizer_Type_organic'] = 0
    
    fertilizer = df_raw['Fertilizer_Type'].iloc[0]
    if fertilizer == 'none':
        df_raw['Fertilizer_Type_none'] = 1
    elif fertilizer == 'organic':
        df_raw['Fertilizer_Type_organic'] = 1
    # 'chemical' is the reference class (all zeros)

    # 4. Final Feature Selection and Ordering
    X_processed = df_raw[TRAINING_FEATURES_ORDER]
    
    # Fill any NaN resulting from missing or invalid categorical inputs with 0
    # (While we display an error later, this prevents model crash)
    X_processed = X_processed.fillna(0)

    return X_processed


# --- 3. Streamlit Application Interface ---
st.title("🌱 Plant Growth Milestone Predictor")
st.markdown("Enter the environmental conditions and soil parameters to predict if the plant will reach its growth milestone.")

if clf is not None:
    # Create input fields for the user
    with st.form("input_form"):
        st.header("Environmental Data")
        
        # Numeric Inputs
        sunlight_hours = st.slider("Sunlight Hours (h)", min_value=1.0, max_value=10.0, value=6.0, step=0.1)
        temperature = st.slider("Temperature (°C)", min_value=15.0, max_value=35.0, value=25.0, step=0.1)
        humidity = st.slider("Humidity (%)", min_value=40.0, max_value=80.0, value=60.0, step=0.1)

        st.header("Plant Parameters")

        # Categorical Inputs
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
            
            # Check for bad mapping resulting in NaNs (e.g., if a new soil type was added)
            if X_final['Soil_Type'].isnull().any() or X_final['Water_Frequency'].isnull().any():
                st.error("Invalid categorical value detected. Check soil type or water frequency input.")
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
            st.exception(f"An unexpected error occurred during prediction: {e}")
