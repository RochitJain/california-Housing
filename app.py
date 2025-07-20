import streamlit as st
import joblib
import numpy as np

st.title("🏡 California Housing Price Predictor")

# Safe model loading
try:
    model = joblib.load('california_pipeline.pkl')
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

# Feature inputs
MedInc = st.number_input("Median Income", min_value=0.0)
HouseAge = st.number_input("House Age", min_value=0.0)
AveRooms = st.number_input("Average Rooms", min_value=0.0)
AveBedrms = st.number_input("Average Bedrooms", min_value=0.0)
Population = st.number_input("Population", min_value=0.0)
AveOccup = st.number_input("Average Occupants", min_value=0.0)
Latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0)
Longitude = st.number_input("Longitude", min_value=-125.0, max_value=-113.0)

if st.button("Predict Price"):
    try:
        input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                                Population, AveOccup, Latitude, Longitude]])
        prediction = model.predict(input_data)
        st.success(f"🏠 Predicted House Price: ${prediction[0]*100000:.2f}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")