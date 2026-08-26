import streamlit as st
import pandas as pd
import joblib


# -------------------------
# Load trained model
# -------------------------

model = joblib.load("house_price_model.pkl")


# -------------------------
# Page configuration
# -------------------------

st.set_page_config(
    page_title="Bangalore House Price Predictor",
    page_icon="🏠",
    layout="centered"
)


# -------------------------
# Title
# -------------------------

st.title("🏠 Bangalore House Price Predictor")

st.write(
    "Enter the property details below to estimate its price."
)


# -------------------------
# User inputs
# -------------------------

area_type = st.selectbox(
    "Area Type",
    [
        "Super built-up Area",
        "Built-up Area",
        "Plot Area",
        "Carpet Area"
    ]
)


availability = st.text_input(
    "Availability",
    placeholder="e.g. Ready To Move"
)


location = st.text_input(
    "Location",
    placeholder="e.g. Whitefield"
)


total_sqft = st.text_input(
    "Total Sqft",
    placeholder="e.g. 1200 or 1200-1500 or 100 Sq. Meter"
)


bhk = st.number_input(
    "BHK",
    min_value=1,
    max_value=20,
    value=2,
    step=1
)


bath = st.number_input(
    "Bathrooms",
    min_value=1,
    max_value=20,
    value=2,
    step=1
)


balcony = st.number_input(
    "Balconies",
    min_value=0,
    max_value=10,
    value=1,
    step=1
)


# -------------------------
# Prediction
# -------------------------

if st.button("Predict Price"):

    input_data = pd.DataFrame({
        "area_type": [area_type],
        "availability": [availability],
        "location": [location],
        "total_sqft": [total_sqft],
        "bath": [bath],
        "balcony": [balcony],
        "bhk": [bhk]
    })

    prediction = model.predict(input_data)[0]

    st.success(
        f"Estimated Price: ₹{prediction:.2f} Lakhs"
    )
