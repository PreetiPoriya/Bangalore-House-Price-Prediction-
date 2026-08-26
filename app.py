import streamlit as st
import pandas as pd
import joblib

import re

def clean_total_sqft(X):

    X = X.copy()
    s = X["total_sqft"].astype(str).str.strip()

    # Start with NaN
    result = pd.Series(np.nan, index=s.index, dtype=float)


    # --------------------------------------------------
    # 1. Numeric values
    # --------------------------------------------------

    numeric_mask = s.str.fullmatch(
        r"\d+(\.\d+)?",
        na=False
    )

    result.loc[numeric_mask] = (
        s.loc[numeric_mask].astype(float)
    )


    # --------------------------------------------------
    # 2. Range values → midpoint
    # --------------------------------------------------

    range_mask = s.str.fullmatch(
        r"\d+(\.\d+)?\s*-\s*\d+(\.\d+)?",
        na=False
    )

    ranges = s.loc[range_mask].str.split("-", expand=True)

    result.loc[range_mask] = (
        ranges[0].astype(float) +
        ranges[1].astype(float)
    ) / 2


    # --------------------------------------------------
    # 3. Unit-based values → convert to sqft
    # --------------------------------------------------

    conversion_factors = {
        "Sq. Meter": 10.7639,
        "Sq. Yards": 9,
        "Acres": 43560,
        "Cents": 435.6,
        "Guntha": 1089,
        "Grounds": 2400,
        "Perch": 272.25
    }

    for unit, factor in conversion_factors.items():

        mask = s.str.fullmatch(
            rf"\d+(\.\d+)?\s*{re.escape(unit)}",
            na=False
        )

        values = (
            s.loc[mask]
            .str.extract(r"(\d+(?:\.\d+)?)")[0]
            .astype(float)
        )

        result.loc[mask] = values * factor


    # Return DataFrame
    return result.to_frame(name="total_sqft")


def clean_availability(X):
    X = X.copy()

    X["availability"] = np.where(
        X["availability"].isin(
            ["Ready To Move", "Immediate Possession"]
        ),
        "Ready / Immediate Possession",
        "Date-based Possession"
    )

    return X

def clean_location(X):
    X = X.copy()

    # Standardize location names
    X["location"] = (
        X["location"]
        .str.strip()
        .str.lower()
    )

    # Fill missing values with mode
    mode = X["location"].mode()[0]
    X["location"] = X["location"].fillna(mode)

    return X


# --------------------------------------------------
# Load trained pipeline
# --------------------------------------------------

model = joblib.load("house_price_model.pkl")


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Bangalore House Price Predictor",
    page_icon="🏠",
    layout="centered"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🏠 Bangalore House Price Predictor")

st.write(
    "Enter the property details below to estimate its price."
)


# --------------------------------------------------
# Input fields
# --------------------------------------------------

area_type = st.selectbox(
    "Area Type",
    [
        "Super built-up Area",
        "Built-up Area",
        "Plot Area",
        "Carpet Area"
    ]
)


availability = st.selectbox(
    "Availability",
    [
        "Ready To Move",
        "Immediate Possession",
        "15-Aug-2026",
        "18-Dec-2026",
        "30-Jun-2027"
    ]
)


location = st.text_input(
    "Location",
    placeholder="e.g. Whitefield"
)


total_sqft = st.text_input(
    "Total Sqft",
    placeholder="e.g. 1200 or 1200-1500 or 100 Sq. Meter"
)


col1, col2, col3 = st.columns(3)

with col1:
    bhk = st.number_input(
        "BHK",
        min_value=1,
        max_value=20,
        value=2,
        step=1
    )

with col2:
    bath = st.number_input(
        "Bathrooms",
        min_value=1,
        max_value=20,
        value=2,
        step=1
    )

with col3:
    balcony = st.number_input(
        "Balconies",
        min_value=0,
        max_value=10,
        value=1,
        step=1
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

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
