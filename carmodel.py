import streamlit as st
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

st.title("🚗 Car Price Prediction")
st.write("Enter car details to predict price")

# --- Train model (background mein) ---
df = pd.read_csv("CarPrice_Assignment.csv")
df = pd.get_dummies(df, columns=["fueltype","doornumber","carbody","enginetype"])

x = df[['fueltype_diesel','fueltype_gas','doornumber_four','doornumber_two',
        'carbody_convertible','carbody_hardtop','carbody_hatchback','carbody_sedan',
        'carbody_wagon','enginetype_dohc','enginetype_dohcv','enginetype_l',
        'enginetype_ohc','enginetype_ohcf','enginetype_ohcv','enginetype_rotor',
        "enginesize","curbweight","horsepower"]]
y = df["price"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
lr = linear_model.LinearRegression()
lr.fit(x_train, y_train)

# --- User inputs ---
st.subheader("🔧 Car Details")

col1, col2 = st.columns(2)

with col1:
    enginesize = st.number_input("Engine Size", min_value=50, max_value=400, value=130)
    curbweight = st.number_input("Curb Weight (lbs)", min_value=1000, max_value=6000, value=2500)
    horsepower = st.number_input("Horsepower", min_value=40, max_value=300, value=100)
    fueltype = st.selectbox("Fuel Type", ["gas", "diesel"])
    doornumber = st.selectbox("Door Number", ["four", "two"])

with col2:
    carbody = st.selectbox("Car Body", ["sedan", "hatchback", "wagon", "hardtop", "convertible"])
    enginetype = st.selectbox("Engine Type", ["ohc", "dohc", "ohcv", "ohcf", "l", "dohcv", "rotor"])

# --- Build input row ---
input_data = {
    'fueltype_diesel': [1 if fueltype == 'diesel' else 0],
    'fueltype_gas':    [1 if fueltype == 'gas' else 0],
    'doornumber_four': [1 if doornumber == 'four' else 0],
    'doornumber_two':  [1 if doornumber == 'two' else 0],
    'carbody_convertible': [1 if carbody == 'convertible' else 0],
    'carbody_hardtop':     [1 if carbody == 'hardtop' else 0],
    'carbody_hatchback':   [1 if carbody == 'hatchback' else 0],
    'carbody_sedan':       [1 if carbody == 'sedan' else 0],
    'carbody_wagon':       [1 if carbody == 'wagon' else 0],
    'enginetype_dohc':  [1 if enginetype == 'dohc' else 0],
    'enginetype_dohcv': [1 if enginetype == 'dohcv' else 0],
    'enginetype_l':     [1 if enginetype == 'l' else 0],
    'enginetype_ohc':   [1 if enginetype == 'ohc' else 0],
    'enginetype_ohcf':  [1 if enginetype == 'ohcf' else 0],
    'enginetype_ohcv':  [1 if enginetype == 'ohcv' else 0],
    'enginetype_rotor': [1 if enginetype == 'rotor' else 0],
    'enginesize':   [enginesize],
    'curbweight':   [curbweight],
    'horsepower':   [horsepower],
}

input_df = pd.DataFrame(input_data)

# --- Predict button ---
if st.button("🔍 Predict Price"):
    predicted_price = lr.predict(input_df)[0]
    st.success(f"💰 Estimated Car Price: **${predicted_price:,.2f}**")
    st.info(f"Model R² Score: {lr.score(x_test, y_test):.2f}")
