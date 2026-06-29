import streamlit as st
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

st.title("🚗 Car Price Prediction")
st.write("Multiple Linear Regression Model")

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
prediction = lr.predict(x_test)

score = lr.score(x_test, y_test)
st.metric("R² Score", f"{score:.2f}")

# Results table
result = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted": prediction,
    "Difference": y_test.values - prediction
})
st.subheader("Predictions")
st.dataframe(result)

# Visualization
st.subheader("Actual vs Predicted")
fig, ax = plt.subplots()
sns.scatterplot(x=y_test, y=prediction, ax=ax)
ax.set_xlabel("Actual Price")
ax.set_ylabel("Predicted Price")
st.pyplot(fig)  # plt.show() ki jagah st.pyplot(fig)
