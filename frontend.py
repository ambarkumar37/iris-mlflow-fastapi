import streamlit as st
import requests

st.title("🌼 Iris Flower Prediction App")

st.write("Enter the features below:")

sepal_length = st.number_input("Sepal Length", min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.number_input("Sepal Width", min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.number_input("Petal Length", min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.number_input("Petal Width", min_value=0.0, max_value=10.0, step=0.1)

# FastAPI endpoint
API_URL = "http://127.0.0.1:8000/predict"

if st.button("Predict"):
    payload = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    }

    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        result = response.json()["prediction"]
        st.success(f"🌸 Predicted Iris Species: **{result}**")
    else:
        st.error("❌ Error contacting prediction API")
