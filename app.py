import numpy as np
import streamlit as st
import pickle

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Page config
st.set_page_config(
    page_title="Flower Class Prediction",
    page_icon="🌸",
    layout="centered"
)

# Title
st.title("🌸 Flower Class Prediction")
st.write("Enter the flower measurements to predict the species")

# Input fields
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal Length (cm)", 0.0, 10.0, 5.0, step=0.1)
    petal_length = st.number_input("Petal Length (cm)", 0.0, 10.0, 3.0, step=0.1)

with col2:
    sepal_width = st.number_input("Sepal Width (cm)", 0.0, 10.0, 3.0, step=0.1)
    petal_width = st.number_input("Petal Width (cm)", 0.0, 10.0, 1.0, step=0.1)

# Predict button
if st.button("🔮 Predict Flower Species", type="primary"):
    try:
        # Create feature array
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Scale features (this line is essential)
        scaled_features = scaler.transform(features)

        # Predict
        prediction = model.predict(scaled_features)[0]

        st.success(f"🌼 The Flower Species is: **{prediction}**")
        st.snow()

    except Exception as e:
        st.error(f"Error occurred: {e}")

# About section
with st.expander("ℹ️ About this App"):
    st.write("""
    This application uses a Random Forest Classifier trained on the Iris dataset 
    to predict flower species based on their measurements.

    **Features:**
    - Sepal Length: Length of the sepal in centimeters
    - Sepal Width: Width of the sepal in centimeters
    - Petal Length: Length of the petal in centimeters
    - Petal Width: Width of the petal in centimeters

    **Possible Species:**
    - Setosa
    - Versicolor
    - Virginica
    """)
