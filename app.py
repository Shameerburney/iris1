import numpy as np
import streamlit as st
import pickle

# Load the model
model = pickle.load(open("model.pkl", "rb"))

# Set page configuration
st.set_page_config(
    page_title="Flower Class Prediction",
    page_icon="🌸",
    layout="centered"
)

# Title
st.title("🌸 Flower Class Prediction")
st.write("Enter the flower measurements to predict the species")

# Create input fields
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input(
        "Sepal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.1,
        format="%.1f"
    )
    
    petal_length = st.number_input(
        "Petal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=3.0,
        step=0.1,
        format="%.1f"
    )

with col2:
    sepal_width = st.number_input(
        "Sepal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=3.0,
        step=0.1,
        format="%.1f"
    )
    
    petal_width = st.number_input(
        "Petal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1,
        format="%.1f"
    )

# Predict button
if st.button("🔮 Predict Flower Species", type="primary"):
    try:
        # Create feature array
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Display result
        st.success(f"**The Flower Species is: {prediction}**")
        
        # Add some visual feedback
        st.balloons()
        
    except Exception as e:
        st.error(f"Error occurred: {e}")

# Add information section
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
