import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model  # Change to pycaret.regression if it's a regression model

# Load the trained PyCaret model
model = load_model('bestmodel.pkl')  # Replace 'your_model_file' with your actual model file name

# Function to make predictions
def predict(input_df):
    predictions = predict_model(model, data=input_df)
    return predictions['Label']

# Streamlit App
st.title('PyCaret Model Prediction App')

# Sidebar for user input features
st.sidebar.header('Input Features')

# Example: Define the input features for your model
def user_input_features():
    feature1 = st.sidebar.number_input('Feature 1', min_value=0, max_value=100, value=50)
    feature2 = st.sidebar.number_input('Feature 2', min_value=0, max_value=100, value=50)
    feature3 = st.sidebar.number_input('Feature 3', min_value=0.0, max_value=10.0, value=5.0)
    data = {
        'Feature1': feature1,
        'Feature2': feature2,
        'Feature3': feature3,
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display the input features
st.subheader('User Input Features')
st.write(input_df)

# Predict and display the results
if st.button('Predict'):
    prediction = predict(input_df)
    st.subheader('Prediction')
    st.write(prediction[0])

# Additional information (optional)
st.subheader('About the Model')
st.write("""
This app uses a machine learning model built with PyCaret to make predictions.
The model was trained on historical data and can predict the outcome based on the input features.
""")
