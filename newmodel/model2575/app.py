import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

# Load the PyCaret model (make sure to replace 'model_name' with your actual model name)
model = load_model('model_results2575_withkeepers.pkl')

# Streamlit app
st.title("Player Attribute Prediction")

# File uploader widget
uploaded_file = st.file_uploader("Upload a CSV file with player attributes", type="csv")

if uploaded_file is not None:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)
    
    # Display the dataframe
    st.write("Uploaded DataFrame:")
    st.write(df.head())
    
    # Make predictions
    predictions = predict_model(model, data=df)
    
    # Display predictions
    st.write("Predictions:")
    st.write(predictions)
    
    # Optionally, you can display just the prediction column
    st.write("Predicted Labels:")
    st.write(predictions['Label'])