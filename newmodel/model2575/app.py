import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

# Load the PyCaret model (make sure to replace 'model_name' with your actual model name)
model1 = load_model('model_y1_tradkeeper.1')
model2 = load_model('model_y2_sweeperkeeper.1')
model3 = load_model('model_y3_ballplayingdefender.1')
model4 = load_model('model_y4_nononsensedefender.1')
model5 = load_model('model_y5_fullback.1')
model6 = load_model('model_y6_allactionmidfielder.1')
model7 = load_model('model_y7_midfieldplaymaker.1')
model8 = load_model('model_y8_traditionalwinger.1')
model9 = load_model('model_y9_invertedwinger.1')
model10 = load_model('model_y10_goalpoacher.1')
model11 = load_model('model_y11_targetman.1')


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