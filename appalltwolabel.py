import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

# Load the trained PyCaret model
model = load_model('model_Ball Playing Defender')  # Replace with your actual model file path

# Function to make predictions
def predict(input_df):
    required_columns = ['x1_one_on_one', 'x2_goals', 'x3_passing_accuracy', 'x4_vision',
                        'x5_tackles', 'x6_strength', 'x7_speed', 'x8_aerial_duels']

    if not all(col in input_df.columns for col in required_columns):
        st.error("Please provide all the required features for prediction.")
        return None

    prediction_df = input_df[required_columns]
    predictions = predict_model(model, data=prediction_df)
    prediction = predictions['prediction_label'][0]  # Extract the first prediction

    # Map prediction to text
    if prediction == 1:
        prediction_text = "Ball Playing Defender"
    else:
        prediction_text = "Maybe not too good"

    return prediction_text

# Streamlit App
st.title('PyCaret Model Prediction App')

# Sidebar for user input features
st.sidebar.header('Input Features')

def user_input_features():
    x1_one_on_one = st.sidebar.number_input('x1_One on One', min_value=0, max_value=100, value=50)
    x2_goals = st.sidebar.number_input('x2_Goals', min_value=0, max_value=100, value=50)
    x3_passing_accuracy = st.sidebar.number_input('x3_Passing Accuracy', min_value=0, max_value=100, value=50)
    x4_vision = st.sidebar.number_input('x4_Vision', min_value=0, max_value=100, value=50)
    x5_tackles = st.sidebar.number_input('x5_Tackles', min_value=0, max_value=100, value=50)
    x6_strength = st.sidebar.number_input('x6_Strength', min_value=0, max_value=100, value=50)
    x7_speed = st.sidebar.number_input('x7_Speed', min_value=0, max_value=100, value=50)
    x8_aerial_duels = st.sidebar.number_input('x8_Aerial Duels', min_value=0, max_value=100, value=50)

    data = {
        'x1_one_on_one': x1_one_on_one,
        'x2_goals': x2_goals,
        'x3_passing_accuracy': x3_passing_accuracy,
        'x4_vision': x4_vision,
        'x5_tackles': x5_tackles,
        'x6_strength': x6_strength,
        'x7_speed': x7_speed,
        'x8_aerial_duels': x8_aerial_duels
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
    if prediction is not None:
        st.subheader('Prediction')
        st.write(prediction)

# Additional information (optional)
st.subheader('About the Model')
st.write("""
This app uses a machine learning model built with PyCaret to make predictions.
The model was trained on historical data and can predict the outcome based on the input features.
""")