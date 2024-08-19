import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

# Load the trained PyCaret models (replace with your actual model file paths)
model1 = load_model('model_Ball Playing Defender')
model2 = load_model('model_Box to Box Midfielder')
model3 = load_model('model_Goal Poacher')
model4 = load_model('model_Midfield Playmaker')
model5 = load_model('model_No Nonsense Defender')
model6 = load_model('model_Sweeper Keeper')
model7 = load_model('model_Target Man')
model8 = load_model('model_Traditional Goalkeeper')

# ... and so on for other models

# Create a mapping of models to target columns
model_map = {
    'target1': model1,
    'target2': model2,
    'target3': model3,
    'target4': model4,
    'target5': model5,
    'target6': model6,
    'target7': model7,
    'target8': model8,
    # ... add other models and targets
}

# Function to make predictions
def predict(input_df):
    required_columns = ['x1_one_on_one', 'x2_goals', 'x3_passing_accuracy', 'x4_vision',
                        'x5_tackles', 'x6_strength', 'x7_speed', 'x8_aerial_duels']

    if not all(col in input_df.columns for col in required_columns):
        st.error("Please provide all the required features for prediction.")
        return None

    prediction_df = input_df[required_columns]

    predictions = {}
    for target, model in model_map.items():
        predictions[target] = predict_model(model, data=prediction_df)['prediction_label'][0]

    return predictions

# Streamlit App
st.title('PyCaret Multi-Model Prediction App')

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
    predictions = predict(input_df)
    if predictions:
        for target, prediction in predictions.items():
            st.write(f"Prediction for {target}: {prediction}")

# Additional information (optional)
st.subheader('About the Model')
st.write("""
This app uses multiple machine learning models built with PyCaret to make predictions.
The models were trained on historical data and can predict different outcomes based on the input features.
""")