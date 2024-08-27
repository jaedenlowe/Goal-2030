import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

# Load the PyCaret models
models = [
    load_model('model_y1_tradkeeper'),
    load_model('model_y2_sweeperkeeper'),
    load_model('model_y3_ballplayingdefender'),
    load_model('model_y4_nononsensedefender'),
    load_model('model_y5_fullback'),
    load_model('model_y6_allactionmidfielder'),
    load_model('model_y7_midfieldplaymaker'),
    load_model('model_y8_traditionalwinger'),
    load_model('model_y9_invertedwinger'),
    load_model('model_y10_goalpoacher'),
    load_model('model_y11_targetman')
]

# Model names corresponding to their roles
model_names = [
    "Traditional Keeper",
    "Sweeper Keeper",
    "Ball-Playing Defender",
    "No-Nonsense Defender",
    "Full-Back",
    "All-Action Midfielder",
    "Midfield Playmaker",
    "Traditional Winger",
    "Inverted Winger",
    "Goal Poacher",
    "Target Man"
]

# Dictionary mapping model names to their score column names
score_column_map = {
    "Traditional Keeper": "y1_tradkeeper",
    "Sweeper Keeper": "y2_sweeperkeeper",
    "Ball-Playing Defender": "y3_ballplayingdefender",
    "No-Nonsense Defender": "y4_nononsensedefender",
    "Full-Back": "y5_fullback",
    "All-Action Midfielder": "y6_allactionmidfielder",
    "Midfield Playmaker": "y7_midfieldplaymaker",
    "Traditional Winger": "y8_traditionalwinger",
    "Inverted Winger": "y9_invertedwinger",
    "Goal Poacher": "y10_goalpoacher",
    "Target Man": "y11_targetman"
}

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

    # Make predictions for each model
    predictions = [predict_model(model, data=df) for model in models]

    # Display predictions
    st.write("Predictions:")

    # Create checkboxes for each model
    model_checkboxes = st.multiselect("Select Models:", model_names)

    # Create a slider for the prediction threshold
    threshold = st.slider("Prediction Threshold:", 0.0, 1.0, 0.5)

    # Create checkboxes for filtering by prediction_label
    show_recommended = st.checkbox("Show Recommended")
    show_not_recommended = st.checkbox("Show Not Recommended")

    for i, (prediction, model_name) in enumerate(zip(predictions, model_names)):
        if model_name in model_checkboxes:
            # Access the correct score column based on the model name
            score_column = score_column_map.get(model_name, "score")  # Default to "score" if not found

            # Filter predictions based on the threshold
            filtered_prediction = prediction[prediction['prediction_score'] >= threshold]

            # Filter predictions based on prediction_label
            if show_recommended and not show_not_recommended:
                filtered_prediction = filtered_prediction[filtered_prediction['prediction_label'] == 1]
            elif show_not_recommended and not show_recommended:
                filtered_prediction = filtered_prediction[filtered_prediction['prediction_label'] == 0]

            # Rename the 'prediction_label' column to 'Recommended'
            filtered_prediction.rename(columns={'prediction_label': 'Recommended'}, inplace=True)

            # Display model name and filtered prediction results
            st.header(f"{model_name}")
            st.write(filtered_prediction[['Player', score_column, 'Recommended', 'prediction_score']])