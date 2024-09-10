import streamlit as st
import pandas as pd
import numpy as np
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
    model_checkboxes = st.multiselect("Select a Position/Role:", model_names)

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

            # Rename the 'prediction_label' column to 'Recommended' and convert values
            filtered_prediction['Recommended'] = filtered_prediction['prediction_label'].apply(lambda x: "Recommended" if x == 1 else "Not Recommended")
            filtered_prediction.drop('prediction_label', axis=1, inplace=True)

            # Rename the score column to the corresponding model name
            filtered_prediction.rename(columns={score_column: model_name}, inplace=True)

            # Display model name and filtered prediction results
            st.header(f"{model_name}")
            st.write(filtered_prediction[['Player', model_name, 'Recommended', 'prediction_score']])

# Start Streamlit app
st.title("Squad Generation")

# 1. Squad size input from user
max_squad_size = 35
st.subheader("Select the number of players for each position")

n_goalkeepers = st.number_input("Number of Goalkeepers", min_value=0, max_value=3, value=1)
n_defenders = st.number_input("Number of Defenders", min_value=0, max_value=10, value=4)
n_midfielders = st.number_input("Number of Midfielders", min_value=0, max_value=10, value=5)
n_attackers = st.number_input("Number of Attackers", min_value=0, max_value=10, value=3)

total_players = n_goalkeepers + n_defenders + n_midfielders + n_attackers
if total_players > max_squad_size:
    st.error(f"Total players cannot exceed {max_squad_size}. Currently selected: {total_players}")
    st.stop()

# 2. Role breakdown for each position
st.subheader("Goalkeeper Roles:")
traditional_keepers = st.number_input("Traditional Keepers", min_value=0, max_value=n_goalkeepers, value=1)
sweeper_keepers = st.number_input("Sweeper Keepers", min_value=0, max_value=n_goalkeepers - traditional_keepers, value=0)

st.subheader("Defender Roles:")
ball_playing_defenders = st.number_input("Ball-Playing Defenders", min_value=0, max_value=n_defenders, value=2)
no_nonsense_defenders = st.number_input("No-Nonsense Defenders", min_value=0, max_value=n_defenders - ball_playing_defenders, value=2)


# Repeat similar blocks for Midfielders and Attackers roles based on your model outputs.

# 3. Squad Selection Logic
def select_players(role, count):
    """
    Select top players for a given role. This function will fetch 'count' number
    of top-performing players for the specified role based on the predictions.
    """
    if count == 0:
        return pd.DataFrame()  # Return an empty DataFrame if no players are needed for this role
    
    # Sort players based on the score in the specified role and select the top 'count' players
    sorted_players = df_predictions[['Player', role]].sort_values(by=role, ascending=False).head(count)
    return sorted_players[['Player', role]]

# Example role-wise player selection
goalkeepers_selected = pd.concat([
    select_players('Traditional Keeper', traditional_keepers),
    select_players('Sweeper Keeper', sweeper_keepers)
])

defenders_selected = pd.concat([
    select_players('Ball-Playing Defender', ball_playing_defenders),
    select_players('No-Nonsense Defender', no_nonsense_defenders)
])

# Similarly, you can add selections for Midfielders and Attackers

# 4. Handling players who qualify for multiple roles
def resolve_conflicts(selected_players):
    """
    This function ensures that no player is assigned to multiple roles.
    Players with higher scores in one role will be prioritized, and other roles will get the next best players.
    """
    resolved_players = selected_players.drop_duplicates(subset=['Player'], keep='first')
    return resolved_players

# Resolve conflicts in the selected players
final_goalkeepers = resolve_conflicts(goalkeepers_selected)
final_defenders = resolve_conflicts(defenders_selected)
# Apply the same for midfielders and attackers

# 5. Display the final squad selection as a table
st.subheader("Final Squad Selection")

# Example squad display for goalkeepers and defenders
squad_table = pd.concat([final_goalkeepers, final_defenders])  # Add more positions as needed
st.dataframe(squad_table)

