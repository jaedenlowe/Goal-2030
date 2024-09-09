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
    model_checkboxes = st.multiselect("Select a Position/Role:", model_names)

    # Create a slider for the prediction threshold
    threshold = st.slider("Prediction Threshold:", 0.0, 1.0, 0.5)

    # Create checkboxes for filtering by prediction_label
    show_recommended = st.checkbox("Show Recommended")
    show_not_recommended = st.checkbox("Show Not Recommended")

    # Additional input fields
    goalkeepers_needed = st.number_input("Goalkeepers Needed", min_value=0)
    defenders_needed = st.number_input("Defenders Needed", min_value=0)
    midfielders_needed = st.number_input("Midfielders Needed", min_value=0)
    attackers_needed = st.number_input("Attackers Needed", min_value=0)

    # Role-based inputs (example for Goalkeepers)
    traditional_keepers_needed = st.number_input("Traditional Keepers Needed", min_value=0)
    sweeper_keepers_needed = st.number_input("Sweeper Keepers Needed", min_value=0)

    # ... other role-based inputs for all roles in model_names
    ball_playing_defenders_needed = st.number_input("Ball-Playing Defenders Needed", min_value=0)
    no_nonsense_defenders_needed = st.number_input("No-Nonsense Defenders Needed", min_value=0)
    full_backs_needed = st.number_input("Full-Backs Needed", min_value=0)
    all_action_midfielders_needed = st.number_input("All-Action Midfielders Needed", min_value=0)
    midfield_playmakers_needed = st.number_input("Midfield Playmakers Needed", min_value=0)
    traditional_wingers_needed = st.number_input("Traditional Wingers Needed", min_value=0)
    inverted_wingers_needed = st.number_input("Inverted Wingers Needed", min_value=0)
    goal_poachers_needed = st.number_input("Goal Poachers Needed", min_value=0)
    target_men_needed = st.number_input("Target Men Needed", min_value=0)

    # Squad generation logic
    def generate_squad(predictions, roles_needed, positions):
        squad = []
        for role, needed in roles_needed.items():
            filtered_predictions = []
            for prediction_df in predictions:
                if prediction_df['prediction_label'].isin([role]).any():
                    filtered_predictions.append(prediction_df.copy())
            # Check for empty list and handle accordingly
            if filtered_predictions:
                sorted_predictions = pd.concat(filtered_predictions, ignore_index=True).sort_values(by=role, ascending=False)
                squad.extend(sorted_predictions[:needed])
            else:
                # No predictions for this role - handle appropriately
                st.warning(f"No predictions found for role: {role}")
        return pd.concat(squad)

    # ... (existing code)

    # Generate and display squad
    if st.button("Generate Squad") and any(count > 0 for count in roles_needed.values()):
        roles_needed = {
            "Traditional Keeper": traditional_keepers_needed,
            "Sweeper Keeper": sweeper_keepers_needed,
            "Ball-Playing Defender": ball_playing_defenders_needed,
            "No-Nonsense Defender": no_nonsense_defenders_needed,
            "Full-Back": full_backs_needed,
            "All-Action Midfielder": all_action_midfielders_needed,
            "Midfield Playmaker": midfield_playmakers_needed,
            "Traditional Winger": traditional_wingers_needed,
            "Inverted Winger": inverted_wingers_needed,
            "Goal Poacher": goal_poachers_needed,
            "Target Man": target_men_needed
        }
        squad = generate_squad(predictions, roles_needed, positions_needed)
        st.write("Recommended Squad:")
        st.write(squad)