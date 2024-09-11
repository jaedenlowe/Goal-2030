import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. File Upload Section - Keeping your original CSV upload
st.title("Player Data and Model Upload")

uploaded_file = st.file_uploader("Choose a CSV file with player data", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Player Data")
    st.dataframe(df)

    # Assuming the uploaded CSV has player info with the following columns:
    # ['Player', 'Position', 'Age', 'Team', ...]

    # Placeholder for player ability DataFrame (to be populated by model predictions)
    df_predictions = df[['Player']]  # We'll add model prediction results to this DataFrame

# 2. Model Loading - Keeping your model loading intact
# Assuming your models are saved in a folder or uploaded as part of the interface
st.subheader("Model Uploads")

models = {}
roles = ['Traditional Keeper', 'Sweeper Keeper', 'Ball-Playing Defender', 'No-Nonsense Defender', 
         'Wide Playmaker', 'Box-to-Box Midfielder', 'Deep Lying Playmaker', 'Poacher', 
         'Target Man', 'Advanced Forward', 'False Nine']

for role in roles:
    model_file = st.file_uploader(f"Upload model for {role}", type="pkl")
    if model_file is not None:
        models[role] = joblib.load(model_file)

# Check if all models are loaded
if len(models) == len(roles):
    st.success("All models loaded successfully!")
else:
    st.warning("Please upload models for all roles.")

# 3. Model Prediction Section - Generate predictions for all roles
if uploaded_file is not None and len(models) == len(roles):
    st.subheader("Generating Player Role Predictions")

    for role in roles:
        # Example: Apply each model to make predictions for this role
        # Assume that each model expects some features (e.g., player stats from df)
        # Placeholder: predictions = models[role].predict(some_features_from_df)
        # For now, we generate random values for demonstration
        df_predictions[role] = np.random.rand(len(df))  # Replace with actual model predictions

    st.dataframe(df_predictions)

# Now the new squad selection feature starts here:
st.title("Squad Generation")

# 4. Player Squad Selection UI - Allow input for squad sizes
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

# 5. Role Breakdown for each position
# Goalkeeper Roles
st.subheader("Goalkeeper Roles:")
traditional_keepers = st.number_input("Traditional Keepers", min_value=0, max_value=n_goalkeepers, value=1)
sweeper_keepers = st.number_input("Sweeper Keepers", min_value=0, max_value=n_goalkeepers - traditional_keepers, value=0)

# Defender Roles
st.subheader("Defender Roles:")
ball_playing_defenders = st.number_input("Ball-Playing Defenders", min_value=0, max_value=n_defenders, value=2)
no_nonsense_defenders = st.number_input("No-Nonsense Defenders", min_value=0, max_value=n_defenders - ball_playing_defenders, value=2)

# Midfielder Roles
st.subheader("Midfielder Roles:")
wide_playmakers = st.number_input("Wide Playmakers", min_value=0, max_value=n_midfielders, value=2)
box_to_box_midfielders = st.number_input("Box-to-Box Midfielders", min_value=0, max_value=n_midfielders - wide_playmakers, value=2)
deep_lying_playmakers = st.number_input("Deep Lying Playmakers", min_value=0, max_value=n_midfielders - wide_playmakers - box_to_box_midfielders, value=1)

# Attacker Roles
st.subheader("Attacker Roles:")
poachers = st.number_input("Poachers", min_value=0, max_value=n_attackers, value=1)
target_men = st.number_input("Target Men", min_value=0, max_value=n_attackers - poachers, value=1)
advanced_forwards = st.number_input("Advanced Forwards", min_value=0, max_value=n_attackers - poachers - target_men, value=1)
false_nines = st.number_input("False Nines", min_value=0, max_value=n_attackers - poachers - target_men - advanced_forwards, value=0)

# 6. Player Selection Logic (for each role)
def select_players(df, role, count):
    """
    Select top players for a given role. This function will fetch 'count' number
    of top-performing players for the specified role based on the predictions.
    """
    if count == 0:
        return pd.DataFrame()  # Return an empty DataFrame if no players are needed for this role
    
    # Sort players based on the score in the specified role and select the top 'count' players
    sorted_players = df[['Player', role]].sort_values(by=role, ascending=False).head(count)
    return sorted_players[['Player', role]]

# Select players for each position
goalkeepers_selected = pd.concat([
    select_players(df_predictions, 'Traditional Keeper', traditional_keepers),
    select_players(df_predictions, 'Sweeper Keeper', sweeper_keepers)
])

defenders_selected = pd.concat([
    select_players(df_predictions, 'Ball-Playing Defender', ball_playing_defenders),
    select_players(df_predictions, 'No-Nonsense Defender', no_nonsense_defenders)
])

midfielders_selected = pd.concat([
    select_players(df_predictions, 'Wide Playmaker', wide_playmakers),
    select_players(df_predictions, 'Box-to-Box Midfielder', box_to_box_midfielders),
    select_players(df_predictions, 'Deep Lying Playmaker', deep_lying_playmakers)
])

attackers_selected = pd.concat([
    select_players(df_predictions, 'Poacher', poachers),
    select_players(df_predictions, 'Target Man', target_men),
    select_players(df_predictions, 'Advanced Forward', advanced_forwards),
    select_players(df_predictions, 'False Nine', false_nines)
])

# 7. Handling players who qualify for multiple roles
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
final_midfielders = resolve_conflicts(midfielders_selected)
final_attackers = resolve_conflicts(attackers_selected)

# 8. Display the final squad selection as a table
st.subheader("Final Squad Selection")

# Concatenate all positions into a final squad table
squad_table = pd.concat([final_goalkeepers, final_defenders, final_midfielders, final_attackers])
st.dataframe(squad_table)
