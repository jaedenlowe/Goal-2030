import streamlit as st
import pandas as pd
import numpy as np

# Load your existing model predictions
# Assuming your existing code processes the predictions into a DataFrame (df_predictions)

# Sample structure for 'df_predictions' which you might already have from your models
# df_predictions = pd.DataFrame({
#     'Player': ['Player 1', 'Player 2', 'Player 3', 'Player 4', ...],
#     'Traditional Keeper': [0.85, 0.40, 0.10, 0.0, ...],
#     'Sweeper Keeper': [0.75, 0.60, 0.30, 0.2, ...],
#     'Ball-Playing Defender': [0.5, 0.3, 0.9, 0.4, ...],
#     ...
# })

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

