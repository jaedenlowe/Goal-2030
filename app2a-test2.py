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

# Function to display the predictions for a given role
def display_role_predictions(df, role):
    score_column = score_column_map.get(role)
    if score_column:
        st.write(f"## {role} Predictions")
        st.dataframe(df[[score_column]].sort_values(by=score_column, ascending=False))
    else:
        st.write(f"Role {role} does not have an associated score column.")

# Additional functions for squad generation
def generate_squad(prediction_results, num_players_per_position, selected_roles):
    """Generates a squad based on prediction results, number of players per position, and selected roles."""
    squad = []
    for position, num_players in num_players_per_position.items():
        roles = selected_roles.get(position, [])
        for role in roles:
            score_column = score_column_map.get(role)

            if score_column:
                position_predictions = [result for result in prediction_results if score_column in result]

                # Sort predictions by score in descending order
                position_predictions = sorted(position_predictions, key=lambda x: x[score_column], reverse=True)

                # Select the top N players based on the sorted predictions
                squad.append(position_predictions[:num_players])
            else:
                st.write(f"Warning: No score column found for role {role}. Skipping this role.")
    
    return squad

# Display functions
def display_squad(squad):
    """Displays the generated squad using Streamlit."""
    st.write("## Generated Squad")
    for position in squad:
        st.write(position)

# CSV Upload Section
uploaded_file = st.file_uploader("Upload your predictions CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("## Displaying CSV Data")
    st.dataframe(df.head())

    # Display predictions for each role
    for role in model_names:
        display_role_predictions(df, role)

    # Squad generation feature
    st.write("## Define squad roles and number of players for each position")
    num_goalkeepers = st.number_input("Goalkeepers", min_value=0, value=1)
    num_defenders = st.number_input("Defenders", min_value=0, value=4)
    num_midfielders = st.number_input("Midfielders", min_value=0, value=4)
    num_attackers = st.number_input("Attackers", min_value=0, value=2)

    st.write("**Goalkeeper Roles:**")
    num_traditional_keepers = st.number_input("Traditional Keepers", min_value=0, max_value=num_goalkeepers, value=1)
    num_sweeper_keepers = st.number_input("Sweeper Keepers", min_value=0, max_value=num_goalkeepers, value=0)
    st.write("**Defender Roles:**")
    num_ball_playing_defenders = st.number_input("Ball-Playing Defenders", min_value=0, max_value=num_defenders, value=2)
    num_no_nonsense_defenders = st.number_input("No-Nonsense Defenders", min_value=0, max_value=num_defenders, value=1)
    num_fullbacks = st.number_input("Full-Backs", min_value=0, max_value=num_defenders, value=1)
    st.write("**Midfielder Roles:**")
    num_all_action_midfielders = st.number_input("All-Action Midfielders", min_value=0, max_value=num_midfielders, value=2)
    num_midfield_playmakers = st.number_input("Midfield Playmakers", min_value=0, max_value=num_midfielders, value=2)
    num_traditional_wingers = st.number_input("Traditional Wingers", min_value=0, max_value=num_midfielders, value=1)
    num_inverted_wingers = st.number_input("Inverted Wingers", min_value=0, max_value=num_midfielders, value=0)
    st.write("**Attacker Roles:**")
    num_goal_poachers = st.number_input("Goal Poachers", min_value=0, max_value=num_attackers, value=2)
    num_target_men = st.number_input("Target Men", min_value=0, max_value=num_attackers, value=1)

    if st.button("Generate Squad"):
        num_players_per_position = {
            "Goalkeeper": num_traditional_keepers + num_sweeper_keepers,
            "Defender": num_ball_playing_defenders + num_no_nonsense_defenders + num_fullbacks,
            "Midfielder": num_all_action_midfielders + num_midfield_playmakers + num_traditional_wingers + num_inverted_wingers,
            "Attacker": num_goal_poachers + num_target_men
        }

        selected_roles = {
            "Goalkeeper": ["Traditional Keeper"] * num_traditional_keepers + ["Sweeper Keeper"] * num_sweeper_keepers,
            "Defender": ["Ball-Playing Defender"] * num_ball_playing_defenders + ["No-Nonsense Defender"] * num_no_nonsense_defenders + ["Full-Back"] * num_fullbacks,
            "Midfielder": ["All-Action Midfielder"] * num_all_action_midfielders + ["Midfield Playmaker"] * num_midfield_playmakers + ["Traditional Winger"] * num_traditional_wingers + ["Inverted Winger"] * num_inverted_wingers,
            "Attacker": ["Goal Poacher"] * num_goal_poachers + ["Target Man"] * num_target_men
        }

        # Generate the squad using the generate_squad function
        squad = generate_squad(df, num_players_per_position, selected_roles)

        # Display the generated squad
        display_squad(squad)
