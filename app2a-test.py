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

# Function for squad generation with limit constraints
# Mapping of general positions to specific roles
position_to_roles_map = {
    "Goalkeeper": ["Traditional Keeper", "Sweeper Keeper"],
    "Defender": ["Ball-Playing Defender", "No-Nonsense Defender", "Full-Back"],
    "Midfielder": ["All-Action Midfielder", "Midfield Playmaker", "Traditional Winger", "Inverted Winger"],
    "Attacker": ["Goal Poacher", "Target Man"]
}

def generate_squad(prediction_results, num_players_per_position, total_squad_size):
    """Generates a squad based on prediction results, number of players per position, and selected roles with limits."""
    
    squad = []
    total_players_selected = 0
    selected_players = set()  # To track selected players and avoid duplicates

    for position, num_players in num_players_per_position.items():
        if total_players_selected >= total_squad_size:
            break
        
        # Get roles corresponding to the position (Goalkeeper, Defender, etc.)
        position_roles = position_to_roles_map[position]
        
        for role in position_roles:
            # Filter predictions for the role
            role_predictions = prediction_results[prediction_results['model_names'] == role]
            
            # Check if role_predictions is empty
            if role_predictions.empty:
                st.write(f"No predictions found for role: {role}")
                continue
            
            # Sort predictions by score
            role_predictions = role_predictions.sort_values(by='prediction_score', ascending=False)
            
            # Select top players based on the role limit
            num_players_to_add = min(num_players, len(role_predictions))
            top_players = role_predictions[:num_players_to_add]
            
            # Add players to squad if not already selected and within total squad size limit
            for _, player in top_players.iterrows():
                if total_players_selected >= total_squad_size:
                    break
                if player['Player'] not in selected_players:
                    selected_players.add(player['Player'])
                    squad.append(player)
                    total_players_selected += 1

            # If the squad size exceeds the limit, truncate the last batch of players
            if total_players_selected >= total_squad_size:
                break

    # Convert the squad list to a DataFrame
    if squad:
        final_squad = pd.DataFrame(squad)
    else:
        final_squad = pd.DataFrame()  # Return empty DataFrame if no players

    return final_squad

def display_squad(squad):
    """Displays the generated squad."""

    st.header("Generated Squad")
    st.write(f"Total Players: {len(squad)}")
    if squad.empty:
        st.write("No players found.")
    else:
        for index, player in squad.iterrows():
            st.write(f"- {player['Player']}: {player['model_names']} ({player['prediction_score']:.2f})")

# Streamlit app
st.title("Player Attribute Prediction and Squad Generation")

# File uploader widget
uploaded_file = st.file_uploader("Upload a CSV file with player attributes", type="csv")

if uploaded_file is not None:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)

    # Display the dataframe
    st.write("Uploaded DataFrame:")
    st.write(df.head())

    # Make predictions for each model
    predictions = []
    for model, model_name in zip(models, model_names):
        # Make prediction for the current model
        prediction = predict_model(model, data=df)
        # Add a new column to identify the model/role
        prediction['model_names'] = model_name
        predictions.append(prediction)

    # Combine predictions from all models
    combined_predictions = pd.concat(predictions, ignore_index=True)

    # Display predictions
    st.write("Predictions:")

    # Create checkboxes for each model
    model_checkboxes = st.multiselect("Select a Position/Role:", model_names)

    # Create a slider for the prediction threshold
    threshold = st.slider("Prediction Threshold:", 0.0, 1.0, 0.5)

    # Create checkboxes for filtering by prediction_label
    show_recommended = st.checkbox("Show Recommended")
    show_not_recommended = st.checkbox("Show Not Recommended")

    for model_name in model_names:
        if model_name in model_checkboxes:
            # Filter predictions for the selected model/role
            filtered_prediction = combined_predictions[combined_predictions['model_names'] == model_name]
            filtered_prediction = filtered_prediction[filtered_prediction['prediction_score'] >= threshold]

            # Filter predictions based on prediction_label
            if show_recommended and not show_not_recommended:
                filtered_prediction = filtered_prediction[filtered_prediction['prediction_label'] == 1]
            elif show_not_recommended and not show_recommended:
                filtered_prediction = filtered_prediction[filtered_prediction['prediction_label'] == 0]

            # Rename the 'prediction_label' column to 'Recommended' and convert values
            filtered_prediction['Recommended'] = filtered_prediction['prediction_label'].apply(lambda x: "Recommended" if x == 1 else "Not Recommended")
            filtered_prediction.drop('prediction_label', axis=1, inplace=True)

            # Display model name and filtered prediction results
            st.header(f"{model_name}")
            st.write(filtered_prediction[['Player', 'Recommended', 'prediction_score']])

    # Squad generation section
    st.subheader("Squad Generation")

    # Input for total squad size limit (e.g. 35 players)
    total_squad_size = st.number_input("Total Squad Size", min_value=0, max_value=35, value=35)

    # Input for number of players per position
    num_goalkeepers = st.number_input("Number of Goalkeepers", min_value=0, max_value=5, value=3)
    num_defenders = st.number_input("Number of Defenders", min_value=0, max_value=10, value=8)
    num_midfielders = st.number_input("Number of Midfielders", min_value=0, max_value=10, value=4)
    num_attackers = st.number_input("Number of Attackers", min_value=0, max_value=5, value=3)

    # Input for number of each role per position
    st.write("**Goalkeeper Roles:**")
    num_traditional_keepers = st.number_input("Traditional Keepers", min_value=0, max_value=num_goalkeepers, value=2)
    num_sweeper_keepers = st.number_input("Sweeper Keepers", min_value=0, max_value=num_goalkeepers, value=1)
    st.write("**Defender Roles:**")
    num_ball_playing_defenders = st.number_input("Ball-Playing Defenders", min_value=0, max_value=num_defenders, value=2)
    num_no_nonsense_defenders = st.number_input("No-Nonsense Defenders", min_value=0, max_value=num_defenders, value=2)
    num_fullbacks = st.number_input("Full-Backs", min_value=0, max_value=num_defenders, value=4)
    st.write("**Midfielder Roles:**")
    num_all_action_midfielders = st.number_input("All-Action Midfielders", min_value=0, max_value=num_midfielders, value=2)
    num_midfield_playmakers = st.number_input("Midfield Playmakers", min_value=0, max_value=num_midfielders, value=2)
    num_traditional_wingers = st.number_input("Traditional Wingers", min_value=0, max_value=num_midfielders, value=2)
    num_inverted_wingers = st.number_input("Inverted Wingers", min_value=0, max_value=num_midfielders, value=2)
    st.write("**Attacker Roles:**")
    num_goal_poachers = st.number_input("Goal Poachers", min_value=0, max_value=num_attackers, value=2)
    num_target_men = st.number_input("Target Men", min_value=0, max_value=num_attackers, value=2)

    # Create a dictionary to store the number of players per position
    num_players_per_position = {
        "Goalkeeper": num_traditional_keepers + num_sweeper_keepers,
        "Defender": num_ball_playing_defenders + num_no_nonsense_defenders + num_fullbacks,
        "Midfielder": num_all_action_midfielders + num_midfield_playmakers + num_traditional_wingers + num_inverted_wingers,
        "Attacker": num_goal_poachers + num_target_men
    }

    # **Add the Generate Squad button**
    if st.button("Generate Squad"):
        # Generate the squad using the generate_squad function
        squad = generate_squad(combined_predictions, num_players_per_position, total_squad_size)

        # Display the generated squad
        display_squad(squad)
    
    # Debugging information
    st.write("Debugging Info:")
    st.write(f"Total Squad Size: {total_squad_size}")
    st.write(f"Players per Position: {num_players_per_position}")
