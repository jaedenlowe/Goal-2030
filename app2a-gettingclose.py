import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

# Load the PyCaret models
def load_all_models():
    try:
        models = {
            "Traditional Keeper": load_model('model_y1_tradkeeper'),
            "Sweeper Keeper": load_model('model_y2_sweeperkeeper'),
            "Ball-Playing Defender": load_model('model_y3_ballplayingdefender'),
            "No-Nonsense Defender": load_model('model_y4_nononsensedefender'),
            "Full-Back": load_model('model_y5_fullback'),
            "All-Action Midfielder": load_model('model_y6_allactionmidfielder'),
            "Midfield Playmaker": load_model('model_y7_midfieldplaymaker'),
            "Traditional Winger": load_model('model_y8_traditionalwinger'),
            "Inverted Winger": load_model('model_y9_invertedwinger'),
            "Goal Poacher": load_model('model_y10_goalpoacher'),
            "Target Man": load_model('model_y11_targetman')
        }
        st.write("Models loaded successfully.")
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return {}

models = load_all_models()

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

def generate_squad(prediction_results, num_players_per_position):
    # Initialize a set to store the selected players and their roles
    selected_players = set()
    squad = []

    # Create a temporary DataFrame to store each player's score for all roles
    all_scores = pd.DataFrame()

    for role in score_column_map.keys():
        role_predictions = prediction_results[prediction_results['model_names'] == role]

        # Access the correct score column based on the role
        score_column = score_column_map.get(role, "prediction_score")
        role_predictions = role_predictions[['Player', score_column]]
        role_predictions.rename(columns={score_column: role}, inplace=True)

        # Merge the role predictions into the all_scores DataFrame
        if all_scores.empty:
            all_scores = role_predictions
        else:
            all_scores = pd.merge(all_scores, role_predictions, on='Player', how='outer')

    # Convert 'Player' column from categorical to string type (if applicable)
    if pd.api.types.is_categorical_dtype(all_scores['Player']):
        all_scores['Player'] = all_scores['Player'].astype(str)

    # Fill NaN values with 0 (for players who don't have scores for certain roles)
    all_scores.fillna(0, inplace=True)

    # Add a column to track the best score and the role associated with it
    all_scores['Best Score'] = all_scores.drop('Player', axis=1).max(axis=1)
    all_scores['Best Role'] = all_scores.drop(['Player', 'Best Score'], axis=1).idxmax(axis=1)

    # Sort players by their best score, ensuring the highest-scoring players are considered first
    sorted_scores = all_scores.sort_values(by='Best Score', ascending=False)

    # Initialize a total count of players
    total_selected_players = 0

    # Allocate players based on their highest score for each role
    for role, num_players in num_players_per_position.items():
        print(f"\nSelecting {num_players} players for role: {role}")

        # Filter players for the current role
        role_players = sorted_scores[
            (sorted_scores['Best Role'] == role) & (~sorted_scores['Player'].isin(selected_players))
        ]

        # If not enough players are found for the role, try using fallback players
        if len(role_players) < num_players:
            st.write(f"Warning: Not enough players available for role: {role}. Needed {num_players}, found {len(role_players)}.")
            
            # Find additional players who have lower scores in this role but can fill it
            fallback_players = sorted_scores[
                (~sorted_scores['Player'].isin(selected_players)) & (sorted_scores[role] > 0)
            ].sort_values(by=role, ascending=False)

            # Calculate how many more players we need to fill the role
            remaining_players_needed = num_players - len(role_players)

            # Add fallback players to the role_players DataFrame
            role_players = pd.concat([role_players, fallback_players.head(remaining_players_needed)])

        # Select the top players for the current role, up to the specified number
        top_players = role_players.head(num_players)

        # Add the selected players to the set of selected players
        selected_players.update(top_players['Player'])

        # Append the top players for this role to the squad list
        squad.append(top_players[['Player', 'Best Role', 'Best Score']])

        # Track the total number of players selected
        total_selected_players += len(top_players)

        # Output the number of players selected for this role for debugging
        print(f"Selected {len(top_players)} players for role {role}. Total so far: {total_selected_players}")

    # Concatenate all selected players into a final DataFrame
    final_squad = pd.concat(squad, ignore_index=True) if squad else pd.DataFrame()

    # Check if the total number of selected players matches the user's input
    total_players_needed = sum(num_players_per_position.values())
    if total_selected_players != total_players_needed:
        st.write(f"Warning: Squad selection does not meet the required number of players. Selected {total_selected_players}, needed {total_players_needed}.")

    # Return the final squad DataFrame
    return final_squad








def display_squad(squad):
    """Displays the generated squad in a formatted table with demarcations."""
    st.header("Generated Squad")
    
    if squad.empty:
        st.write("No players found.")
        return
    
    # Create a DataFrame for displaying
    squad_data = []
    position_types = {
        "Goalkeeper": ["Traditional Keeper", "Sweeper Keeper"],
        "Defender": ["Ball-Playing Defender", "No-Nonsense Defender", "Full-Back"],
        "Midfielder": ["All-Action Midfielder", "Midfield Playmaker", "Traditional Winger", "Inverted Winger"],
        "Attacker": ["Goal Poacher", "Target Man"]
    }
    
    for position, roles in position_types.items():
        # Add a demarcation for the position type
        squad_data.append(["", "", position, ""])
        
        # Filter players by role
        for role in roles:
            role_players = squad[squad['Best Role'] == role]
            
            if role_players.empty:
                continue
            
            for _, player in role_players.iterrows():
                squad_data.append([player['Player'], position, role, f"{player['Best Score']:.2f}"])
    
    # Convert the list to a DataFrame
    squad_df = pd.DataFrame(squad_data, columns=["Player Name", "Position", "Role", "Score"])
    
    # Display the DataFrame
    st.write(squad_df)




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
    for model_name, model in models.items():
        if model is not None:
            # Make prediction for the current model
            prediction = predict_model(model, data=df)
            # Add a new column to identify the model/role
            prediction['model_names'] = model_name
            predictions.append(prediction)
        else:
            st.write(f"Model not found for {model_name}")

    if predictions:
        # Combine predictions from all models
        combined_predictions = pd.concat(predictions, ignore_index=True)

        # Display predictions
        st.write("Predictions:")

        # Create checkboxes for each model
        model_checkboxes = st.multiselect("Select a Position/Role:", list(models.keys()))

        # Create a slider for the prediction threshold
        threshold = st.slider("Prediction Threshold:", 0.0, 1.0, 0.5)

        # Create checkboxes for filtering by prediction_label
        show_recommended = st.checkbox("Show Recommended")
        show_not_recommended = st.checkbox("Show Not Recommended")

        for model_name in models.keys():
            if model_name in model_checkboxes:
                # Access the correct score column based on the model name
                score_column = score_column_map.get(model_name, "prediction_score")

                # Filter predictions for the selected model/role
                filtered_prediction = combined_predictions[combined_predictions['model_names'] == model_name]
                filtered_prediction = filtered_prediction[filtered_prediction[score_column] >= threshold]

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

        # Squad generation section
        st.subheader("Squad Generation")

        # Input for number of players per position
        num_goalkeepers = st.number_input("Number of Goalkeepers", min_value=0, max_value=5, value=1)
        num_defenders = st.number_input("Number of Defenders", min_value=0, max_value=10, value=4)
        num_midfielders = st.number_input("Number of Midfielders", min_value=0, max_value=10, value=5)
        num_attackers = st.number_input("Number of Attackers", min_value=0, max_value=5, value=3)

        # Input for number of each role per position
        st.write("**Goalkeeper Roles:**")
        num_traditional_keepers = st.number_input("Traditional Keepers", min_value=0, max_value=num_goalkeepers, value=1)
        num_sweeper_keepers = st.number_input("Sweeper Keepers", min_value=0, max_value=num_goalkeepers, value=0)
        st.write("**Defender Roles:**")
        num_ball_playing_defenders = st.number_input("Ball-Playing Defenders", min_value=0, max_value=num_defenders, value=2)
        num_no_nonsense_defenders = st.number_input("No-Nonsense Defenders", min_value=0, max_value=num_defenders, value=2)
        num_fullbacks = st.number_input("Full-Backs", min_value=0, max_value=num_defenders, value=0)
        st.write("**Midfielder Roles:**")
        num_all_action_midfielders = st.number_input("All-Action Midfielders", min_value=0, max_value=num_midfielders, value=2)
        num_midfield_playmakers = st.number_input("Midfield Playmakers", min_value=0, max_value=num_midfielders, value=2)
        num_traditional_wingers = st.number_input("Traditional Wingers", min_value=0, max_value=num_midfielders, value=1)
        num_inverted_wingers = st.number_input("Inverted Wingers", min_value=0, max_value=num_midfielders, value=0)
        st.write("**Attacker Roles:**")
        num_goal_poachers = st.number_input("Goal Poachers", min_value=0, max_value=num_attackers, value=2)
        num_target_men = st.number_input("Target Men", min_value=0, max_value=num_attackers, value=1)

        # Squad generation section
        if st.button("Generate Squad"):
            # Create a dictionary to store the selected roles for each position
            selected_roles = {
                "Goalkeeper": [
                    "Traditional Keeper" if num_traditional_keepers > 0 else None,
                    "Sweeper Keeper" if num_sweeper_keepers > 0 else None
                ],
                "Defender": [
                    "Ball-Playing Defender" if num_ball_playing_defenders > 0 else None,
                    "No-Nonsense Defender" if num_no_nonsense_defenders > 0 else None,
                    "Full-Back" if num_fullbacks > 0 else None
                ],
                "Midfielder": [
                    "All-Action Midfielder" if num_all_action_midfielders > 0 else None,
                    "Midfield Playmaker" if num_midfield_playmakers > 0 else None,
                    "Traditional Winger" if num_traditional_wingers > 0 else None,
                    "Inverted Winger" if num_inverted_wingers > 0 else None
                ],
                "Attacker": [
                    "Goal Poacher" if num_goal_poachers > 0 else None,
                    "Target Man" if num_target_men > 0 else None
                ]
            }

            # Create a dictionary for the number of players required for each role
            num_players_per_position = {
                "Traditional Keeper": num_traditional_keepers,
                "Sweeper Keeper": num_sweeper_keepers,
                "Ball-Playing Defender": num_ball_playing_defenders,
                "No-Nonsense Defender": num_no_nonsense_defenders,
                "Full-Back": num_fullbacks,
                "All-Action Midfielder": num_all_action_midfielders,
                "Midfield Playmaker": num_midfield_playmakers,
                "Traditional Winger": num_traditional_wingers,
                "Inverted Winger": num_inverted_wingers,
                "Goal Poacher": num_goal_poachers,
                "Target Man": num_target_men
            }

            # Generate the squad based on the updated player counts
            final_squad = generate_squad(combined_predictions, num_players_per_position)

            # Display the generated squad
            display_squad(final_squad)

else:
    st.write("Please upload a CSV file to begin.")
