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

def generate_squad(prediction_results, num_players_per_position):
    """Generates a squad based on prediction results and number of players per position."""
    
    # Create a DataFrame to hold the best score and role for each player
    player_roles = pd.DataFrame()
    
    for role in model_names:
        # Filter predictions for the role
        role_predictions = prediction_results[prediction_results['model_names'] == role]
        
        # Access the correct score column based on the model name
        score_column = score_column_map.get(role, "prediction_score")
        
        # Add a new column to store the role-specific score
        role_predictions['role_score'] = role_predictions[score_column]
        
        # Merge with the main DataFrame to ensure we keep the highest score per player
        if player_roles.empty:
            player_roles = role_predictions[['Player', 'role_score', 'model_names']].rename(columns={'role_score': score_column})
        else:
            # Update with the new scores where they are higher
            player_roles = player_roles.merge(role_predictions[['Player', 'role_score', 'model_names']].rename(columns={'role_score': score_column}), 
                                              on='Player', 
                                              how='outer', 
                                              suffixes=('', '_new'))
            
            for col in player_roles.columns:
                if col.endswith('_new'):
                    base_col = col.replace('_new', '')
                    player_roles[base_col] = player_roles[[base_col, col]].max(axis=1)
            
            player_roles = player_roles.drop(columns=[col for col in player_roles.columns if col.endswith('_new')])

    # Determine the top players for each role
    final_squad = pd.DataFrame()
    for role in model_names:
        score_column = score_column_map.get(role, "prediction_score")
        num_players = num_players_per_position.get(role, 0)
        
        # Select the top players for this role based on their score
        top_players = player_roles.nlargest(num_players, score_column)
        
        # Add to the final squad
        final_squad = pd.concat([final_squad, top_players], ignore_index=True)
    
    return final_squad

def display_squad(squad):
    """Displays the generated squad."""
    
    st.header("Generated Squad")
    if squad.empty:
        st.write("No players found.")
    else:
        # Initialize a DataFrame to store the final squad with the highest score for each role
        final_squad = pd.DataFrame()
        
        for player_name, player_data in squad.groupby('Player'):
            # Find the role with the highest score for this player
            max_score_role = player_data.loc[player_data[score_column_map[player_data['model_names'].iloc[0]]].idxmax()]
            highest_score = max_score_role[score_column_map[max_score_role['model_names']]]
            
            # Append to final squad
            final_squad = final_squad.append({
                'Player': player_name,
                'Role': max_score_role['model_names'],
                'Score': highest_score
            }, ignore_index=True)
        
        # Display the final squad
        for index, player in final_squad.iterrows():
            st.write(f"- {player['Player']}: {player['Role']} ({player['Score']:.2f})")


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
    num_traditional_wingers = st.number_input("Traditional Wingers", min_value=0, max_value=num_attackers, value=2)
    num_inverted_wingers = st.number_input("Inverted Wingers", min_value=0, max_value=num_attackers, value=2)
    st.write("**Attacker Roles:**")
    num_goal_poachers = st.number_input("Goal Poachers", min_value=0, max_value=num_attackers, value=2)
    num_target_men = st.number_input("Target Men", min_value=0, max_value=num_attackers, value=2)

    # Compile the number of players required for each role
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

    # Generate the squad
    squad = generate_squad(combined_predictions, num_players_per_position)

    # Display the squad
    display_squad(squad)