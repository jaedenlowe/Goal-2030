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

# Additional functions for squad generation
def generate_squad(prediction_results, num_players_per_position, selected_roles):
  """Generates a squad based on prediction results, number of players per position, and selected roles."""

  squad = []
  for position, num_players in num_players_per_position.items():
    # Filter predictions for the current position
    position_predictions = [result for result in prediction_results if result['model_name'].startswith(position)]

    # Sort predictions by prediction_score in descending order
    position_predictions = sorted(position_predictions, key=lambda x: x['prediction_score'], reverse=True)

    # Select the top num_players based on prediction_score and selected roles
    selected_players = []
    for player in position_predictions:
      if player['model_name'] in selected_roles[position]:
        selected_players.append(player)
        if len(selected_players) >= num_players:
          break

    squad.extend(selected_players)

  return squad

def display_squad(squad):
  """Displays the generated squad."""

  st.header("Generated Squad")
  for player in squad:
    st.write(f"- {player['Player']}: {player['model_name']} ({player['prediction_score']:.2f})")

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
      score_column = score_column_map.get(model_name, "score") # Default to "score" if not found

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

# Squad generation section
st.subheader("Squad Generation")

# Input for number of players per position
num_goalkeepers = st.number_input("Number of Goalkeepers", min_value=0, max_value=5, value=1)
num_defenders = st.number_input("Number of Defenders", min_value=0, max_value=10, value=4)
num_midfielders = st.number_input("Number of Midfielders", min_value=0, max_value=10, value=5)
num_attackers = st.number_input("Number of Attackers", min_value=0, max_value=5, value=3)

# Input for selected roles per position
st.write("**Goalkeeper Roles:**")
selected_goalkeeper_roles = st.multiselect("Select Roles", options=["Traditional Keeper", "Sweeper Keeper"], default=["Traditional Keeper"])
st.write("**Defender Roles:**")
selected_defender_roles = st.multiselect("Select Roles", options=["Ball-Playing Defender", "No-Nonsense Defender", "Full-Back"], default=["Ball-Playing Defender", "No-Nonsense Defender"])
st.write("**Midfielder Roles:**")
selected_midfielder_roles = st.multiselect("Select Roles", options=["All-Action Midfielder", "Midfield Playmaker", "Traditional Winger", "Inverted Winger"], default=["All-Action Midfielder", "Midfield Playmaker"])
st.write("**Attacker Roles:**")
selected_attacker_roles = st.multiselect("Select Roles", options=["Goal Poacher", "Target Man"], default=["Goal Poacher", "Target Man"])

# Button to generate the squad
if st.button("Generate Squad"):
  num_players_per_position = {
      "Goalkeeper": num_goalkeepers,
      "Defender": num_defenders,
      "Midfielder": num_midfielders,
      "Attacker": num_attackers
  }

  selected_roles = {
      "Goalkeeper": selected_goalkeeper_roles,
      "Defender": selected_defender_roles,
      "Midfielder": selected_midfielder_roles,
      "Attacker": selected_attacker_roles
  }

  squad = generate_squad(predictions, num_players_per_position, selected_roles)
  display_squad(squad)