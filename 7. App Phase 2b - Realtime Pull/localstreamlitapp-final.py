#Import libraries
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import random
from bs4 import BeautifulSoup as bs
import pandas as pd
import re
import requests
import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

### Define Functions

## Scrape Team Links
def scrape_team_urls():
    # Initialize the WebDriver (Make sure you have the appropriate driver installed, e.g., chromedriver for Chrome)
    driver = webdriver.Chrome()

    # Navigate to the webpage
    driver.get('https://www.sofascore.com/tournament/football/singapore/premier-league/634')

    # Find all div elements with the specified class name using a CSS selector
    div_elements = driver.find_elements(By.CSS_SELECTOR, "div.Box.eHXJll")

    # Create a list to store all the player page urls
    team_urls = []

    # Loop through each div element and find all a elements within it
    for div in div_elements:
        # Find all <a> elements within the current <div>
        link_elements = div.find_elements(By.CSS_SELECTOR, "a")
        
        # Loop through each <a> element and extract its href attribute
        for link in link_elements:
            href = link.get_attribute('href')
            team_urls.append(href)
            print(href)

    # Close the WebDriver
    driver.quit()

    # Return the list of team URLs
    return team_urls

## Scrape Player Links
def scrape_player_urls(team_urls):
    # Initialize the WebDriver (Make sure you have the appropriate driver installed, e.g., chromedriver for Chrome)
    driver = webdriver.Chrome()

    # Create a list to store all the player page urls
    player_urls = []

    # Loop through each team URL
    for team_url in team_urls:
        
        # Navigate to the team webpage
        driver.get(team_url)

        # Find the element using XPath and click it to navigate to the squad page
        try:
            element = driver.find_element(By.XPATH, "/html/body/div[1]/main/div[1]/div[3]/div[1]/div/div/div/h2[4]/a")
            element.click()

            # Find all div elements with the specified class name using a CSS selector
            div_elements = driver.find_elements(By.CSS_SELECTOR, "div.Box.dflyPx")

            # Loop through each div element and find all <a> elements within it
            for div in div_elements:
                link_elements = div.find_elements(By.CSS_SELECTOR, "a")
                
                # Loop through each <a> element and extract the href attribute
                for link in link_elements:
                    href = link.get_attribute('href')
                    player_urls.append(href)
                    print(href)

        except Exception as e:
            print(f"Error processing {team_url}: {e}")

    # Close the WebDriver
    driver.quit()

    # Return the list of player URLs
    return player_urls

## Scrape Player Data

def scrape_player_data(player_urls):
    # Initialize the WebDriver (Make sure you have the appropriate driver installed, e.g., chromedriver for Chrome)
    driver = webdriver.Chrome()

    # Instantiate a list to store the stats of each player
    players_list = []

    for player_url in player_urls:
        
        # Navigate to the player's webpage
        driver.get(player_url)

        # Pause to allow the page to load with a random delay
        time.sleep(random.uniform(2, 3))
        
        # Check if the bdi element contains the value 2024
        try:
            bdi_element = driver.find_element(By.CSS_SELECTOR, "bdi.Text.jFxLbA")
            if bdi_element.text != "2024":
                continue  # Skip this player if the value is not 2024
        except:
            continue  # Skip this player if the bdi element is not found
        
        # Define the CSS selectors
        css_selectors = [
            "div.Box.Flex.dlyXLO.bnpRyo",
            "div.Box.fGLgkO",
            "div.Box.jwDcoO"
        ]
        
        player_list = []

        # Find and collect elements for each CSS selector
        for selector in css_selectors:
            div_elements = driver.find_elements(By.CSS_SELECTOR, selector)
            for div in div_elements:
                lines = div.text.split("\n")
                if len(lines) >= 2:  # Ensure there are at least two lines to avoid errors
                    player_list.append(lines[0])
                    player_list.append(lines[1])
        
        # Initialize an empty dictionary
        player_dict = {}

        # Extract the string between the last two backslashes
        last_part = player_url.rstrip('/').split('/')[-2]

        # Remove any hyphens from the extracted string
        last_part_cleaned = last_part.replace('-', ' ')

        # Store player name
        player_dict["Player Name"] = last_part_cleaned

        # Loop through the list with index
        for i in range(0, len(player_list), 2):
            # Assign even-indexed value as key and odd-indexed value as value
            player_dict[player_list[i]] = player_list[i + 1]

        # Append the dictionary to the list
        players_list.append(player_dict)
    
    # Close the WebDriver
    driver.quit()

    # Create and return DataFrame of players
    players_df = pd.DataFrame(players_list)
    return players_df

# Load the PyCaret models
def load_all_models():
    try:
        models = {
            "Traditional Keeper": load_model('model_Class_Traditional Keeper'),
            "Sweeper Keeper": load_model('model_Class_Sweeper Keeper'),
            "Ball-Playing Defender": load_model('model_Class_Ball-Playing Defender'),
            "No-Nonsense Defender": load_model('model_Class_No-Nonsense Defender'),
            "Full-Back": load_model('model_Class_Full-Back'),
            "All-Action Midfielder": load_model('model_Class_All-Action Midfielder'),
            "Midfield Playmaker": load_model('model_Class_Midfield Playmaker'),
            "Traditional Winger": load_model('model_Class_Traditional Winger'),
            "Inverted Winger": load_model('model_Class_Inverted Winger'),
            "Goal Poacher": load_model('model_Class_Goal Poacher'),
            "Target Man": load_model('model_Class_Target Man')
        }
        st.write("Models loaded successfully.")
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return {}


### Squad Generation Functions

def generate_squad(prediction_results, num_players_per_position):
    """Generates a squad based on prediction results and number of players per position."""
    
    # Separate goalkeepers and outfield players
    keepers_df = prediction_results[prediction_results['POSITION'] == 'G']
    positions_of_interest = ['M', 'D', 'F']
    outfield_df = prediction_results[prediction_results['POSITION'].isin(positions_of_interest)]
    
    # Create DataFrame to hold scores for all roles
    all_scores_df = pd.DataFrame()
    
    # Evaluate goalkeepers for goalkeeper roles
    for role in ["Traditional Keeper", "Sweeper Keeper"]:
        role_predictions = keepers_df[keepers_df['model_names'] == role]
        
        if not role_predictions.empty:
            score_column = score_column_map.get(role, "prediction_score")
            role_predictions = role_predictions[['Player Name', score_column]].rename(columns={score_column: 'Score'})
            role_predictions['Role'] = role
            all_scores_df = pd.concat([all_scores_df, role_predictions], ignore_index=True)
    
    # Evaluate outfield players for outfield roles
    for role in score_column_map.keys():
        if role not in ["Traditional Keeper", "Sweeper Keeper"]:
            role_predictions = outfield_df[outfield_df['model_names'] == role]
            
            if not role_predictions.empty:
                score_column = score_column_map.get(role, "prediction_score")
                role_predictions = role_predictions[['Player Name', score_column]].rename(columns={score_column: 'Score'})
                role_predictions['Role'] = role
                all_scores_df = pd.concat([all_scores_df, role_predictions], ignore_index=True)
    
    # Sort by score in descending order
    all_scores_df = all_scores_df.sort_values(by='Score', ascending=False)
    
    # Initialize a dictionary to keep track of the number of players selected for each role
    role_counts = {role: 0 for role in score_column_map.keys()}
    
    # Initialize a list to store the selected players and their roles
    selected_players = set()
    squad = []

    for _, player in all_scores_df.iterrows():
        player_name = player['Player Name']
        player_role = player['Role']
        
        if player_name in selected_players:
            continue
        
        if role_counts[player_role] < num_players_per_position.get(player_role, 0):
            # Assign player to the role
            squad.append(player)
            selected_players.add(player_name)
            role_counts[player_role] += 1
            
            # Stop if the squad is full
            if len(selected_players) == sum(num_players_per_position.values()):
                break
    
    # Convert the squad list to a DataFrame
    final_squad = pd.DataFrame(squad) if squad else pd.DataFrame()

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
            role_players = squad[squad['Role'] == role]
            
            # Get the correct score column based on the role
            score_column = score_column_map.get(role, "prediction_score")
            
            for _, player in role_players.iterrows():
                squad_data.append([player['Player Name'], position, role, f"{player['Score']:.2f}"])
    
    # Convert the list to a DataFrame
    squad_df = pd.DataFrame(squad_data, columns=["Player Name", "Position", "Role", "Score"])
    
    # Display the DataFrame
    st.write(squad_df)

# Create tabs
st.title("GOAL 2030? Back On!")

tab1, tab2, tab3 = st.tabs(["Scrape & Clean", "Predictions", "Squad Generator"])

### Scraping 
with st.sidebar:

    st.title("Live Player Data Scraping")

    # Initialize session state variables if they don't exist
    if 'scraped_data' not in st.session_state:
        st.session_state.scraped_data = None
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    if 'current_data' not in st.session_state:
        st.session_state.current_data = None

    # Radio button for data source selection
    option = st.radio("Choose Data Source", ["Scrape Data", "Upload CSV"])

    if option == "Scrape Data":
        if st.button("Confirm Scrape"):
            st.write("Scraping player data...")
            # Replace these functions with your actual scraping functions
            team_urls = scrape_team_urls()
            player_urls = scrape_player_urls(team_urls)
            scraped_data = scrape_player_data(player_urls)
            
            st.write("Scraped Player Data:")
            st.write(scraped_data.head())
            
            # Save scraped data to CSV
            scraped_data.to_csv('livescrape.csv', index=False)
            
            # Update session state with scraped data
            st.session_state.scraped_data = scraped_data
            st.session_state.current_data = scraped_data

    elif option == "Upload CSV":
        uploaded_file = st.file_uploader("Upload a CSV file with player attributes", type="csv")
        if uploaded_file:
            uploaded_data = pd.read_csv(uploaded_file)
            # Update session state with uploaded data
            st.session_state.uploaded_data = uploaded_data
            st.session_state.current_data = uploaded_data

    # Allow user to choose which dataset to process
    data_choice = st.radio("Choose which data to process", ["Scraped Data", "Uploaded Data"])

    # Set the DataFrame to use based on the user's choice
    if data_choice == "Scraped Data":
        players_df = st.session_state.scraped_data
    elif data_choice == "Uploaded Data":
        players_df = st.session_state.uploaded_data
    else:
        players_df = None

with tab1:

    # Check if players_df is not None before proceeding with data cleaning and processing
    if players_df is not None:
        # Display the DataFrame if it exists
        st.write("Loaded Player Data:")
        st.write(players_df.head())

        ### Cleaning

        # Drop Junk Columns
        # List of columns to keep based on index (0-based index)
        columns_to_keep = list(range(0, 32)) + [33, 34, 35, 36, 48] + list(range(58, 63))

        # Select columns
        players_df_filtered = players_df.iloc[:, columns_to_keep]

        # Define new columns to keep for further analysis
        new_columns_to_keep = [0, 6, 12] + list(range(14, 21)) + list(range(22, 25)) + list(range(31, 37)) + [38]  # Example indices

        # Filter the DataFrame again
        players_df_analysis = players_df_filtered.iloc[:, new_columns_to_keep]

        # Define the original order of columns
        original_columns = players_df_analysis.columns.tolist()

        # Define the indices of columns to move and their new positions
        columns_to_move = original_columns[13:18]  # Columns 13 to 17
        remaining_columns = [col for col in original_columns if col not in columns_to_move]

        # Define the new column order
        new_order = remaining_columns[:1] + columns_to_move + remaining_columns[1:]

        # Rearrange columns
        players_df_rearranged = players_df_analysis[new_order]

        # Define the columns with percentage data
        percentage_columns = list(range(8, 11)) + list(range(15, 18)) + [19]

        # Function to extract percentage and convert to decimal
        def extract_percentage(value):
            if isinstance(value, str):
                match = re.search(r'\((\d+)%\)', value)
                if match:
                    return float(match.group(1)) / 100
            return None

        # Apply the function to each relevant column
        for col_index in percentage_columns:
            col_name = players_df_rearranged.columns[col_index]
            # Convert values to string and apply the function
            players_df_rearranged.loc[:, col_name] = players_df_rearranged[col_name].astype(str).apply(extract_percentage)


        # Filter to retain only players with SIN nationality
        players_df_sin = players_df_rearranged[players_df_rearranged.iloc[:, 1] == 'SIN']

        # Define column indices
        scale_columns = list(range(6, 13)) + list(range(14, 20))
        reverse_code_column = 13

        # Convert columns to numeric, errors='coerce' will turn non-convertible values to NaN
        for col_index in scale_columns:
            col_name = players_df_sin.columns[col_index]
            players_df_sin.loc[:, col_name] = pd.to_numeric(players_df_sin[col_name], errors='coerce')

        # Convert reverse coding column to numeric
        reverse_code_col_name = players_df_sin.columns[reverse_code_column]
        players_df_sin.loc[:, reverse_code_col_name] = pd.to_numeric(players_df_sin[reverse_code_col_name], errors='coerce')

        # Function to scale values
        def scale_values(series):
            min_val = series.min()
            max_val = series.max()
            return 25 + ((series - min_val) / (max_val - min_val)) * 75

        # Apply scaling to specified columns
        for col_index in scale_columns:
            col_name = players_df_sin.columns[col_index]
            players_df_sin.loc[:, col_name] = scale_values(players_df_sin[col_name])

        # Function to reverse code values
        def reverse_code_values(series):
            min_val = series.min()
            max_val = series.max()
            return 25 + ((max_val - series) / (max_val - min_val)) * 75

        # Apply reverse coding to column 13
        players_df_sin.loc[:, reverse_code_col_name] = reverse_code_values(players_df_sin[reverse_code_col_name])


        # Define the column indices
        columns_to_fill = [18, 19]

        # Fill NaNs with 0 in the specified columns
        for col_index in columns_to_fill:
            col_name = players_df_sin.columns[col_index]
            # Use .loc[] to ensure we are modifying the DataFrame correctly
            players_df_sin.loc[:, col_name] = players_df_sin[col_name].fillna(0)


        # Define weights for each role
        weights = {
            'Traditional Keeper': [0.01, 0.01, 0.02, 0.02, 0.02, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.24, 0.23],
            'Sweeper Keeper': [0.01, 0.01, 0.1, 0.1, 0.03, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.15, 0.15],
            'Ball-Playing Defender': [0.02, 0.01, 0.15, 0.12, 0.02, 0.12, 0.15, 0.05, 0.1, 0.05, 0.1, 0.11],
            'No-Nonsense Defender': [0.01, 0.01, 0.02, 0.02, 0.02, 0.18, 0.12, 0.12, 0.18, 0.15, 0.1, 0.07],
            'Full-Back': [0.02, 0.05, 0.05, 0.05, 0.15, 0.1, 0.1, 0.05, 0.05, 0.15, 0.15, 0.08],
            'All-Action Midfielder': [0.05, 0.05, 0.08, 0.05, 0.05, 0.15, 0.15, 0.1, 0.08, 0.08, 0.08, 0.08],
            'Midfield Playmaker': [0.02, 0.2, 0.2, 0.08, 0.08, 0.05, 0.05, 0.05, 0.05, 0.1, 0.08, 0.04],
            'Traditional Winger': [0.2, 0.15, 0.05, 0.05, 0.2, 0.02, 0.02, 0.02, 0.05, 0.15, 0.05, 0.04],
            'Inverted Winger': [0.25, 0.15, 0.05, 0.05, 0.05, 0.02, 0.02, 0.02, 0.1, 0.15, 0.15, 0.04],
            'Goal Poacher': [0.35, 0.05, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.05, 0.15, 0.15, 0.13],
            'Target Man': [0.2, 0.05, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.05, 0.3, 0.2, 0.08]
        }

        # Define column indices for each role
        col_indices = {
            'Traditional Keeper': list(range(6, 20)),
            'Sweeper Keeper': list(range(6, 20)),
            'Ball-Playing Defender': list(range(6, 18)),
            'No-Nonsense Defender': list(range(6, 18)),
            'Full-Back': list(range(6, 18)),
            'All-Action Midfielder': list(range(6, 18)),
            'Midfield Playmaker': list(range(6, 18)),
            'Traditional Winger': list(range(6, 18)),
            'Inverted Winger': list(range(6, 18)),
            'Goal Poacher': list(range(6, 18)),
            'Target Man': list(range(6, 18))
        }

        # Make a copy of the DataFrame to avoid SettingWithCopyWarning
        players_df_sin_scores = players_df_sin.copy()

        # Calculate the score for each role and add it as a new column
        for role, weights_list in weights.items():
            if len(weights_list) != len(col_indices[role]):
                raise ValueError(f"Weight list length for '{role}' does not match the number of columns")
        
            weighted_sum = sum(weights_list[i] * players_df_sin_scores.iloc[:, col_index]
                            for i, col_index in enumerate(col_indices[role]))
            players_df_sin_scores[role] = weighted_sum


        # Define the column indices for the original values and the new classification columns
        original_columns = list(range(20, 31))  # Columns to classify
        classification_columns = list(range(31, 42))  # New columns for classifications

        # Function to classify values
        def classify_value(value):
            return 1 if value > 50 else 0

        # Make a copy of the DataFrame to avoid modifying the original DataFrame
        players_df_sin_reco = players_df_sin_scores.copy()

        # Apply the classification to each relevant column and create new columns
        for orig_col, class_col in zip(original_columns, classification_columns):
            orig_col_name = players_df_sin_reco.columns[orig_col]
            class_col_name = f'Class_{orig_col_name}'
        
            # Apply classification
            players_df_sin_reco[class_col_name] = players_df_sin_reco.iloc[:, orig_col].apply(classify_value)

        # Display the updated DataFrame
        st.write("Final Data after Cleaning:")
        players_df_sin_reco

        players_df_sin_reco.to_csv('players_df_sin_reco.csv', index=False)

    else:
        st.write("No data available. Please scrape or upload data.")

        ### Data Cleaning Ends

### Prediction and Squad Generator Begins

with tab2:
    models = load_all_models()

    # Dictionary mapping model names to their score column names
    score_column_map = {
        "Traditional Keeper": "Traditional Keeper",
        "Sweeper Keeper": "Sweeper Keeper",
        "Ball-Playing Defender": "Ball-Playing Defender",
        "No-Nonsense Defender": "No-Nonsense Defender",
        "Full-Back": "Full-Back",
        "All-Action Midfielder": "All-Action Midfielder",
        "Midfield Playmaker": "Midfield Playmaker",
        "Traditional Winger": "Traditional Winger",
        "Inverted Winger": "Inverted Winger",
        "Goal Poacher": "Goal Poacher",
        "Target Man": "Target Man"
    }

    if 'players_df_sin_reco' in locals():
        # Make predictions for each model
        predictions = []
        for model_name, model in models.items():
            if model is not None:
                # Make prediction for the current model
                prediction = predict_model(model, data=players_df_sin_reco)
                # Add a new column to identify the model/role
                prediction['model_names'] = model_name
                predictions.append(prediction)

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

                    # Ensure the score_column is numeric
                    if pd.api.types.is_categorical_dtype(filtered_prediction[score_column]):
                        filtered_prediction[score_column] = filtered_prediction[score_column].astype(float)

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
                    st.write(filtered_prediction[['Player Name', model_name, 'Recommended']])

with tab3:
            # Squad generation section
            st.subheader("Squad Generation")

            # Slider for total squad size
            total_squad_size = st.slider("Select Total Squad Size:", min_value=11, max_value=35, value=23)

            # Input for number of players per position
            st.write("Set the number of players per position based on the total squad size:")
            num_goalkeepers = st.number_input("Total Goalkeepers:", min_value=1, max_value=total_squad_size, value=3)
            num_defenders = st.number_input("Total Defenders:", min_value=1, max_value=total_squad_size, value=8)
            num_midfielders = st.number_input("Total Midfielders:", min_value=1, max_value=total_squad_size, value=8)
            num_attackers = st.number_input("Total Attackers:", min_value=1, max_value=total_squad_size, value=4)

            # Check that the total number of players matches the squad size
            if num_attackers + num_defenders + num_midfielders + num_goalkeepers > total_squad_size:
                st.error("Total number of players exceeds the selected squad size.")
            else:
                num_players_per_position = {
                    "Goalkeeper": num_goalkeepers,
                    "Defender": num_defenders,
                    "Midfielder": num_midfielders,
                    "Attacker": num_attackers
                }

                st.write("Set the number of players per role based on the total players chosen for that position:")
                
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

                # Check that the total number of players per role does not exceed the total number of players per position
                if (num_traditional_keepers + num_sweeper_keepers > num_goalkeepers or
                    num_ball_playing_defenders + num_no_nonsense_defenders + num_fullbacks > num_defenders or
                    num_all_action_midfielders + num_midfield_playmakers + num_traditional_wingers + num_inverted_wingers > num_midfielders or
                    num_goal_poachers + num_target_men > num_attackers):
                    st.error("The total number of players per role exceeds the number of players chosen for that position.")
                else:
                    # Define the role constraints
                    num_players_per_role = {
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

            # Button to generate squad
            if st.button("Generate Squad"):
                # Assuming generate_squad() and display_squad() are defined elsewhere
                squad = generate_squad(combined_predictions, num_players_per_role)
                display_squad(squad)
