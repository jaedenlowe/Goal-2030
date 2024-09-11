# Import necessary libraries
import streamlit as st
import pandas as pd
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time
import random
from pycaret.classification import load_model, predict_model
import os

def make_file_executable(file_path):
    if os.path.isfile(file_path):
        st.write(f"Changing permissions for: {file_path}")
        st.write(f"Current permissions: {oct(os.stat(file_path).st_mode)}")
        
        # Add executable permissions
        st_mode = os.stat(file_path).st_mode
        os.chmod(file_path, st_mode | stat.S_IEXEC)
        
        st.write(f"Updated permissions: {oct(os.stat(file_path).st_mode)}")
    else:
        st.write(f"File does not exist at: {file_path}")

# Change permissions of chromedriver
chromedriver_path = 'streamlitchromedriver/chromedriver'
make_file_executable(chromedriver_path)

def check_file_permissions(file_path):
    if os.path.isfile(file_path):
        st.write(f"File exists at: {file_path}")
        st.write(f"File permissions: {oct(os.stat(file_path).st_mode)}")
        st.write(f"Readable: {os.access(file_path, os.R_OK)}")
        st.write(f"Writable: {os.access(file_path, os.W_OK)}")
        st.write(f"Executable: {os.access(file_path, os.X_OK)}")
    else:
        st.write(f"File does not exist at: {file_path}")

# Check permissions of chromedriver
chromedriver_path = 'streamlitchromedriver/chromedriver'
check_file_permissions(chromedriver_path)
        
def scrape_player_urls():
    # Path to your Linux-compatible ChromeDriver
    chrome_driver_path = 'streamlitchromedriver/chromedriver'
    
    # Setup Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--remote-debugging-port=9222")  # For remote debugging
    chrome_options.add_argument("--disable-gpu")  # Disable GPU acceleration

    # Initialize WebDriver
    driver_service = Service(chrome_driver_path)
    driver = webdriver.Chrome(service=driver_service, options=chrome_options)

    # Navigate to the webpage
    driver.get('https://www.sofascore.com/tournament/football/singapore/premier-league/634')

    # Find all div elements with the specified class name using a CSS selector
    div_elements = driver.find_elements(By.CSS_SELECTOR, "div.Box.eHXJll")

    # Create a list to store all the player page URLs
    team_urls = []

    # Loop through each div element and find all a elements within it
    for div in div_elements:
        link_elements = div.find_elements(By.CSS_SELECTOR, "a")
        for link in link_elements:
            href = link.get_attribute('href')
            team_urls.append(href)

    # Close the WebDriver
    driver.quit()

    player_urls = []

    # Scrape player URLs from team pages
    for team_url in team_urls:
        driver = webdriver.Chrome()
        driver.get(team_url)
        try:
            element = driver.find_element(By.XPATH, "/html/body/div[1]/main/div[1]/div[3]/div[1]/div/div/div/h2[4]/a")
            element.click()

            div_elements = driver.find_elements(By.CSS_SELECTOR, "div.Box.dflyPx")
            for div in div_elements:
                link_elements = div.find_elements(By.CSS_SELECTOR, "a")
                for link in link_elements:
                    href = link.get_attribute('href')
                    player_urls.append(href)
        except Exception as e:
            print(f"Error fetching team data: {e}")
        driver.quit()

    return player_urls

# Scraping and storing player stats
def scrape_player_stats(player_urls):
    players_list = []
    driver = webdriver.Chrome()

    for player_url in player_urls:
        driver.get(player_url)
        time.sleep(random.uniform(2, 3))

        try:
            bdi_element = driver.find_element(By.CSS_SELECTOR, "bdi.Text.jFxLbA")
            if bdi_element.text != "2024":
                continue  # Skip players not from 2024
        except:
            continue

        div_elements = driver.find_elements(By.CSS_SELECTOR, "div.Box.Flex.dlyXLO.bnpRyo")
        player_list = []
        for div in div_elements:
            player_list.append(div.text.split("\n")[0])
            player_list.append(div.text.split("\n")[1])

        player_dict = {}
        last_part = player_url.rstrip('/').split('/')[-2].replace('-', ' ')
        player_dict["Player Name"] = last_part

        for i in range(0, len(player_list), 2):
            player_dict[player_list[i]] = player_list[i + 1]

        players_list.append(player_dict)

    driver.quit()
    return pd.DataFrame(players_list)

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

# Function for generating squad
def generate_squad(prediction_results, num_players_per_position):
    """Generates a squad based on prediction results and number of players per position."""
    
    # Separate goalkeepers and outfield players
    keepers_df = prediction_results[prediction_results['position'] == 'gk']
    outfield_df = prediction_results[prediction_results['position'] == 'non-gk']
    
    # Create DataFrame to hold scores for all roles
    all_scores_df = pd.DataFrame()
    
    # Evaluate goalkeepers for goalkeeper roles
    for role in ["Traditional Keeper", "Sweeper Keeper"]:
        role_predictions = keepers_df[keepers_df['model_names'] == role]
        
        if not role_predictions.empty:
            score_column = score_column_map.get(role, "prediction_score")
            role_predictions = role_predictions[['Player', score_column]].rename(columns={score_column: 'Score'})
            role_predictions['Role'] = role
            all_scores_df = pd.concat([all_scores_df, role_predictions], ignore_index=True)
    
    # Evaluate outfield players for outfield roles
    for role in score_column_map.keys():
        if role not in ["Traditional Keeper", "Sweeper Keeper"]:
            role_predictions = outfield_df[outfield_df['model_names'] == role]
            
            if not role_predictions.empty:
                score_column = score_column_map.get(role, "prediction_score")
                role_predictions = role_predictions[['Player', score_column]].rename(columns={score_column: 'Score'})
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
        player_name = player['Player']
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

# Streamlit interface
st.title("Live Player Data Scraping and Squad Generation")

player_data = None  # Initialize player_data to None

if st.button("Scrape Player Data"):
    st.write("Scraping player data...")
    player_urls = scrape_player_urls()
    player_data = scrape_player_stats(player_urls)

    st.write("Scraped Player Data:")
    st.write(player_data.head())

    # Save scraped data to CSV
    player_data.to_csv('scraped_player_data.csv', index=False)

# Load models
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

# Load the CSV file
uploaded_file = st.file_uploader("Upload a CSV file with player attributes (or use scraped data)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
elif player_data is not None:
    df = player_data

if 'df' in locals():
    st.write("Loaded Player Data:")
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

        # Slider for total squad size
        total_squad_size = st.slider("Select Total Squad Size:", min_value=11, max_value=35, value=23)

        # Input for number of players per position
        st.write("Set the number of players per position based on the total squad size:")
        num_attackers = st.number_input("Total Attackers:", min_value=1, max_value=total_squad_size, value=4)
        num_defenders = st.number_input("Total Defenders:", min_value=1, max_value=total_squad_size, value=8)
        num_midfielders = st.number_input("Total Midfielders:", min_value=1, max_value=total_squad_size, value=8)
        num_goalkeepers = st.number_input("Total Goalkeepers:", min_value=1, max_value=total_squad_size, value=3)

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
