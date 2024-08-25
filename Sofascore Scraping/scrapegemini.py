from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.exceptions import NoSuchElementException, StaleElementReferenceException
from time import sleep
import random
import os

# Verify the ChromeDriver path
chrome_driver_path = "D:\\Downloads\\chromedriver-win64\\chromedriver-win64\\chromedriver.exe"
if not os.path.exists(chrome_driver_path):
    raise FileNotFoundError(f"ChromeDriver not found at {chrome_driver_path}")

# Set up the ChromeDriver service
service = Service(executable_path=chrome_driver_path)

# Setting up ChromeDriver with anti-detection measures
options = webdriver.ChromeOptions()
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)

# Initialize the WebDriver with the Service object
try:
    driver = webdriver.Chrome(service=service, options=options)
except Exception as e:
    print(f"Failed to initialize WebDriver: {e}")
    raise

# Adding anti-detection measures
driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
    "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
})

# Sofascore Singapore Premier League URL
base_url = "https://www.sofascore.com/tournament/football/singapore/premier-league/634#id:59708"


def scrape_singaporean_players(max_retries=3):
    """
    Scrapes Singaporean players from Sofascore with retries for potential exceptions.

    Args:
        max_retries (int, optional): The maximum number of retries for failed element lookups. Defaults to 3.
    """
    driver.get(base_url)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

    # Navigate to each team page
    teams = driver.find_elements(By.CSS_SELECTOR, "a[href*='/team/']")
    for team in teams:
        team_name = team.text
        team_url = team.get_attribute('href')
        print(f"Scraping team: {team_name}")

        driver.get(team_url)
        sleep(random.uniform(1, 3))  # Random delay to mimic human interaction

        try:
            players = driver.find_elements(By.CSS_SELECTOR, "a[href*='/player/']")
        except (NoSuchElementException, StaleElementReferenceException) as e:
            # Retry finding players if element lookup fails (up to max_retries)
            print(f"Error finding players on team page: {e}")
            if max_retries > 0:
                scrape_singaporean_players(max_retries - 1)
                continue
            else:
                print(f"Failed to find players after {max_retries} retries. Skipping team.")
                continue

        for player in players:
            player_name = player.text
            player_url = player.get_attribute('href')

            # Navigate to player page
            driver.get(player_url)
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

            # Example: check for Singapore nationality (update selector as necessary)
            try:
                nationality = driver.find_element(By.XPATH, "//div[contains(text(),'Singapore')]")
                if nationality:
                    print(f"Singaporean player found: {player_name}, URL: {player_url}")

                    # Extract player stats here, using the correct HTML structure
                    # Example:
                    # appearances_element = driver.find_element(By.XPATH, "//span[text()='Appearances']/following-sibling::span")
                    # appearances = appearances_element.text
                    #
                    # goals_element = driver.find_element(By.XPATH, "//span[text()='Goals']/following-sibling::span")
                    # goals = goals_element.text
                    #
                    # # Add more stats as needed

                    # Save or process the data as needed
                    # Example:
                    # player_data = {
                    #     'name': player_name,
                    #     'url': player_url,
                    #     'appearances': appearances,
                    #     'goals': goals,
                    #     # Add more stats
                    # }
                    #
                    # # Append player data to a list or save to a file
                    # player_data_list.append(player_data)

            except (NoSuchElementException, StaleElementReferenceException) as e:
                print(f"Player {player_name} does not match criteria or an error occurred: {e}")

            driver.back()
            sleep(random.uniform(1, 3))

scrape_singaporean_players()