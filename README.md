# Goal 2030? 
## Taking Singapore to the World Cup
### By Jaeden Lowe

## Content Directory:
- [Background](#Background)
- [Problem Statement](#Problem-Statement)
- [Aim](#Aim)
- [Objectives](#Objectives)
- [Data Collection](#Data-Collection)
- [Modelling](#Modelling)
- [Modelling Results](#Modelling-Results)
- [Solution](#Solution)
- [Future Work](#Future-Work)

## Background

Football is the most popular sport globally, capturing the hearts of millions. In the realm of football analytics, data has become an indispensable tool for improving team performance, scouting new talent, and making strategic decisions.

In Singapore, football is a beloved sport, but both the local league and our national team's performance has been in the doldrums for the past decade. The national team's performance can benefit from more sophisticated selection methods. 

By leveraging club-level statistics, we can enhance the selection process for the national team, ensuring that the best players are chosen based on their performance metrics. This approach aligns with the global trend of data-driven decision-making in sports.

## Problem Statement

### The lack of comprehensive and accessible data, and consequentially, data-driven tools, to faciliate the selection of players for the Singapore national team meant that selection is still heavily reliant of subjective judgement of coaches and scouts. While experience and intuition are still useful, this method can lead to biases and overlooked talents.

## Aim

This project aims to address these issues by developing a data-driven tool/app to player selection for the Singapore national football team. By analyzing in-depth club statistics, including goals, assists, passes, and other performance metrics, we can create a more objective and transparent selection process. This approach not only helps in identifying the best players for specific roles and formations but also ensures that the selection process is based on empirical evidence rather than subjective opinions.

## Objectives

1. **Data Collection and Integration**: Gather comprehensive football statistics from reliable sources, such as SofaScore, for Singaporean football clubs.
2. **Performance Analysis**: Analyze players' performance data to identify key metrics that are indicative of success in various positions and roles.
3. **Role-Based Prediction**: Develop a framework to suggest players for specific roles in different formations based on their performance metrics.
4. **Squad Generation**: Create a tool/app that can generate a squad based on coaches/scouts input, based on their predicted role ability.
5. **Validation**: Test the recommendations against historical data and expert opinions to validate the accuracy and effectiveness of the data-driven approach.

By achieving these objectives, the project seeks to revolutionize the national team selection process, making it more scientific, transparent, and fair, ultimately contributing to the improvement of Singapore's national football team's performance on the international stage.

## Data Collection

**Sources**:
- **SofaScore**: Comprehensive match statistics, including goals, assists, passes, and other performance metrics.

**Process**:
1. **Web Scraping**: Using web scraping techniques to extract data from SofaScore
   - **Tools**: `Selenium` to navigate the complex CSS nature of the website.
2. **Data Integration**: Consolidating data from multiple sources into a unified database.
   - **Tools**: `pandas` for data manipulation and integration.
3. **Data Cleaning**: Ensuring the data is clean, accurate, and formatted consistently.
   - Handling missing values, duplicates, and data type inconsistencies.
4. **Exploratory Data Analysis**: Analyzing the data to uncover patterns, trends, and insights.
   - Tools: pandas, matplotlib, seaborn
   - Techniques:
        - Descriptive statistics: Calculating mean, median, mode, standard deviation, etc.
        - Visualization: Creating charts (histograms, scatter plots, box plots, etc.) to visualize data distributions and relationships.
        - Correlation analysis: Identifying relationships between variables.

Our data scraping, pre-processing, modeling process and exploratory data analysis steps can be found in the following notebooks:

[01_scraping](/2.%20Data%20Scrape/01_scraping.ipynb)  
[02_modelling](/3.%20Final%20Model/02_modelling.ipynb)  
[03_eda](03_eda.ipynb)  

## Modelling

### 1. Define Specific Roles
For each position, define the specific roles required for each.

**Goalkeepers**:
- Traditional Keeper
- Sweeper Keeper

**Defenders**:
- Ball-Playing Defender
- No-Nonsense Defender
- Full-Back

**Midfielders**:
- All-Action Midfielder
- Midfield Playmaker
- Traditional Winger
- Inverted Winger

**Forwards**:
- Goal Poacher
- Target Man

### 2. Collect and Preprocess Data
Collect data relevant to these specific roles. Gather both general and role-specific statistics:
- **General Attributes**: Goals, assists, tackles, interceptions, passing accuracy.
- **Role-Specific Attributes**: Long passes for a deep-lying playmaker, aerial duels for a target man.

### 3. Role Suitability Analysis
Develop a scoring system to evaluate how well a player fits into a specific role:
- **Attribute Weights**: Assign different weights to attributes based on their importance for each role.
- **Composite Scores**: Calculate a composite score for each player for each role.

### 4. Machine Learning Models
Train machine learning models to predict role suitability:
- **Training Data**: Use historical data of players who have successfully played in specific roles.
- **Models**: Experiment with models through pycaret, like Random Forests.

## Modelling Results

| Average Model Accuracy  | 0.9485 |
|-------------------------|--------|

| Role                        |   M1   |   M2   |   M3   |
|-----------------------------|:------:|:------:|:------:|
| Class_Traditional Keeper    | 0.9857 | 0.9857 | 0.9857 |
| Class_Sweeper Keeper        | 0.9381 | 0.9214 | 0.9214 |
| Class_Ball Playing Defender | 0.9714 | 0.9690 | 0.9690 |
| Class_No Nonsense Defender  | 0.9548 | 0.9524 | 0.9524 |
| Class_Full Back             | 0.9357 | 0.9048 | 0.9048 |
| Class_All Action Midfielder | 0.9333 | 0.9214 | 0.9214 |
| Class_Midfield Playmaker    | 0.9405 | 0.9405 | 0.9405 |
| Class_Traditional Winger    | 0.9690 | 0.8905 | 0.8905 |
| Class_Inverted Winger       | 0.9500 | 0.9381 | 0.9190 |
| Class_Goal Poacher          | 0.9381 | 0.8548 | 0.8524 |
| Class_Target Man            | 0.9167 | 0.9095 | 0.8857 |

|             Role            |        1st Model       |        2nd Model       |          3rd Model         |
|:---------------------------:|:----------------------:|:----------------------:|:--------------------------:|
| Class_Traditional Keeper    | KNeighborsClassifier   | DecisionTreeClassifier | RidgeClassifier            |
| Class_Sweeper Keeper        | LogisticRegression     | ExtraTreesClassifier   | GaussianNB                 |
| Class_Ball Playing Defender | ExtraTreesClassifier   | RidgeClassifier        | LinearDiscriminantAnalysis |
| Class_No Nonsense Defender  | LogisticRegression     | ExtraTreesClassifier   | LGBMClassifier             |
| Class_Full Back             | LogisticRegression     | RidgeClassifier        | ExtraTreesClassifier       |
| Class_All Action Midfielder | LogisticRegression     | KNeighborsClassifier   | RidgeClassifier            |
| Class_Midfield Playmaker    | LogisticRegression     | ExtraTreesClassifier   | XGBClassifier              |
| Class_Traditional Winger    | LogisticRegression     | RidgeClassifier        | LinearDiscriminantAnalysis |
| Class_Inverted Winger       | LogisticRegression     | RidgeClassifier        | LinearDiscriminantAnalysis |
| Class_Goal Poacher          | KNeighborsClassifier   | ExtraTreesClassifier   | LogisticRegression         |
| Class_Target Man            | RandomForestClassifier | XGBClassifier          | ExtraTreesClassifier       |


## Solution

The app can be found here: [Streamlit](/7.%20App%20Phase%202b%20-%20Realtime%20Pull/localstreamlitapp-final.py)  

## Future Work

This project has laid a strong foundation for a data-driven approach to Singapore national football team selection. It can only get better by:

**1. Integrating Overseas Singaporean Players:**

Currently, the project focuses on players from the Singaporean league. Expanding the data collection to include overseas Singaporean players plying their trade in other leagues would create a more comprehensive player pool.
Potential challenges include:
- Accessing data from various international leagues.
- Accounting for the varying levels of competition in different leagues.
Solutions could involve:
- Employing data normalization techniques.
- Developing a weighting system to adjust for league difficulty.

**2. Squad Generation Based on Opposition Data:**

The current model focuses on individual player performance. Taking it a step further, the app could generate a squad based on the specific strengths and weaknesses of the upcoming opponent.
This would involve:
- Gathering data on opposing teams' playing style, formations, and key players.
- Developing algorithms that analyze both our players' strengths and the opponent's weaknesses.
- Integrating this data with the existing player suitability models.
This would require:
- Building new models that account for both player and team-level factors.

By tackling these future work areas, the project can evolve into a powerful tool that helps coaches select the most effective squad for any given opponent, significantly enhancing Singapore's chances of success on the international stage.

