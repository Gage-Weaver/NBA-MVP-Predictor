import requests  # Import requests to access webpage
from bs4 import BeautifulSoup  # Import BeautifulSoup to parse webpage
import pandas as pd  # Import pandas to create DataFrame
years = list(range(1990, 2025))  # Define list of years to search
url = 'https://www.basketball-reference.com/awards/NBA_{}.html'  # Define base URL
""" This only runs once, to avoid making too many requests to the website, do same for player stats later (not shown)
for year in years:  # Iterate through years
    url = url.format(year)  # Format URL with year
    data = requests.get(url)  # Get webpage data
    with open("mvp/{}.html".format(year), "w+") as file:  # Open file to write data
        file.write(data.text)  # Write data to file
"""
dataframes1 = []  # Create empty list to store DataFrames
""" This only runs once, to avoid long run times creating the mvps.csv file
for year in years:  # Iterate through years
    with open("mvp/{}.html".format(year), encoding='utf-8') as f:  # Open file
        page = f.read()  # Read webpage
    soup = BeautifulSoup(page, 'html.parser')  # Parse webpage
    table = soup.find('tr', class_='over_header').decompose()  # Find redundant row and remove
    mvptable = soup.find(id='mvp')  # Find MVP table
    mvp = pd.read_html(str(mvptable))[0]  # Read table into DataFrame
    mvp['Year'] = year  # Add year column to identify year of winner
    dataframes1.append(mvp)  # Append DataFrame to list
mvps = pd.concat(dataframes1)  # Concatenate DataFrames to one frame
mvps.to_csv('mvps.csv')  # Save DataFrame to CSV file
"""
dataframes2 = []
"""
for year in years:  # Iterate through years
    with open("player/{}.html".format(year), encoding='utf-8') as f:  # Open file
        page = f.read()  # Read webpage
        soup = BeautifulSoup(page, 'html.parser')  # Parse webpage
        soup.find('tr', class_='thead').decompose()  # Find redundant row and remove
        playertable = soup.find(id='per_game_stats')  # Find player table
        player = pd.read_html(str(playertable))[0]  # Read table into DataFrame
        player['Year'] = year  # Add year column to identify year of stats
        dataframes2.append(player)
players = pd.concat(dataframes2)  # Concatenate DataFrames to one frame
players.to_csv('players.csv')  # Save DataFrame to CSV file
"""
dataframes3=[]
for year in years:  # Iterate through years
    with open("team/{}.html".format(year), encoding='utf-8') as f:  # Open file
        page = f.read()  # Read webpage
    soup = BeautifulSoup(page, 'html.parser')  # Parse webpage
    table = soup.find('tr', class_='thead').decompose()  # Find redundant row and remove
    teamtable = soup.find(id='divs_standings_E')  # Find MVP table
    team = pd.read_html(str(teamtable))[0]  # Read table into DataFrame
    team['Year'] = year  # Add year column to identify year of winner
    team['Team'] = team['Eastern Conference']
    del team['Eastern Conference']
    dataframes3.append(team)  # Append DataFrame to list
    soup = BeautifulSoup(page, 'html.parser')  # Parse webpage
    table = soup.find('tr', class_='thead').decompose()  # Find redundant row and remove
    teamtable = soup.find(id='divs_standings_W')  # Find MVP table
    team = pd.read_html(str(teamtable))[0]  # Read table into DataFrame
    team['Year'] = year  # Add year column to identify year of winner
    team['Team'] = team['Western Conference']
    del team['Western Conference']
    dataframes3.append(team)  # Append DataFrame to list
teams = pd.concat(dataframes3)  # Concatenate DataFrames to one frame
teams.to_csv('teams.csv')  # Save DataFrame to CSV file

