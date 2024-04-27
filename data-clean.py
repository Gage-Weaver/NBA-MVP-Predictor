import pandas as pd  # Import pandas
# MVP Data Cleaning
mvps = pd.read_csv('mvps.csv')  # Read MVP data
mvps = mvps[['Player', 'Year', 'Pts Won', 'Pts Max', 'Share']]  # Select specific columns only, stats not needed
# Player Data Cleaning
players = pd.read_csv('players.csv')  # Read player data
del players['Rk']  # Remove redundant column
del players['Unnamed: 0']  # Remove redundant column
players['Player'] = players['Player'].str.replace('*', '', regex=False)  # Remove asterisk from player names
def one_row(df):  # Function to check if player has mutiple teams and fix
    if df.shape[0] == 1:  # If player only played for one team
        return df  # Return row
    else:  # If player played for multiple teams
        row = df[df['Tm'] == 'TOT']  # Find row with Total stats
        row['Tm'] = df.iloc[-1, :]['Tm']  # Set team to last team played for
        return row  # Return row


players = players.groupby(['Player', 'Year']).apply(one_row)  # Apply function to DataFrame
players.index = players.index.droplevel()  # Drop redundant index level
players.index = players.index.droplevel()  # Drop redundant index level
combined = players.merge(mvps, how='outer', on=['Player', 'Year'])  # Merge DataFrames
combined[["Pts Won", "Pts Max", "Share"]] = combined[["Pts Won", "Pts Max", "Share"]].fillna(0)  # set NaN values to 0
# Team Data Cleaning
teams = pd.read_csv('teams.csv')  # Read team data
teams = teams[~teams['W'].str.contains('Division')]  # Remove rows with division names
teams['Team'] = teams['Team'].str.replace("*", "", regex=False)  # Remove asterisk from team names
nicknames = {}
with open('nicknames.txt') as file:  # Open file
    lines = file.readlines()  # Read lines
    for line in lines[1:]:
        abbrev, name = line.replace("\n", '').split(',')  # Remove newline character
        nicknames[abbrev] = name
combined['Team'] = combined['Tm'].map(nicknames)  # Map team abbreviations to full names
# Merging
stats = combined.merge(teams, how='outer', on=['Team', 'Year'])  # Merge DataFrames
del stats["Unnamed: 0"]  # Remove redundant column
stats['GB'] = stats['GB'].str.replace('—', '0')  # Replace '—' with 0, as it represents 0 games back
stats = stats.apply(pd.to_numeric, errors='ignore')  # Convert columns to numeric values if possible
stats.to_csv('stats.csv')  # Save DataFrame to CSV file
highest_points = stats[stats["G"] > 70].sort_values('PTS', ascending=False).head(10)  # Filter players with more than 70 games and sort by points



