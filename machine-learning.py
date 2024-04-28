import pandas as pd  # Import pandas
from sklearn.linear_model import Ridge  # Import Ridge regression (best to not overfit data)
from sklearn.ensemble import RandomForestRegressor  # Import Random Forest Regressor
import math
stats = pd.read_csv('stats.csv')  # Read stats data
del (stats['Unnamed: 0'])  # Remove redundant column
null = pd.isnull(stats)  # Find null values (Found nulls in % columns)
stats = stats.fillna(0)  # Fill null values with 0
# Populate predictors with everything that is not a string or direct mvp data
predictors = [ 'Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA',
'FT%', 'ORB','DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'Year', 'W', 'L', 'W/L%', 'GB', 'PS/G','PA/G'
,'SRS']
# Split data into train and test
train = stats[stats['Year'] < 2023]  # Train data up to 2022
test = stats[stats['Year'] == 2023]  # Test 2023
# Create Ridge regression model
reg = Ridge(alpha=.01)  # Create Ridge regression model (alpha is shrinking factor)
reg.fit(train[predictors], train['Share'])  # Fit model with training data (to try and predict mvp voting share)
# Predictions
predictions = reg.predict(test[predictors])  # Predict 2023 mvp voting share
predictions = pd.DataFrame(predictions, columns=['Predictions'], index=test.index)  # Create DataFrame with predictions
combination = pd.concat([test[['Player', 'Share']], predictions], axis=1)  # Combine predictions with actual data
combination = combination.sort_values('Share', ascending=False).head(10)  # Sort by actual voting share
actual_share = combination.sort_values('Share', ascending=False)  # Sort by actual voting share
combination['Rank'] = list(range(1, combination.shape[0]+1))  # Add rank column
combination = combination.sort_values('Predictions', ascending=False)  # Sort by predicted voting share
combination['Predicted_Rank'] = list(range(1, combination.shape[0]+1))  # Add predicted rank column


# Error fitting
def average_precision(combination):  # Function to calculate average precision for error fitting
    actual = combination.sort_values('Share', ascending=False).head(5)  # Sort by actual voting share take top 5
    predictor = combination.sort_values('Predictions', ascending=False)  # Sort by predicted voting share
    ps = []  # Create empty list to store precision values
    found = 0  # Initialize found counter
    seen = 1  # Initialize seen counter
    for index, row in predictor.iterrows():  # Iterate through rows
        if row['Player'] in actual['Player'].values:  # If player is in actual top 5
            found += 1  # Increment found counter
            ps.append(found/seen)  # Append precision value
        seen += 1  # Increment seen counter
    return sum(ps)/len(ps)  # Return average precision (sum of precision values divided by number of precision values

# Backtesting
years = list(range(1990, 2024))  # Define list of years to search
aps = []  # Create empty list to store average precision values
all_predictions = []  # Create empty list to store predictions
for year in years[5:]:  # Iterate through years leaving first 5 for training
    train = stats[stats['Year'] < year]  # Train data up to year
    test = stats[stats['Year'] == year]  # Test year
    reg.fit(train[predictors], train['Share'])  # Fit model with training data
    predictions = reg.predict(test[predictors])  # Predict year mvp voting share
    predictions = pd.DataFrame(predictions, columns=['Predictions'], index=test.index)  # Create DF with Predictions
    combination = pd.concat([test[['Player', 'Share']], predictions], axis=1)  # Combine predict with actual
    all_predictions.append(combination)  # Append average precision value
    aps.append(average_precision(combination))  # Append average precision value


# Add Ranks Function
def add_ranks(combination):
    combination = combination.sort_values('Share', ascending=False)  # Sort by actual voting share
    combination['Rank'] = list(range(1, combination.shape[0]+1))  # Add rank column
    combination = combination.sort_values('Predictions', ascending=False)  # Sort by predicted voting share
    combination['Predicted_Rank'] = list(range(1, combination.shape[0]+1))  # Add predicted rank column
    combination['Difference'] = combination['Rank'] - combination['Predicted_Rank']  # Add difference column
    return combination  # Return DataFrame


#Single function for all backtesting capbility
def backtest(stats,model, year, predictors):  # Function to backtest model
    aps = []  # Create empty list to store average precision values
    all_predictions = []  # Create empty list to store predictions
    for year in years[5:]:  # Iterate through years leaving first 5 for training
        train = stats[stats['Year'] < year]  # Train data up to year
        test = stats[stats['Year'] == year]  # Test year
        model.fit(train[predictors], train['Share'])  # Fit model with training data
        predictions = model.predict(test[predictors])  # Predict year mvp voting share
        predictions = pd.DataFrame(predictions, columns=['Predictions'], index=test.index)  # Create DF with Predictions
        combination = pd.concat([test[['Player', 'Share']], predictions], axis=1)  # Combine predict with actual
        combination = add_ranks(combination)  # Add ranks
        all_predictions.append(combination)  # Append average precision value
        aps.append(average_precision(combination))  # Append average precision value
    return sum(aps)/len(aps), aps, pd.concat(all_predictions)  # Return average precision,ap list, and all precision


mean_ap, aps, all_predictions = backtest(stats, reg, years[28:], predictors)  # Backtest model
# Print statement to see what predictors the model values most
# print(pd.concat([pd.Series(reg.coef_), pd.Series(predictors)], axis=1).sort_values(0, ascending=False))
# Fine-tuning model
""" Below is code for testing different alphas and 1000 turned out to be the best
alphas = [0.01, 0.1, 1, 10, 100, 1000]  # Define list of alphas to test
mean_aps = []  # Create empty list to store average precision values
for alpha in alphas:  # Iterate through alphas
    reg = Ridge(alpha=alpha)  # Create Ridge regression model with alpha
    mean_ap, aps, all_predictions = backtest(stats, reg, years[5:], predictors)  # Backtest model
    mean_aps.append(mean_ap)  # Append average precision value
print(mean_aps)  # Print mean average precision values for different alphas, (1000 is best)
"""
mean_ap, aps, all_predictions = backtest(stats, reg, years[28:], predictors)  # Backtest model with ridge regression
# Actual Ridge Regression Model
train = stats[stats['Year'] < 2023]  # Train data up to 2022
test = stats[stats['Year'] == 2023]  # Test 2023
# Create Ridge regression model
reg = Ridge(alpha=1000)  # Create Ridge regression model (alpha is shrinking factor)
reg.fit(train[predictors], train['Share'])  # Fit model with training data (to try and predict mvp voting share)
# Predictions
predictions = reg.predict(test[predictors])  # Predict 2023 mvp voting share
predictions = pd.DataFrame(predictions, columns=['Predictions'], index=test.index)  # Create DataFrame with predictions
combination = pd.concat([test[['Player', 'Share']], predictions], axis=1)  # Combine predictions with actual data
combination = combination.sort_values('Share', ascending=False).head(10)  # Sort by actual voting share
actual_share = combination.sort_values('Share', ascending=False)  # Sort by actual voting share
combination['Rank'] = list(range(1, combination.shape[0]+1))  # Add rank column
combination = combination.sort_values('Predictions', ascending=False)  # Sort by predicted voting share
combination['Predicted_Rank'] = list(range(1, combination.shape[0]+1))  # Add predicted rank column
print("Prediction for 2023 MVP Voting Share vs Actual Using Ridge Regression")
print(combination.sort_values('Predictions', ascending=False).head(10))  # Print top 10 predictions


# Random Forest
train = stats[stats['Year'] < 2023]  # Train data up to 2022
test = stats[stats['Year'] == 2023]  # Test 2023
# Create Random Forest regression model
rf = RandomForestRegressor(n_estimators=25, random_state=1, min_samples_split=5)  # Create Random Forest regression model
rf.fit(train[predictors], train['Share'])  # Fit model with training data (to try and predict mvp voting share)
# Predictions
print()
predictions = rf.predict(test[predictors])  # Predict 2023 mvp voting share
predictions = pd.DataFrame(predictions, columns=['Predictions'], index=test.index)  # Create DataFrame with predictions
combination = pd.concat([test[['Player', 'Share']], predictions], axis=1)  # Combine predictions with actual data
combination = combination.sort_values('Share', ascending=False).head(10)  # Sort by actual voting share
actual_share = combination.sort_values('Share', ascending=False)  # Sort by actual voting share
combination['Rank'] = list(range(1, combination.shape[0]+1))  # Add rank column
combination = combination.sort_values('Predictions', ascending=False)  # Sort by predicted voting share
combination['Predicted_Rank'] = list(range(1, combination.shape[0]+1))  # Add predicted rank column
print("Prediction for 2023 MVP Voting Share vs Actual Using Random Forest")
print(combination.sort_values('Predictions', ascending=False).head(10))  # Print top 10 predictions


# Predictions for 2024
train = stats[stats['Year'] < 2024]  # Train data up to 2023
test = stats[stats['Year'] == 2024]  # Test 2024
# Create Ridge regression model
print()
reg = Ridge(alpha=1000)  # Create Ridge regression model (alpha is shrinking factor)
reg.fit(train[predictors], train['Share'])  # Fit model with training data (to try and predict mvp voting share)
# Predictions
predictions = reg.predict(test[predictors])  # Predict 2024 mvp voting share
predictions = pd.DataFrame(predictions, columns=['Predictions'], index=test.index)  # Create DataFrame with predictions
combination = pd.concat([test[['Player', 'Share']], predictions], axis=1)  # Combine predictions with actual data
combination = combination.sort_values('Predictions', ascending=False).head(10)  # Sort by predicted voting share
combination['Rank'] = list(range(1, combination.shape[0]+1))  # Add rank column
combination = combination.sort_values('Predictions', ascending=False)  # Sort by predicted voting share
combination['Predicted_Rank'] = list(range(1, combination.shape[0]+1))  # Add predicted rank column
print("Prediction for 2024 MVP Voting Share vs Actual Using Ridge Regression")
print(combination.sort_values('Predictions', ascending=False).head(10))  # Print top 10 predictions
# Create Random Forest regression model
print()
rf = RandomForestRegressor(n_estimators=50, random_state=1, min_samples_split=5)  # Create Random Forest regression model
rf.fit(train[predictors], train['Share'])  # Fit model with training data (to try and predict mvp voting share)
# Predictions
predictions = rf.predict(test[predictors])  # Predict 2024 mvp voting share
predictions = pd.DataFrame(predictions, columns=['Predictions'], index=test.index)  # Create DataFrame with predictions
combination = pd.concat([test[['Player', 'Share']], predictions], axis=1)  # Combine predictions with actual data
combination = combination.sort_values('Predictions', ascending=False).head(10)  # Sort by predicted voting share
combination['Rank'] = list(range(1, combination.shape[0]+1))  # Add rank column
combination = combination.sort_values('Predictions', ascending=False)  # Sort by predicted voting share
combination['Predicted_Rank'] = list(range(1, combination.shape[0]+1))  # Add predicted rank column
print("Prediction for 2024 MVP Voting Share vs Actual Using Random Forest")
print(combination.sort_values('Predictions', ascending=False).head(10))  # Print top 10 predictions


