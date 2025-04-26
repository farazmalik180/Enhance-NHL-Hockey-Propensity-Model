import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
from utils import save_model, load_model
import torch.nn.functional as F
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score,classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier



def XGB_Classifier(total_game_list):
    """
    Performs classification on game data using XGBoost to compute win probability.

    Parameters:
    - total_game_list: List of game objects.

    Returns:
    - xpoints: List of rating differentials.
    - ypoints: List of game outcomes (win/loss).
    - model: Trained XGBoost classifier.
    """
    # Define Features: Rating Differential
    xpoints = np.array([
    [
        game.home_team.power - game.away_team.power,  # Rating Differential
        game.home_team.win_pct - game.away_team.win_pct,  # Win % Difference
        game.home_team.last_5_games_win_pct - game.away_team.last_5_games_win_pct,  # Recent Form
        game.home_advantage  # Home Advantage (1 or 0)
    ] 
    for game in total_game_list
])

    
    # Define Target: Convert Ties to Binary (0 for loss, 1 for win)
    ypoints = np.array([
        1 if game.home_score > game.away_score else 0
        for game in total_game_list
    ])

    # Load existing model if available
    model = load_model()
    if model is not None:
        return xpoints, ypoints, model

    # Train XGBoost Classifier
    model = XGBClassifier(
        n_estimators=500,  
        learning_rate=0.05,  
        max_depth=5,  
        subsample=0.8,  
        colsample_bytree=0.8,  
        gamma=0,  
        reg_alpha=0.01,  
        reg_lambda=1,  
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )

    # Train Model
    model.fit(xpoints, ypoints)

    # Save the trained model
    save_model(model)

    # Test Accuracy
    y_pred = model.predict(xpoints)
    accuracy = accuracy_score(ypoints, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    return xpoints, ypoints, model



def calculate_probabilities(home_team, away_team, param):
    """
    Calculates the win probability for a home team vs. an away team.

    Parameters:
    - home_team: Home Team object.
    - away_team: Away Team object.
    - param: Fitted parameter from the logistic regression model.

    Returns:
    - win_prob: Probability of the home team winning.
    """
    rating_diff = home_team.power - away_team.power
    win_prob = 1 / (1 + np.exp(-rating_diff / param))
    return win_prob



def calculate_spread_probabilities(home_team, away_team, param, lower_bound, upper_bound):
    """
    Calculates the probability of a team winning by a specific spread.

    Parameters:
    - home_team: Home Team object.
    - away_team: Away Team object.
    - param: Fitted parameter from the logistic regression model.
    - lower_bound: Lower bound of the spread range.
    - upper_bound: Upper bound of the spread range.

    Returns:
    - Probability within the spread range.
    """
    rating_diff = home_team.power - away_team.power

    if lower_bound == '-inf':
        if upper_bound == 'inf':
            return 1.0
        return 1 / (1 + np.exp((upper_bound - rating_diff) / param))
    elif upper_bound == 'inf':
        return 1 - (1 / (1 + np.exp((lower_bound - rating_diff) / param)))
    else:
        prob_upper = 1 / (1 + np.exp((upper_bound - rating_diff) / param))
        prob_lower = 1 / (1 + np.exp((lower_bound - rating_diff) / param))
        return prob_lower - prob_upper



def GradientBoostingClassifier_with_scaling(total_game_list):
    """
    Performs logistic regression with feature scaling and improved feature engineering.
    """
    xpoints = []  # Features: Rating differential & schedule factor
    ypoints = []  # Target: Game outcomes

    # Construct feature and target datasets
    for game in total_game_list:
        xpoints.append([game.home_team.power - game.away_team.power, game.home_team.schedule])
        if game.home_score > game.away_score:
            ypoints.append(1)  # Win
        elif game.home_score < game.away_score:
            ypoints.append(0)  # Loss
        else:
            ypoints.append(0.5)  # Tie (optional)

    # Convert to NumPy array for scaling
    X = np.array(xpoints)
    y = np.array(ypoints)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data for training and validation
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)



    # Gradient Boosting Classifier with Cross-Validation
    gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    scores = cross_val_score(gbc, X_train, y_train, cv=5)
    print(f"Gradient Boosting Cross-Validation Accuracy: {scores.mean():.4f}")

    # Train the model on the entire training data
    gbc.fit(X_train, y_train)
    y_pred = gbc.predict(X_test)
    print(f"Gradient Boosting Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))


    # last log_reg
    return X_scaled, y, gbc


def random_forest_model(total_game_list):
    """
    Implements a Random Forest classifier for predictions.
    """
    xpoints = []
    ypoints = []

    # Prepare features and labels
    for game in total_game_list:
        xpoints.append([game.home_team.power - game.away_team.power, game.home_team.schedule])
        if game.home_score > game.away_score:
            ypoints.append(1)  # Win
        elif game.home_score < game.away_score:
            ypoints.append(0)  # Loss
        else:
            ypoints.append(0.5)  # Tie (optional)

    X = np.array(xpoints)
    y = np.array(ypoints)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Random Forest Classifier
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)

    # Evaluate model
    y_pred = rf_clf.predict(X_test)
    print(f"Random Forest Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    return rf_clf



import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss
from scipy.stats import pearsonr

def model_performance(xpoints, ypoints, model):
    """
    Evaluates the performance of a trained Gradient Boosting model.

    Parameters:
    - xpoints: Feature data (rating differentials and other features)
    - ypoints: True labels (win/loss/tie outcomes)
    - model: Trained Gradient Boosting model
    """
    X = np.array(xpoints)
    # Convert xpoints to NumPy array for prediction
    raw_predictions = model.predict(X)

    # Convert raw predictions to probability using sigmoid function
    predicted_probs = 1 / (1 + np.exp(-raw_predictions))

    # Convert probabilities to binary classes
    predicted_classes = (predicted_probs >= 0.5).astype(int)
    
    confidence_scores = np.abs(predicted_probs - 0.5) * 2
    # Calculate performance metrics
    
    log_loss_value = log_loss(ypoints, predicted_probs)
    r, p = pearsonr(predicted_probs, ypoints)
    
    
    y_pred = model.predict(xpoints)
    accuracy = accuracy_score(ypoints, y_pred)
    
    avg_confidence = np.mean(confidence_scores) 
    
    print(f'Pearson Correlation of Predictions and Outcomes: {r:.3f}')
    print(f'Log Loss: {log_loss_value:.3f}')
    print(f'Accuracy: {accuracy:.2f}%')
    print(f'Average Confidence Score: {avg_confidence:.2f}')
    # Plot decision curve
    plt.scatter(xpoints[:, 0], ypoints, color='grey', alpha=0.5, label='Actual Outcomes')
    plt.scatter(xpoints[:, 0], predicted_probs, color='blue', alpha=0.5, label='Predicted Probabilities')
    plt.axhline(0.5, color='red', linestyle='--', label='Decision Boundary (0.5)')
    
    plt.legend()
    plt.title('Gradient Boosting: Team Rating Difference vs Win Probability')
    plt.xlabel('Rating Difference')
    plt.ylabel('Predicted Win Probability')
    plt.show()





def calculate_probabilities(home_team, away_team, param):
    """
    Calculates the win probability for a home team vs. an away team.

    Parameters:
    - home_team: Home Team object.
    - away_team: Away Team object.
    - param: Fitted parameter from the logistic regression model.

    Returns:
    - win_prob: Probability of the home team winning.
    """
    rating_diff = home_team.power - away_team.power
    win_prob = 1 / (1 + np.exp(-rating_diff / param))
    return win_prob


def evaluate_game_predictions(total_game_list, param):
    """
    Evaluates the model's predictions on game outcomes.

    Parameters:
    - total_game_list: List of Game objects.
    - param: Fitted parameter from the logistic regression model.

    Returns:
    - accuracy: The accuracy of the model's predictions.
    """
    correct_predictions = 0

    for game in total_game_list:
        win_prob = calculate_probabilities(game.home_team, game.away_team, param)
        predicted_win = win_prob > 0.5
        actual_win = game.home_score > game.away_score

        if predicted_win == actual_win:
            correct_predictions += 1

    accuracy = correct_predictions / len(total_game_list)
    print(f"Prediction Accuracy: {accuracy:.2%}")
    return accuracy


def calc_prob(team, opponent, param):
    return 1/(1+np.exp((team.power-opponent.power)/param))


def calc_spread(team, opponent, param, lower_bound_spread, upper_bound_spread):
    if lower_bound_spread == '-inf':
        if upper_bound_spread == 'inf':
            return 1
        return 1/(1+np.exp((upper_bound_spread-(team.power-opponent.power))/param))
    elif upper_bound_spread == 'inf': 
        return 1 - 1/(1+np.exp((lower_bound_spread-(team.power-opponent.power))/param))
    else: 
        return 1/(1+np.exp((upper_bound_spread-(team.power-opponent.power))/param)) - 1/(1+np.exp((lower_bound_spread-(team.power-opponent.power))/param))



def custom_game_selector(param, team_list):
    valid = False
    while valid == False:
        home_team_input = input('Enter the home team: ')
        for team in team_list:
            if home_team_input.strip().lower() == team.name.lower().replace('é','e'):
                home_team = team
                valid = True
        if valid == False:
            print('Sorry, I am not familiar with this team. Maybe check your spelling?')

    valid = False
    while valid == False:
        away_team_input = input('Enter the away team: ')
        for team in team_list:
            if away_team_input.strip().lower() == team.name.lower().replace('é','e'):
                away_team = team
                valid = True
        if valid == False:
            print('Sorry, I am not familiar with this team. Maybe check your spelling?')

    game_probability_df = pd.DataFrame(columns = ['', home_team.name, away_team.name])

    game_probability_df = pd.concat([game_probability_df, pd.DataFrame.from_dict([{'':'Rating', home_team.name:f'{home_team.power:.3f}', away_team.name:f'{away_team.power:.3f}'}])], ignore_index=True)
    game_probability_df = pd.concat([game_probability_df, pd.DataFrame.from_dict([{'':'Record', home_team.name:f'{home_team.record}', away_team.name:f'{away_team.record}'}])], ignore_index=True)
    game_probability_df = pd.concat([game_probability_df, pd.DataFrame.from_dict([{'':'Point PCT', home_team.name:f'{home_team.pct:.3f}', away_team.name:f'{away_team.pct:.3f}'}])], ignore_index=True)
    game_probability_df = pd.concat([game_probability_df, pd.DataFrame.from_dict([{'':'Win Probability', home_team.name:f'{calc_prob(home_team, away_team, param)*100:.2f}%', away_team.name:f'{(calc_prob(away_team, home_team, param))*100:.2f}%'}])], ignore_index=True)
    game_probability_df = pd.concat([game_probability_df, pd.DataFrame.from_dict([{'':'Win by 1 Goal', home_team.name:f'{calc_spread(home_team, away_team, param, 0, 1.5)*100:.2f}%', away_team.name:f'{calc_spread(away_team, home_team, param, 0, 1.5)*100:.2f}%'}])], ignore_index=True)
    game_probability_df = pd.concat([game_probability_df, pd.DataFrame.from_dict([{'':'Win by 2 Goals', home_team.name:f'{calc_spread(home_team, away_team, param, 1.5, 2.5)*100:.2f}%', away_team.name:f'{calc_spread(away_team, home_team, param, 1.5, 2.5)*100:.2f}%'}])], ignore_index=True)
    game_probability_df = pd.concat([game_probability_df, pd.DataFrame.from_dict([{'':'Win by 3 Goals', home_team.name:f'{calc_spread(home_team, away_team, param, 2.5, 3.5)*100:.2f}%', away_team.name:f'{calc_spread(away_team, home_team, param, 2.5, 3.5)*100:.2f}%'}])], ignore_index=True)
    game_probability_df = pd.concat([game_probability_df, pd.DataFrame.from_dict([{'':'Win by 4 Goals', home_team.name:f'{calc_spread(home_team, away_team, param, 3.5, 4.5)*100:.2f}%', away_team.name:f'{calc_spread(away_team, home_team, param, 3.5, 4.5)*100:.2f}%'}])], ignore_index=True)
    game_probability_df = pd.concat([game_probability_df, pd.DataFrame.from_dict([{'':'Win by 5+ Goals', home_team.name:f'{calc_spread(home_team, away_team, param, 4.5, "inf")*100:.2f}%', away_team.name:f'{calc_spread(away_team, home_team, param, 4.5, "inf")*100:.2f}%'}])], ignore_index=True)
    game_probability_df = game_probability_df.set_index('')

    return home_team, away_team, game_probability_df



def get_upsets(total_game_list):
    upset_df = pd.DataFrame(columns = ['Home Team', 'Home Goals', 'Away Goals', 'Away Team', 'Date', 'xGD', 'GD', 'Upset Rating'])

    for game in total_game_list:
        expected_score_diff = game.home_team.power - game.away_team.power #home - away
        actaul_score_diff = game.home_score - game.away_score
        upset_rating = actaul_score_diff - expected_score_diff #Positive score is an upset by the home team. Negative scores are upsets by the visiting team.

        upset_df = pd.concat([upset_df, pd.DataFrame.from_dict([{'Home Team':game.home_team.name, 'Home Goals':int(game.home_score), 'Away Goals':int(game.away_score), 'Away Team':game.away_team.name, 'Date':game.date,'xGD':f'{expected_score_diff:.2f}', 'GD':int(actaul_score_diff), 'Upset Rating':f'{abs(upset_rating):.2f}'}])], ignore_index=True)

    upset_df = upset_df.sort_values(by=['Upset Rating'], ascending=False)
    upset_df = upset_df.reset_index(drop=True)
    upset_df.index += 1
    return upset_df

def get_best_performances(total_game_list):
    performance_df = pd.DataFrame(columns = ['Team', 'Opponent', 'GF', 'GA', 'Date', 'xGD', 'Performance'])

    for game in total_game_list:
        performance_df = pd.concat([performance_df, pd.DataFrame.from_dict([{'Team':game.home_team.name, 'Opponent':game.away_team.name, 'GF':int(game.home_score), 'GA':int(game.away_score), 'Date':game.date, 'xGD':f'{game.home_team.power-game.away_team.power:.2f}', 'Performance':round(game.away_team.power+game.home_score-game.away_score,2)}])], ignore_index = True)
        performance_df = pd.concat([performance_df, pd.DataFrame.from_dict([{'Team':game.away_team.name, 'Opponent':game.home_team.name, 'GF':int(game.away_score), 'GA':int(game.home_score), 'Date':game.date, 'xGD':f'{game.away_team.power-game.home_team.power:.2f}', 'Performance':round(game.home_team.power+game.away_score-game.home_score,2)}])], ignore_index = True)

    performance_df = performance_df.sort_values(by=['Performance'], ascending=False)
    performance_df = performance_df.reset_index(drop=True)
    performance_df.index += 1
    return performance_df

def get_team_consistency(team_list):
    consistency_df = pd.DataFrame(columns = ['Team', 'Rating', 'Consistency (z-Score)'])

    for team in team_list:
        consistency_df = pd.concat([consistency_df, pd.DataFrame.from_dict([{'Team':team.name, 'Rating':f'{team.power:.2f}', 'Consistency (z-Score)':team.calc_consistency()}])], ignore_index = True)

    consistency_df['Consistency (z-Score)'] = consistency_df['Consistency (z-Score)'].apply(lambda x: (x-consistency_df['Consistency (z-Score)'].mean())/-consistency_df['Consistency (z-Score)'].std())

    consistency_df = consistency_df.sort_values(by=['Consistency (z-Score)'], ascending=False)
    consistency_df = consistency_df.reset_index(drop=True)
    consistency_df.index += 1
    consistency_df['Consistency (z-Score)'] = consistency_df['Consistency (z-Score)'].apply(lambda x: f'{x:.2f}')
    return consistency_df

def team_game_log(team_list):
    valid = False
    while valid == False:
        input_team = input('Enter a team: ')
        for team_obj in team_list:
            if input_team.strip().lower() == team_obj.name.lower().replace('é','e'):
                team = team_obj
                valid = True
        if valid == False:
            print('Sorry, I am not familiar with this team. Maybe check your spelling?')

    game_log_df = pd.DataFrame(columns = ['Date', 'Opponent', 'GF', 'GA', 'Performance'])
    for game in team.team_game_list:
        if team == game.home_team:
            goals_for = game.home_score
            opponent = game.away_team
            goals_against = game.away_score
        else:
            goals_for = game.away_score
            opponent = game.home_team
            goals_against = game.home_score

        game_log_df = pd.concat([game_log_df, pd.DataFrame.from_dict([{'Date':game.date, 'Opponent':opponent.name, 'GF':int(goals_for), 'GA':int(goals_against), 'Performance':round(opponent.power + goals_for - goals_against,2)}])], ignore_index = True)
            
    game_log_df.index += 1 
    return team, game_log_df

def get_team_prob_breakdown(team_list, param):
    valid = False
    while valid == False:
        input_team = input('Enter a team: ')
        for team_obj in team_list:
            if input_team.strip().lower() == team_obj.name.lower().replace('é','e'):
                team = team_obj
                valid = True
        if valid == False:
            print('Sorry, I am not familiar with this team. Maybe check your spelling?')

    prob_breakdown_df = pd.DataFrame(columns = ['Opponent', 'Record', 'PCT', 'Win Probability', 'Lose by 5+', 'Lose by 4', 'Lose by 3', 'Lose by 2', 'Lose by 1', 'Win by 1', 'Win by 2', 'Win by 3', 'Win by 4', 'Win by 5+'])
    for opp_team in team_list:
        if opp_team is not team:
            prob_breakdown_df = pd.concat([prob_breakdown_df, pd.DataFrame.from_dict([{'Opponent': opp_team.name, 
            'Record': opp_team.record,
            'PCT': f'{opp_team.calc_pct():.3f}',
            'Win Probability':f'{calc_prob(team, opp_team, param)*100:.2f}%', 
            'Lose by 5+': f'{calc_spread(team, opp_team, param, "-inf", -4.5)*100:.2f}%',
            'Lose by 4': f'{calc_spread(team, opp_team, param, -4.5, -3.5)*100:.2f}%', 
            'Lose by 3': f'{calc_spread(team, opp_team, param, -3.5, -2.5)*100:.2f}%', 
            'Lose by 2': f'{calc_spread(team, opp_team, param, -2.5, -1.5)*100:.2f}%', 
            'Lose by 1': f'{calc_spread(team, opp_team, param, -1.5, 0)*100:.2f}%', 
            'Win by 1': f'{calc_spread(team, opp_team, param, 0, 1.5)*100:.2f}%', 
            'Win by 2': f'{calc_spread(team, opp_team, param, 1.5, 2.5)*100:.2f}%', 
            'Win by 3': f'{calc_spread(team, opp_team, param, 2.5, 3.5)*100:.2f}%', 
            'Win by 4': f'{calc_spread(team, opp_team, param, 3.5, 4.5)*100:.2f}%',
            'Win by 5+': f'{calc_spread(team, opp_team, param, 4.5, "inf")*100:.2f}%'}])], ignore_index = True)

    prob_breakdown_df = prob_breakdown_df.set_index('Opponent')
    prob_breakdown_df = prob_breakdown_df.sort_values(by=['PCT'], ascending=False)
    return team, prob_breakdown_df


