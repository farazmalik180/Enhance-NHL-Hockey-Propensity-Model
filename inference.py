import os
import pandas as pd
import requests
import numpy as np

from models import Game as game
#def calc_prob(team, opponent, param):
#    return 1/(1+np.exp((team.power-opponent.power)/param))
from utils import save_model, load_model
def calc_prob(team, opponent, model):
    X_input = np.array([
    
        game.home_team.power - game.away_team.power,  # Rating Differential
        game.home_team.win_pct - game.away_team.win_pct,  # Win % Difference
        game.home_team.last_5_games_win_pct - game.away_team.last_5_games_win_pct,  # Recent Form
        game.home_advantage  # Home Advantage (1 or 0)
     
    
])
    X_input = X_input.reshape(1, -1)
    home_win_prob = model.predict(X_input)
    return home_win_prob[0]



def get_todays_games(param, team_list, team_id_dict):
    

    # Fetch today's schedule from NHL API
    response = requests.get("https://api-web.nhle.com/v1/schedule/now")
    
    if response.status_code != 200:
        print(f"API Request Failed! Status Code: {response.status_code}")
        return None, None  # Stop execution if API fails

    today_schedule = response.json()

    # Extract date safely
    date = today_schedule.get('gameWeek', [{}])[0].get('date', "Unknown Date")

    # Extract games
    games_list = today_schedule.get('gameWeek', [{}])[0].get('games', [])

    if not games_list:
        print("No games found for today.")
        return date, None  # Ensure we return a valid date, even if there are no games

    today_games_df = pd.DataFrame(columns=[
        'GameID', 'Game State', 'Home Team', 'Home Goals', 'Away Goals', 'Away Team',
        'Pre-Game Home Win Probability', 'Pre-Game Away Win Probability', 'Home Record', 'Away Record'
    ])

    for games in games_list:
        home_team_id = games['homeTeam'].get('id', None)
        away_team_id = games['awayTeam'].get('id', None)

        # Use default values if team IDs are missing
        if home_team_id is None:
            print(f"Warning: Home team ID is missing for game {games.get('id', 'Unknown')}. Using default.")
            home_team_id = -1  # Assign a placeholder ID

        if away_team_id is None:
            print(f"Warning: Away team ID is missing for game {games.get('id', 'Unknown')}. Using default.")
            away_team_id = -2  # Assign a placeholder ID

        # Ensure the team exists in the dictionary before accessing it
        if home_team_id not in team_id_dict:
            team_id_dict[home_team_id] = games['homeTeam'].get('commonName', {}).get('default', "Unknown Home Team")

        if away_team_id not in team_id_dict:
            team_id_dict[away_team_id] = games['awayTeam'].get('commonName', {}).get('default', "Unknown Away Team")

        home_team_name = team_id_dict.get(home_team_id, "Unknown Home Team")
        away_team_name = team_id_dict.get(away_team_id, "Unknown Away Team")

        # Find team objects in the team list
        home_team_obj = next((team for team in team_list if team.name == home_team_name), None)
        away_team_obj = next((team for team in team_list if team.name == away_team_name), None)

        # Use default probability if team objects are missing
        if home_team_obj is None or away_team_obj is None:
            print(f"Warning: Missing team objects for game {games.get('id', 'Unknown')}. Using default probabilities.")
            home_win_prob = 0.5  # Assign neutral probability
        else:
            home_win_prob = calc_prob(home_team_obj, away_team_obj, param)

        away_win_prob = 1 - home_win_prob

        today_games_df = pd.concat([today_games_df, pd.DataFrame.from_dict([{
            'GameID': games.get('id', 'Unknown'),
            'Game State': games.get('gameState', 'Unknown'),
            'Home Team': home_team_name,
            'Home Goals': games['homeTeam'].get('score', 0),
            'Away Goals': games['awayTeam'].get('score', 0),
            'Away Team': away_team_name,
            'Pre-Game Home Win Probability': f'{home_win_prob * 100:.2f}%',
            'Pre-Game Away Win Probability': f'{away_win_prob * 100:.2f}%',
            'Home Record': home_team_obj.record if home_team_obj else 'N/A',
            'Away Record': away_team_obj.record if away_team_obj else 'N/A'
        }])], ignore_index=True)

    return date, today_games_df


#***********************************

# import os
# import pandas as pd
# import requests
# import torch
# import numpy as np


# def calc_prob(team, opponent, model):
#     """
#     Calculate the win probability for the home team using the trained neural network model.

#     Parameters:
#     - team: The home team object.
#     - opponent: The away team object.
#     - model: The trained neural network model.

#     Returns:
#     - Probability of the home team winning.
#     """
#     # Calculate the rating differential
#     rating_diff = team.power - opponent.power

#     # Convert to tensor and predict using the model
#     input_tensor = torch.tensor([[rating_diff]], dtype=torch.float32)
#     model.eval()  # Set model to evaluation mode
#     with torch.no_grad():
#         prob = model(input_tensor).item()  # Get the predicted probability

#     return prob

#**********
# def calc_prob(team, opponent, model, scaler=None):
#     """
#     Calculate the win probability for the home team using the trained neural network model.

#     Parameters:
#     - team: The home team object.
#     - opponent: The away team object.
#     - model: The trained neural network model.
#     - scaler: Optional StandardScaler used during training for normalization.

#     Returns:
#     - Probability of the home team winning.
#     """
#     # Calculate the rating differential (feature)
#     rating_diff = np.array([[team.power - opponent.power]])  # Keep 2D for scaler compatibility

#     # Apply normalization if scaler is provided
#     if scaler is not None:
#         rating_diff = scaler.transform(rating_diff)

#     # Convert to PyTorch tensor and pass through the model
#     input_tensor = torch.tensor(rating_diff, dtype=torch.float32)
#     model.eval()  # Set model to evaluation mode
#     with torch.no_grad():
#         prob = model(input_tensor).item()  # Get the predicted probability

#     return prob



# def get_todays_games(model, team_list, team_id_dict):
#     """
#     Fetch today's games and calculate win probabilities for each game.

#     Parameters:
#     - model: The trained neural network model.
#     - team_list: List of team objects with updated powers.
#     - team_id_dict: Mapping of team IDs to team names.

#     Returns:
#     - date: The date of today's games.
#     - today_games_df: DataFrame containing today's games with probabilities.
#     """
#     today_schedule = requests.get("https://api-web.nhle.com/v1/schedule/now").json()

#     today_games_df = pd.DataFrame(
#         columns=[
#             "GameID",
#             "Game State",
#             "Home Team",
#             "Home Goals",
#             "Away Goals",
#             "Away Team",
#             "Pre-Game Home Win Probability",
#             "Pre-Game Away Win Probability",
#             "Home Record",
#             "Away Record",
#         ]
#     )

#     try:
#         date = today_schedule["gameWeek"][0]["date"]
#         for games in today_schedule["gameWeek"][0]["games"]:
#             for team in team_list:
#                 if team.name == team_id_dict[games["homeTeam"]["id"]]:
#                     home_team_obj = team
#                 elif team.name == team_id_dict[games["awayTeam"]["id"]]:
#                     away_team_obj = team

#             # Calculate probabilities using the neural network model
#             home_win_prob = calc_prob(home_team_obj, away_team_obj, model)
#             away_win_prob = 1 - home_win_prob

#             if games["gameState"] == "OFF":  # Final
#                 today_games_df = pd.concat(
#                     [
#                         today_games_df,
#                         pd.DataFrame.from_dict(
#                             [
#                                 {
#                                     "GameID": games["id"],
#                                     "Game State": "Final",
#                                     "Home Team": team_id_dict[games["homeTeam"]["id"]],
#                                     "Home Goals": games["homeTeam"]["score"],
#                                     "Away Goals": games["awayTeam"]["score"],
#                                     "Away Team": team_id_dict[games["awayTeam"]["id"]],
#                                     "Pre-Game Home Win Probability": f"{home_win_prob*100:.2f}%",
#                                     "Pre-Game Away Win Probability": f"{away_win_prob*100:.2f}%",
#                                     "Home Record": home_team_obj.record,
#                                     "Away Record": away_team_obj.record,
#                                 }
#                             ]
#                         ),
#                     ],
#                     ignore_index=True,
#                 )
#             elif games["gameState"] == "FUT":  # Pre-game
#                 today_games_df = pd.concat(
#                     [
#                         today_games_df,
#                         pd.DataFrame.from_dict(
#                             [
#                                 {
#                                     "GameID": games["id"],
#                                     "Game State": "Pre-Game",
#                                     "Home Team": team_id_dict[games["homeTeam"]["id"]],
#                                     "Home Goals": 0,
#                                     "Away Goals": 0,
#                                     "Away Team": team_id_dict[games["awayTeam"]["id"]],
#                                     "Pre-Game Home Win Probability": f"{home_win_prob*100:.2f}%",
#                                     "Pre-Game Away Win Probability": f"{away_win_prob*100:.2f}%",
#                                     "Home Record": home_team_obj.record,
#                                     "Away Record": away_team_obj.record,
#                                 }
#                             ]
#                         ),
#                     ],
#                     ignore_index=True,
#                 )
#             else:  # In progress
#                 try:
#                     today_games_df = pd.concat(
#                         [
#                             today_games_df,
#                             pd.DataFrame.from_dict(
#                                 [
#                                     {
#                                         "GameID": games["id"],
#                                         "Game State": f"Period {games['periodDescriptor']['number']}",
#                                         "Home Team": team_id_dict[games["homeTeam"]["id"]],
#                                         "Home Goals": games["homeTeam"]["score"],
#                                         "Away Goals": games["awayTeam"]["score"],
#                                         "Away Team": team_id_dict[games["awayTeam"]["id"]],
#                                         "Pre-Game Home Win Probability": f"{home_win_prob*100:.2f}%",
#                                         "Pre-Game Away Win Probability": f"{away_win_prob*100:.2f}%",
#                                         "Home Record": home_team_obj.record,
#                                         "Away Record": away_team_obj.record,
#                                     }
#                                 ]
#                             ),
#                         ],
#                         ignore_index=True,
#                     )
#                 except KeyError:
#                     today_games_df = pd.concat(
#                         [
#                             today_games_df,
#                             pd.DataFrame.from_dict(
#                                 [
#                                     {
#                                         "GameID": games["id"],
#                                         "Game State": f"Period {games['periodDescriptor']['number']}",
#                                         "Home Team": team_id_dict[games["homeTeam"]["id"]],
#                                         "Home Goals": 0,
#                                         "Away Goals": 0,
#                                         "Away Team": team_id_dict[games["awayTeam"]["id"]],
#                                         "Pre-Game Home Win Probability": f"{home_win_prob*100:.2f}%",
#                                         "Pre-Game Away Win Probability": f"{away_win_prob*100:.2f}%",
#                                         "Home Record": home_team_obj.record,
#                                         "Away Record": away_team_obj.record,
#                                     }
#                                 ]
#                             ),
#                         ],
#                         ignore_index=True,
#                     )

#         today_games_df.index += 1

#     except IndexError:
#         today_games_df = None
#         date = None

#     return date, today_games_df


#**************
# def get_todays_games(model, team_list, team_id_dict, scaler=None):
#     """
#     Fetch today's games and calculate win probabilities for each game.

#     Parameters:
#     - model: The trained neural network model.
#     - team_list: List of team objects with updated powers.
#     - team_id_dict: Mapping of team IDs to team names.
#     - scaler: Optional StandardScaler used during training for normalization.

#     Returns:
#     - date: The date of today's games.
#     - today_games_df: DataFrame containing today's games with probabilities.
#     """
#     try:
#         # Fetch today's schedule
#         today_schedule = requests.get("https://api-web.nhle.com/v1/schedule/now").json()

#         today_games_df = pd.DataFrame(
#             columns=[
#                 "GameID",
#                 "Game State",
#                 "Home Team",
#                 "Home Goals",
#                 "Away Goals",
#                 "Away Team",
#                 "Pre-Game Home Win Probability",
#                 "Pre-Game Away Win Probability",
#                 "Home Record",
#                 "Away Record",
#             ]
#         )

#         # Extract date and games
#         date = today_schedule.get("gameWeek", [{}])[0].get("date", None)
#         games = today_schedule.get("gameWeek", [{}])[0].get("games", [])

#         if not games:
#             print("No games found for today.")
#             return date, None

#         for game in games:
#             # Identify home and away teams
#             home_team_obj = None
#             away_team_obj = None

#             for team in team_list:
#                 if team.name == team_id_dict.get(game["homeTeam"]["id"]):
#                     home_team_obj = team
#                 elif team.name == team_id_dict.get(game["awayTeam"]["id"]):
#                     away_team_obj = team

#             if not home_team_obj or not away_team_obj:
#                 print(f"Unable to find team objects for game: {game}")
#                 continue

#             # Calculate probabilities using the neural network model
#             home_win_prob = calc_prob(home_team_obj, away_team_obj, model, scaler)
#             away_win_prob = 1 - home_win_prob

#             # Add game information to DataFrame
#             game_data = {
#                 "GameID": game["id"],
#                 "Home Team": team_id_dict.get(game["homeTeam"]["id"], "Unknown"),
#                 "Away Team": team_id_dict.get(game["awayTeam"]["id"], "Unknown"),
#                 "Pre-Game Home Win Probability": f"{home_win_prob * 100:.2f}%",
#                 "Pre-Game Away Win Probability": f"{away_win_prob * 100:.2f}%",
#                 "Home Record": home_team_obj.record,
#                 "Away Record": away_team_obj.record,
#             }

#             if game["gameState"] == "OFF":  # Final state
#                 game_data.update({
#                     "Game State": "Final",
#                     "Home Goals": game["homeTeam"]["score"],
#                     "Away Goals": game["awayTeam"]["score"],
#                 })
#             elif game["gameState"] == "FUT":  # Pre-game
#                 game_data.update({
#                     "Game State": "Pre-Game",
#                     "Home Goals": 0,
#                     "Away Goals": 0,
#                 })
#             else:  # In progress
#                 game_data.update({
#                     "Game State": f"Period {game.get('periodDescriptor', {}).get('number', 'Unknown')}",
#                     "Home Goals": game["homeTeam"].get("score", 0),
#                     "Away Goals": game["awayTeam"].get("score", 0),
#                 })

#             today_games_df = pd.concat(
#                 [today_games_df, pd.DataFrame([game_data])],
#                 ignore_index=True,
#             )

#         today_games_df.index += 1

#     except Exception as e:
#         print(f"Error fetching today's games: {e}")
#         return None, None

#     return date, today_games_df
