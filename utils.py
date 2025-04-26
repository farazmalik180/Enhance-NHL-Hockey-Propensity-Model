import pandas as pd
import pickle
import os

DATA_DIR = "data/"
MODEL_DIR = "models/"

def save_model(model, filename=f"{MODEL_DIR}saved_model.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(model, f)

def load_model(filename=f"{MODEL_DIR}saved_model.pkl"):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    return None

def save_to_csv(data, filepath):
    data.to_csv(filepath, index=False)

def load_from_csv(filepath):
    return pd.read_csv(filepath)

def calculate_records(df):
    # Combine unique teams from both 'Home Team' and 'Away Team' columns
    all_teams = pd.concat([df['Home Team'], df['Away Team']]).unique()
    
    # Initialize records dictionary for all teams
    records = {team: {'wins': 0, 'losses': 0, 'ot_losses': 0} for team in all_teams}

    for index, row in df.iterrows():
        home_team = row['Home Team']
        away_team = row['Away Team']
        home_goals = row['Home Goals']
        away_goals = row['Away Goals']
        final_state = row['FinalState']
        
        if home_goals > away_goals:
            records[home_team]['wins'] += 1
            if final_state == 'REG':
                records[away_team]['losses'] += 1
            else:
                records[away_team]['ot_losses'] += 1
        elif home_goals < away_goals:
            if final_state == 'REG':
                records[home_team]['losses'] += 1
            else:
                records[home_team]['ot_losses'] += 1
            records[away_team]['wins'] += 1
        else:
            print(f'Critical Error: Found Tie | Information: {home_team} {home_goals}-{away_goals} {away_team}') # should never happen
            return

        df.loc[index, 'Home Record'] = f"{records[home_team]['wins']}-{records[home_team]['losses']}-{records[home_team]['ot_losses']}"
        df.loc[index, 'Away Record'] = f"{records[away_team]['wins']}-{records[away_team]['losses']}-{records[away_team]['ot_losses']}"

    return df

def download_csv_option(df, filename):
    """
    Prompts the user to download a DataFrame as a CSV file.

    Parameters:
    - df: DataFrame to save.
    - filename: Name of the CSV file (without extension).
    """
    user_input = input('Would you like to download this as a CSV? (Y/N): ').strip().lower()
    if user_input in ['y', 'yes']:
        file_path = f'Output/{filename}.csv'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        print(f'File saved as {file_path}.')

def get_sportsbook_odds(date):
    """
    Fetch sportsbook odds for the given date.
    This is a placeholder function and should be implemented to fetch real data.
    """
    # Placeholder implementation
    return {
        'game1': {'team1': -110, 'team2': 100},
        'game2': {'team3': -120, 'team4': 110},
        # Add more games and odds as needed
    }





