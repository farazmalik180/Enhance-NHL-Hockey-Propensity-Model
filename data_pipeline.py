import os
import pandas as pd
from utils import calculate_records, save_to_csv, load_from_csv
import requests
from tqdm import tqdm

DATA_DIR = "data/"
os.makedirs(DATA_DIR, exist_ok=True)

def scrape_nhl_data():
    data = []
    team_id_dict = {}

    # Fetch team metadata
    team_metadata = requests.get("https://api.nhle.com/stats/rest/en/team").json()
    for team in tqdm(
        team_metadata['data'], desc='Scraping Teams', dynamic_ncols=True
    ):
        # Skip inactive/historical teams
        if team['fullName'] in [
            'Atlanta Thrashers', 'Hartford Whalers', 'Minnesota North Stars', 'Quebec Nordiques',
            'Colorado Rockies', 'NHL', 'Toronto Arenas', 'California Golden Seals']:
            continue

        team_id_dict[team['id']] = team['fullName']

        # Fetch game data for past seasons
        for season_start in range(2020, 2025):  # Past 10 seasons
            season = f"{season_start}{season_start + 1}"
            try:
                game_metadata = requests.get(
                    f"https://api-web.nhle.com/v1/club-schedule-season/{team['triCode']}/{season}"
                ).json()
                for game in game_metadata.get('games', []):
                    if game['gameType'] == 2 and game['gameState'] == 'OFF':  # Regular season completed games
                        data.append({
                            'GameID': game['id'],
                            'Date': game['gameDate'],
                            'Home Team': game['homeTeam']['id'],
                            'Home Goals': game['homeTeam']['score'],
                            'Away Goals': game['awayTeam']['score'],
                            'Away Team': game['awayTeam']['id'],
                            'FinalState': game['gameOutcome']['lastPeriodType'],
                        })
            except Exception as e:
                print(f"Error fetching data for {team['fullName']} in season {season}: {e}")

    # Process data into DataFrame
    scraped_df = pd.DataFrame(data)
    scraped_df['Home Team'] = scraped_df['Home Team'].replace(team_id_dict)
    scraped_df['Away Team'] = scraped_df['Away Team'].replace(team_id_dict)
    scraped_df = scraped_df.drop_duplicates(subset='GameID')
    scraped_df = scraped_df.sort_values(by=['GameID'])

    # Calculate team records
    scraped_df = calculate_records(scraped_df)

    # Save data
    save_to_csv(scraped_df, f"{DATA_DIR}scraped_data.csv")
    save_to_csv(pd.DataFrame(list(team_id_dict.items()), columns=['ID', 'Name']), f"{DATA_DIR}team_id_dict.csv")
    return scraped_df, team_id_dict

def get_or_load_data():
    if os.path.exists(f"{DATA_DIR}scraped_data.csv"):
        return (
            load_from_csv(f"{DATA_DIR}scraped_data.csv"),
            load_from_csv(f"{DATA_DIR}team_id_dict.csv").set_index('ID').to_dict()['Name']
        )
    else:
        return scrape_nhl_data()
