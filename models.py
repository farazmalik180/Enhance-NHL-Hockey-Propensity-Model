import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
from sklearn.metrics import log_loss
from tqdm import tqdm
import requests
import json
import os
import time

class Team():
    def __init__(self, name):
        self.name = name
        self.team_game_list = []
        self.agd = 0
        self.opponent_power = []
        self.schedule = 0
        self.power = 0
        self.prev_power = 0
        self.goals_for = 0
        self.goals_against = 0
        self.record = '0-0-0'  # Format: Wins-Losses-OTL
        self.pct = 0
        self.win_pct = 0  # Win percentage (calculated later)
        self.last_5_games_win_pct = 0  # Last 5 games win percentage
        self.home_advantage = 0  # 1 if home, 0 if away
 


    def calc_agd(self):
        """ Calculates the average goal differential per game. """
        goal_differential = sum(
            (game.home_score - game.away_score) if self == game.home_team else (game.away_score - game.home_score)
            for game in self.team_game_list
        )
        return goal_differential / len(self.team_game_list) if self.team_game_list else 0

    def calc_agd(self):
        goal_differential = 0
        for game in self.team_game_list:
            if self == game.home_team: 
                goal_differential += game.home_score - game.away_score
            else: 
                goal_differential += game.away_score - game.home_score
        agd = goal_differential / len(self.team_game_list)

        return agd

    def calc_sched(self):
        """ Calculates the average opponent power rating. """
        self.opponent_power = [
            (game.away_team.prev_power if self == game.home_team else game.home_team.prev_power)
            for game in self.team_game_list
        ]
        return sum(self.opponent_power) / len(self.opponent_power) if self.opponent_power else 0

    def calc_power(self):
        return self.calc_sched() + self.agd

    def calc_pct(self):
        wins = int(self.record[:self.record.find('-')])
        losses = int(self.record[len(str(wins))+1:][:self.record[len(str(wins))+1:].find('-')])
        otl = int(self.record[len(str(losses))+len(str(wins))+2:])
        point_percentage = (wins*2+otl)/(len(self.team_game_list)*2)
        return point_percentage
    
    def calc_win_pct(self):
        """ Calculates the season win percentage from the record. """
        try:
            wins, losses, otl = map(int, self.record.split('-'))
            total_games = wins + losses + otl
            return (wins * 2 + otl) / (total_games * 2) if total_games > 0 else 0
        except ValueError:
            return 0

    def calc_last_5_games_win_pct(self):
        """ Calculates win percentage in the last 5 games. """
        last_5_results = self.team_game_list[-5:]  # Get last 5 games
        wins = sum(1 for game in last_5_results if (self == game.home_team and game.home_score > game.away_score) or
                   (self == game.away_team and game.away_score > game.home_score))
        return wins / 5 if len(last_5_results) == 5 else 0

    def update_statistics(self):
        """ Updates all calculated statistics for the team. """
        self.agd = self.calc_agd()
        self.pct = self.calc_win_pct()
        self.win_pct = self.calc_win_pct()
        self.last_5_games_win_pct = self.calc_last_5_games_win_pct()
 

    def calc_consistency(self):
        """ Calculates variance in team performance across games. """
        performance_list = [
            (game.away_team.power + game.home_score - game.away_score) if self == game.home_team else
            (game.away_team.power + game.away_score - game.home_score)
            for game in self.team_game_list
        ]
        return np.var(performance_list) if performance_list else 0

class Game():
    def __init__(self, home_team, away_team, home_score, away_score, date):
        self.home_team = home_team
        self.away_team = away_team
        self.home_score = home_score
        self.away_score = away_score
        self.date = date
        self.home_advantage = 1 
def game_team_object_creation(games_metadf):
    total_game_list = []
    team_list = []

    for index, row in games_metadf.iterrows():
        try:
            row['Home Goals'] = float(row['Home Goals'])
            row['Away Goals'] = float(row['Away Goals'])

            # Find or create home team
            home_team_obj = next((team for team in team_list if team.name == row['Home Team']), None)
            if home_team_obj is None:
                home_team_obj = Team(row['Home Team'])
                team_list.append(home_team_obj)

            # Find or create away team
            away_team_obj = next((team for team in team_list if team.name == row['Away Team']), None)
            if away_team_obj is None:
                away_team_obj = Team(row['Away Team'])
                team_list.append(away_team_obj)

            # Create game object
            game_obj = Game(home_team_obj, away_team_obj, row['Home Goals'], row['Away Goals'], row['Date'])
            total_game_list.append(game_obj)

            # Assign game to teams
            home_team_obj.team_game_list.append(game_obj)
            away_team_obj.team_game_list.append(game_obj)

            # Update goals stats
            home_team_obj.goals_for += game_obj.home_score
            away_team_obj.goals_against += game_obj.home_score
            home_team_obj.goals_against += game_obj.away_score
            away_team_obj.goals_for += game_obj.away_score

            # Assign records
            home_team_obj.record = row['Home Record']
            away_team_obj.record = row['Away Record']

        except ValueError:
            pass

    # Update statistics after all games are processed
    for team in team_list:
        team.update_statistics()

    return team_list, total_game_list

def assign_power(team_list, epochs):
    """
    Updates the power ratings of teams over a specified number of epochs.
    
    Parameters:
    - team_list: List of Team objects to update.
    - epochs: Number of iterations to refine team powers.
    """
    for team in team_list:
        team.agd = team.calc_agd()
        team.pct = team.calc_pct()

    for epoch in range(epochs):  # Rename loop variable to avoid shadowing
        # print(f'EPOCH {epoch + 1}')
        for team in team_list:
            team.schedule = team.calc_sched()
            team.power = team.calc_power()
            # print(f'{team.name}\tAGD: {team.calc_agd():.2f}\tSCHEDULE: {team.schedule:.2f}\tPOWER: {team.power:.2f}')
        for team in team_list:
            team.prev_power = team.power

def prepare_power_rankings(team_list):
    power_df = pd.DataFrame()
    for team in team_list:
        power_df = pd.concat([power_df, pd.DataFrame.from_dict([{'Team':team.name, 'POWER':round(team.power,2), 'Record':team.record, 'PCT':f"{team.calc_pct():.3f}",'Avg Goal Differential':round(team.calc_agd(),2), 'GF/Game':f"{team.goals_for/len(team.team_game_list):.2f}", 'GA/Game':f"{team.goals_against/len(team.team_game_list):.2f}", 'Strength of Schedule':f"{team.schedule:.3f}"}])], ignore_index=True)
    power_df.sort_values(by=['POWER'], inplace=True, ascending=False)
    power_df = power_df.reset_index(drop=True)
    power_df.index += 1 

    return power_df
