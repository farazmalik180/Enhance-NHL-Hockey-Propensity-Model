from utils import download_csv_option
from analysis import get_upsets, get_best_performances, get_team_consistency, team_game_log, get_team_prob_breakdown

def extra_menu(total_game_list, team_list, param):
    while True:
        print("""--EXTRAS MENU--
    1. Biggest Upsets
    2. Best Performances
    3. Most Consistent Teams
    4. Team Game Logs
    5. Team Probability Big Board
    6. Exit to Main Menu""")

        valid = False
        while valid == False:
            user_option = input('Enter a menu option: ')
            try:
                user_option = int(user_option)
                if user_option >= 1 and user_option <= 6:
                    print()
                    valid = True
                else:
                    raise ValueError
            except ValueError:
                print(f'Your option "{user_option}" is invalid.', end=' ')

        if user_option == 1:
            upsets = get_upsets(total_game_list)
            print(upsets)
            download_csv_option(upsets, 'biggest_upsets')
        elif user_option == 2:
            performances = get_best_performances(total_game_list)
            print(performances)
            download_csv_option(performances, 'best_performances')
        elif user_option == 3:
            consistency = get_team_consistency(team_list)
            print(consistency)
            download_csv_option(consistency, 'most_consistent_teams')
        elif user_option == 4:
            team, game_log = team_game_log(team_list)
            print(game_log)
            download_csv_option(game_log, f'{team.name.replace(" ", "_").lower()}_game_log')
        elif user_option == 5:
            team, team_probabilities = get_team_prob_breakdown(team_list, param)
            print(team_probabilities)
            download_csv_option(team_probabilities, f'{team.name.replace(" ", "_").lower()}_prob_breakdown')
        elif user_option == 6:
            pass

        return
