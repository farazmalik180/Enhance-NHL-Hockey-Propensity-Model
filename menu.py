from extra_menu import extra_menu
from utils import download_csv_option
from analysis import custom_game_selector, model_performance

def menu(power_df, today_games_df, xpoints, ypoints, param, computation_time, total_game_list, team_list, date):
    """
    Main menu function to provide options for analysis and visualization.
    """
    def view_power_rankings():
        print(power_df)
        download_csv_option(power_df, 'power_rankings')

    def view_todays_games():
        if today_games_df is not None:
            print(today_games_df)
            download_csv_option(today_games_df, f'{date}_games')
        else:
            print('There are no games today!')

    def custom_game_selector_menu():
        home_team, away_team, custom_game_df = custom_game_selector(param, team_list)
        print(custom_game_df)
        download_csv_option(custom_game_df, f'{home_team.name.replace(" ", "_").lower()}_vs_{away_team.name.replace(" ", "_").lower()}_game_probabilities')

    def view_model_performance():
        model_performance(xpoints, ypoints, param)

    def view_program_performance():
        print(f'Computation Time: {computation_time:.2f} seconds')
        print(f'Games Scraped: {len(total_game_list)}')
        print(f'Rate: {len(total_game_list)/computation_time:.1f} games/second')

    # Menu option dispatch table
    menu_options = {
        1: view_power_rankings,
        2: view_todays_games,
        3: custom_game_selector_menu,
        4: view_model_performance,
        5: view_program_performance,
        6: lambda: extra_menu(total_game_list, team_list, param),  # Call extra_menu from extra_menu.py
        7: exit
    }

    while True:
        print("""
        --MAIN MENU--
        1. View Power Rankings
        2. View Today's Games
        3. Custom Game Selector
        4. View Model Performance
        5. View Program Performance
        6. Extra Options
        7. Quit
        """)
        try:
            user_option = int(input('Enter a menu option: '))
            if user_option in menu_options:
                print()
                menu_options[user_option]()  # Call the corresponding function
            else:
                print(f'Invalid option. Please select a number between 1 and 7.')
        except ValueError:
            print("Invalid input. Please enter a valid number.")

        input('\nPress ENTER to continue...')
        print()
