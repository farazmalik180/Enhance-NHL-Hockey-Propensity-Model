from data_pipeline import get_or_load_data
from models import game_team_object_creation, assign_power, prepare_power_rankings
from analysis import XGB_Classifier, model_performance, random_forest_model
from utils import save_model, load_model
from inference import get_todays_games
from menu import menu
import time

def main():
    start_time = time.time()

    # Step 1: Load or scrape data
    games_metadf, team_id_dict = get_or_load_data()

    # Step 2: Create team and game objects
    team_list, total_game_list = game_team_object_creation(games_metadf)
    assign_power(team_list, epochs=60)
    power_df = prepare_power_rankings(team_list)

    # Step 3: Load or compute XGB_Classifier parameters
    model_data = load_model()  # Load model data
    if model_data and isinstance(model_data, dict):
        param = model_data.get('param')
        xpoints = model_data.get('xpoints', [])
        ypoints = model_data.get('ypoints', [])
    else:
        xpoints, ypoints, param = XGB_Classifier(total_game_list)
        save_model({'param': param, 'xpoints': xpoints, 'ypoints': ypoints})
        # model_performance(xpoints, ypoints, param)

    # Step 4: Fetch today's games
    date, today_games_df = get_todays_games(param, team_list, team_id_dict)

    # Step 5: Launch menu
    computation_time = time.time() - start_time
    menu(power_df, today_games_df, xpoints, ypoints, param, computation_time, total_game_list, team_list, date)


if __name__ == "__main__":
    main()

#****************************
# import time
# import torch
# from data_pipeline import get_or_load_data
# from models import game_team_object_creation, assign_power, prepare_power_rankings
# from neural_network import prepare_data, train_model, evaluate_model
# from utils import save_model, load_model
# from inference import get_todays_games
# from menu import menu


# def main():
#     start_time = time.time()

#     # Step 1: Load or scrape data
#     print("Loading or scraping game data...")
#     games_metadf, team_id_dict = get_or_load_data()

#     # Step 2: Create team and game objects
#     print("Creating team and game objects...")
#     team_list, total_game_list = game_team_object_creation(games_metadf)
#     # assign_power(team_list, epochs=100)
#     power_df = prepare_power_rankings(team_list)

#     # Step 3: Load or train the neural network model
#     print("Checking for pre-trained model...")
#     model_data = load_model()  # Load model data if it exists
#     if model_data and isinstance(model_data, dict):
#         nn_model = model_data.get('model')
#         scaler = model_data.get('scaler')
#         print("Pre-trained model loaded successfully.")
#     else:
#         print("No pre-trained model found. Training a new model...")
#         xpoints, ypoints, scaler = prepare_data(total_game_list)
#         input_size = xpoints.shape[1]
#         nn_model = train_model(xpoints, ypoints, input_size, epochs=100, batch_size=32)

#         # Save the trained model and scaler
#         save_model({'model': nn_model, 'scaler': scaler})
#         print("Model training completed and saved.")

#     # Evaluate the model on training data
#     print("Evaluating the model...")
#     xpoints, ypoints, _ = prepare_data(total_game_list)  # Ensure data matches scaler
#     train_accuracy = evaluate_model(nn_model, xpoints, ypoints)
#     print(f"Training Accuracy: {train_accuracy:.2f}%")

#     # Step 4: Fetch today's games
#     print("Fetching today's games...")
#     date, today_games_df = get_todays_games(nn_model, team_list, team_id_dict, scaler)

#     # Step 5: Launch menu
#     computation_time = time.time() - start_time
#     print("Launching the main menu...")
#     menu(power_df, today_games_df, xpoints, ypoints, nn_model, computation_time, total_game_list, team_list, date)


# if __name__ == "__main__":
#     main()




