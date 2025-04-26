import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# class GameOutcomeNN(nn.Module):
#     def __init__(self, input_size):
#         super(GameOutcomeNN, self).__init__()
#         self.hidden = nn.Sequential(
#             nn.Linear(input_size, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, 1),
#             nn.Sigmoid(),  # Output is a probability
#         )

#     def forward(self, x):
#         return self.hidden(x)


# def prepare_data(total_game_list):
#     """
#     Prepare data for training the neural network.

#     Parameters:
#     - total_game_list: List of game objects.

#     Returns:
#     - X: Features (rating differential).
#     - y: Labels (game outcomes).
#     """
#     X, y = [], []

#     for game in total_game_list:
#         # Feature: Rating differential (Home - Away)
#         X.append([game.home_team.power - game.away_team.power])

#         # Label: Win/Loss outcome
#         if game.home_score > game.away_score:
#             y.append(1)
#         elif game.home_score < game.away_score:
#             y.append(0)
#         else:
#             y.append(0.5)  # Ties are treated as 0.5

#     return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# def train_neural_net(total_game_list, epochs=100, learning_rate=0.001):
#     """
#     Train a neural network to predict game outcomes.

#     Parameters:
#     - total_game_list: List of game objects.
#     - epochs: Number of training epochs.
#     - learning_rate: Learning rate for the optimizer.

#     Returns:
#     - X: Features used for training.
#     - y: Labels used for training.
#     - model: Trained neural network model.
#     """
#     # Prepare data
#     X, y = prepare_data(total_game_list)
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Create model
#     model = GameOutcomeNN(input_size=X.shape[1])
#     criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#     # Convert data to PyTorch tensors
#     X_train_tensor = torch.tensor(X_train)
#     y_train_tensor = torch.tensor(y_train).unsqueeze(1)  # Add dimension for BCELoss
#     X_val_tensor = torch.tensor(X_val)
#     y_val_tensor = torch.tensor(y_val).unsqueeze(1)

#     # Training loop
#     for epoch in range(epochs):
#         model.train()
#         optimizer.zero_grad()
#         outputs = model(X_train_tensor)
#         loss = criterion(outputs, y_train_tensor)
#         loss.backward()
#         optimizer.step()

#         if (epoch + 1) % 10 == 0:
#             model.eval()
#             val_outputs = model(X_val_tensor).detach().numpy()
#             val_outputs = np.round(val_outputs)  # Convert probabilities to binary outputs
#             accuracy = accuracy_score(y_val, val_outputs)
#             print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Validation Accuracy: {accuracy:.2f}")

#     return X, y, model


# def predict_neural_net(model, X):
#     """
#     Predict outcomes using the trained neural network model.

#     Parameters:
#     - model: Trained neural network model.
#     - X: Input features for prediction.

#     Returns:
#     - Predictions as probabilities.
#     """
#     model.eval()
#     X_tensor = torch.tensor(X)
#     with torch.no_grad():
#         predictions = model(X_tensor).numpy()
#     return predictions


import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, log_loss
from torch.nn.functional import binary_cross_entropy_with_logits
# Define the neural network model
class GameOutcomeNN(nn.Module):
    def __init__(self, input_size):
        super(GameOutcomeNN, self).__init__()
        self.model = nn.Sequential(
                    nn.Linear(input_size, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )

    def forward(self, x):
        return self.model(x)

# Enhanced data preparation
def prepare_data(total_game_list):
    xpoints = []
    ypoints = []

    for game in total_game_list:
        features = [
            game.home_team.power - game.away_team.power,  # Power difference
            game.home_team.goals_for - game.away_team.goals_against,  # Offensive/Defensive performance
            game.home_team.goals_against - game.away_team.goals_for,  # Defensive/Offensive performance
            game.home_team.pct - game.away_team.pct  # Win percentage difference
        ]
        xpoints.append(features)

        if game.home_score > game.away_score:
            ypoints.append(1)
        elif game.home_score < game.away_score:
            ypoints.append(0)
        else:
            ypoints.append(0.5)  # Handle ties

    xpoints = np.array(xpoints)
    ypoints = np.array(ypoints)

    scaler = StandardScaler()
    xpoints = scaler.fit_transform(xpoints)

    return xpoints, ypoints, scaler

# Train the model with enhancements
def train_model(xpoints, ypoints, input_size, epochs=200, batch_size=64, learning_rate=0.0005):
    # Split data
    x_train, x_val, y_train, y_val = train_test_split(xpoints, ypoints, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    x_val = torch.tensor(x_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    # Initialize model, loss function, and optimizer
    model = GameOutcomeNN(input_size)
    criterion = nn.BCELoss()  # Weighted Binary Cross-Entropy Loss
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train)
       

        # Calculate weights for positive and negative classes
        positive_weight = len(ypoints) / sum(ypoints)
        negative_weight = len(ypoints) / (len(ypoints) - sum(ypoints))

        # During training
        loss = binary_cross_entropy_with_logits(outputs, y_train, pos_weight=torch.tensor([positive_weight]))

        
        # loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(x_val)
            val_loss = criterion(val_outputs, y_val)
            val_predictions = (val_outputs > 0.5).float()
            acc = accuracy_score(y_val.numpy(), val_predictions.numpy())
            f1 = f1_score(y_val.numpy(), val_predictions.numpy(), average="weighted")

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Accuracy: {acc*100:.2f}%, F1 Score: {f1:.2f}")

    return model


# Evaluate the model
def evaluate_model(model, x_test, y_test):
    model.eval()
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        outputs = model(x_test)
        predictions = (outputs > 0.5).float()
        accuracy = accuracy_score(y_test.numpy(), predictions.numpy())
        f1 = f1_score(y_test.numpy(), predictions.numpy(), average="weighted")
    print(f"Test Accuracy: {accuracy * 100:.2f}%, F1 Score: {f1:.2f}")
    return accuracy, f1


