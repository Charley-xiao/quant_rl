import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MarketObserverMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(MarketObserverMLP, self).__init__()
        # Define the architecture of the MLP-based market observer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Output the predicted trend value

class MarketObserverLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, num_layers=2):
        super(MarketObserverLSTM, self).__init__()
        # Define the architecture of the LSTM-based market observer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # We take the output of the last LSTM cell
        return self.fc(lstm_out[:, -1, :])

class MarketObserver:
    def __init__(self, model_type="MLP", input_dim=10, hidden_dim=64, output_dim=1, lr=0.001):
        """
        Initializes the Market Observer model.
        :param model_type: "MLP" for a Multi-Layer Perceptron model, "LSTM" for a Long Short-Term Memory model.
        :param input_dim: Number of input features (e.g., historical asset prices).
        :param hidden_dim: Number of hidden units in the network.
        :param output_dim: Number of output units (usually 1 for predicting future trend).
        :param lr: Learning rate for the optimizer.
        """
        self.model_type = model_type
        if model_type == "MLP":
            self.model = MarketObserverMLP(input_dim, hidden_dim, output_dim)
        elif model_type == "LSTM":
            self.model = MarketObserverLSTM(input_dim, hidden_dim, output_dim)
        else:
            raise ValueError("Invalid model type. Choose 'MLP' or 'LSTM'.")

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def predict_trends(self, historical_data):
        """
        Predicts the market trends based on historical data.
        :param historical_data: A numpy array or tensor containing historical asset prices.
        :return: Predicted market trends (a tensor with predicted trend values).
        """
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            inputs = torch.tensor(historical_data, dtype=torch.float32).unsqueeze(0)
            predictions = self.model(inputs)
        return predictions

    def update_model(self, historical_data, actual_trends):
        """
        Updates the model based on historical data and the actual market trends.
        :param historical_data: A numpy array or tensor containing historical asset prices.
        :param actual_trends: A numpy array or tensor containing the actual trends to learn from.
        :return: Loss value after the update.
        """
        self.model.train()  # Set the model to training mode
        inputs = torch.tensor(historical_data, dtype=torch.float32).unsqueeze(0)
        targets = torch.tensor(actual_trends, dtype=torch.float32).unsqueeze(0)

        # Forward pass
        predictions = self.model(inputs)
        loss = self.criterion(predictions, targets)

        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
