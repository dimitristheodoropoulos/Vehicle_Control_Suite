import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import joblib  

class FuelPredictionModel(nn.Module):
    def __init__(self):
        super(FuelPredictionModel, self).__init__()
        self.fc1 = nn.Linear(5, 64)  # Adjust input features to 5 (4 features + 1 RF output)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Initialize model
nn_model = FuelPredictionModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(nn_model.parameters(), lr=0.001)

# Load Random Forest model
rf_model = joblib.load("random_forest_model.pkl")

# Predict using Random Forest first, then refine with neural network
def hybrid_predict(input_data):
    # Ensure input_data is in the correct format (torch tensor)
    if isinstance(input_data, np.ndarray):
        input_data = torch.tensor(input_data, dtype=torch.float32)
    elif isinstance(input_data, torch.Tensor):
        input_data = input_data.float()

    # Ensure input_data is a 2D tensor
    if input_data.ndimension() == 1:
        input_data = input_data.unsqueeze(0)  # Convert 1D to 2D tensor

    # Initial prediction by Random Forest
    rf_pred = rf_model.predict(input_data.numpy())  # Random forest prediction
    rf_pred_tensor = torch.tensor(rf_pred, dtype=torch.float32).view(-1, 1)  # Reshape to column tensor

    # Concatenate RF output with original features (5 features: 4 from input data + 1 from RF)
    nn_input = torch.cat([input_data, rf_pred_tensor], dim=1)  # Concatenate along the feature axis (dim=1)
    
    # Make refined prediction using the neural network
    refined_prediction = nn_model(nn_input)  # Forward pass through the neural network

    return refined_prediction.item()
