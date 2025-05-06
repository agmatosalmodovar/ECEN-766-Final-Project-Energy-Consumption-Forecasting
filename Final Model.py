import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Step 1: Data Preparation
class EnergyDataset(Dataset):
    def __init__(self, data):
        self.x = torch.tensor(data[:, 1:], dtype=torch.float32)  # Weather features
        self.y = torch.tensor(data[:, 0], dtype=torch.float32)   # Energy_MW

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# Load the merged CSV
df = pd.read_csv('merged_data.csv')
df = df.dropna()

# Normalize features
features = ['Energy_MW', 'TMAX', 'TMIN', 'PRCP']
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[features])

# Create dataset and dataloader
dataset = EnergyDataset(scaled_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Step 2: Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_size=3, hidden_dim=64, num_layers=2, nhead=2):
        super(TransformerModel, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, src):
        src = src.unsqueeze(1)  # Add sequence dimension
        x = self.input_layer(src)
        x = self.transformer(x)
        x = self.output_layer(x)
        return x.squeeze()

# Initialize model
model = TransformerModel()

# Step 3: Training
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

epochs = 30
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = loss_fn(output, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.6f}")

# Step 4: Prediction on User-Provided Weather Data
def predict_energy(user_weather_file):
    user_df = pd.read_csv(user_weather_file)
    user_df = user_df.dropna()

    # Select and normalize features
    user_weather = user_df[['TMAX', 'TMIN', 'PRCP']]
    weather_scaled = scaler.transform(
        np.hstack([np.zeros((len(user_weather), 1)), user_weather])
    )[:, 1:]  # Only weather columns

    user_weather_tensor = torch.tensor(weather_scaled, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        predictions = model(user_weather_tensor)

    predictions = predictions.numpy()

    # Inverse transform
    inverse_scaled = np.zeros((len(predictions), 4))
    inverse_scaled[:, 0] = predictions
    original_scale = scaler.inverse_transform(inverse_scaled)[:, 0]

    user_df['Predicted_Energy_MW'] = original_scale

    print(user_df[['Date', 'Predicted_Energy_MW']])
    user_df.to_csv('predicted_energy_output.csv', index=False)
    print("Predictions saved to predicted_energy_output.csv")

# Example usage after training
predict_energy('user_uploaded_weather.csv')
