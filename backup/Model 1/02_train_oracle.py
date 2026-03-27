import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os

# 1. The Neural Network Architecture
class TrafficLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(TrafficLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1) # Predicts 1 value (next vehicle count)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) # Take the last time step's output
        return out

def train_oracle():
    # 2. Load and Prepare Data
    print("--- Loading traffic_data.csv ---")
    df = pd.read_csv("data/traffic_data.csv")
    
    # We use 'vehicle_count' as our primary feature
    data = df['vehicle_count'].values.astype(float)
    
    # Normalize data (0 to 1 scale) to help the AI learn faster
    max_val = data.max() if data.max() > 0 else 1
    data_norm = data / max_val

    # Create sequences: Use 10 minutes of past data to predict the 11th minute
    window = 10
    X, y = [], []
    for i in range(len(data_norm) - window):
        X.append(data_norm[i : i + window])
        y.append(data_norm[i + window])
    
    X = torch.FloatTensor(np.array(X)).view(-1, window, 1)
    y = torch.FloatTensor(np.array(y)).view(-1, 1)

    # 3. Training Setup
    model = TrafficLSTM()
    criterion = nn.MSELoss() # Mean Squared Error (Good for prediction)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("--- Starting Training (100 Epochs) ---")
    for epoch in range(100):
        model.train()
        outputs = model(X)
        loss = criterion(outputs, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.6f}")

    # 4. Save the Brain
    if not os.path.exists('models'): os.makedirs('models')
    torch.save({
        'model_state': model.state_dict(),
        'max_val': max_val
    }, "models/oracle_brain.pth")
    
    print("\n✅ SUCCESS: 'models/oracle_brain.pth' created.")
    print("Your AI now understands the traffic patterns of your intersection.")

if __name__ == "__main__":
    train_oracle()