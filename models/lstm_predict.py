import torch
import torch.nn as nn
import numpy as np

# 🔥 SAME ARCHITECTURE AS TRAINING
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


# 🔥 LOAD TRAINED MODEL
model = LSTMModel()
model.load_state_dict(torch.load("models/best_lstm_model.pth"))
model.eval()


# 🔮 PREDICTION FUNCTION
def predict_next_queue(sequence):

    sequence = np.array(sequence).reshape(1, len(sequence), 1)
    sequence = torch.FloatTensor(sequence)

    with torch.no_grad():
        prediction = model(sequence)

    return prediction.item()