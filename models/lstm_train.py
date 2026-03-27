import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("final_dataset.csv")
data = df["queue"].values.reshape(-1, 1)

# =========================
# SCALE DATA
# =========================
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# =========================
# CREATE SEQUENCES
# =========================
SEQ_LEN = 7

X, y = [], []
for i in range(len(data) - SEQ_LEN):
    X.append(data[i:i+SEQ_LEN])
    y.append(data[i+SEQ_LEN])

X = np.array(X)
y = np.array(y)

# =========================
# TRAIN-TEST SPLIT
# =========================
split = int(0.8 * len(X))

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Convert to torch
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# =========================
# MODEL
# =========================
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=64,
            num_layers=2,          # ✅ fixes dropout warning
            batch_first=True,
            dropout=0.1
        )
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

model = LSTMModel()

# =========================
# TRAINING SETUP
# =========================
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
loss_fn = nn.MSELoss()

train_losses = []
val_losses = []

# =========================
# EARLY STOPPING SETUP
# =========================
best_val_loss = float("inf")
patience = 5
counter = 0

# =========================
# TRAIN LOOP
# =========================
EPOCHS = 40

for epoch in range(EPOCHS):

    model.train()

    pred = model(X_train)
    loss = loss_fn(pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    optimizer.step()

    # =========================
    # VALIDATION
    # =========================
    model.eval()
    with torch.no_grad():
        val_pred = model(X_test)
        val_loss = loss_fn(val_pred, y_test)

    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

    print(f"Epoch {epoch+1} | Train: {loss.item():.4f} | Val: {val_loss.item():.4f}")

    # =========================
    # EARLY STOPPING
    # =========================
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "models/best_lstm_model.pth")
        counter = 0
    else:
        counter += 1

    if counter >= patience:
        print("⛔ Early stopping triggered")
        break

print("✅ Model saved")

# =========================
# FINAL PREDICTION
# =========================
model.eval()

last_seq = torch.tensor(X[-1], dtype=torch.float32).unsqueeze(0)
pred = model(last_seq).detach().numpy()

pred_real = scaler.inverse_transform(pred)

print("\n🔮 Predicted Next Queue:", round(pred_real[0][0], 2))

# =========================
# DEBUG PRINT
# =========================
print("\nLoss list lengths:", len(train_losses), len(val_losses))

# =========================
# PLOT GRAPH (FIXED)
# =========================
plt.figure(figsize=(8,5))

plt.plot(train_losses, label="Train Loss", color="blue", linewidth=2)
plt.plot(val_losses, label="Validation Loss", color="red", linewidth=2)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")

plt.legend()
plt.grid(True)

plt.show()