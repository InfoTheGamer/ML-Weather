import os
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# --- Config ---
DATA_DIR = "512-data"
NUM_SAMPLES = 512
TRAIN_SPLIT_PERCENT = 0.7
TRAIN_SPLIT = int(TRAIN_SPLIT_PERCENT * NUM_SAMPLES)
BATCH_SIZE = 32
EPOCHS = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Utility: Trim to even dimensions ---
def trim_even(arr):
    h, w = arr.shape
    return arr[:h - h % 2, :w - w % 2]

# --- Load + Preprocess Data ---
X, Y = [], []
for i in range(NUM_SAMPLES):
    with nc.Dataset(os.path.join(DATA_DIR, f"pair_{i:03}.nc")) as ds:
        var0 = ds.variables["pressure_T"][:]
        var1 = ds.variables["pressure_T_plus_1"][:]

        p0 = trim_even(np.ma.filled(var0, np.nan))
        p1 = trim_even(np.ma.filled(var1, np.nan))

        p0 = np.nan_to_num(p0, nan=np.nanmin(p0))
        p1 = np.nan_to_num(p1, nan=np.nanmin(p1))

        X.append(p0)
        Y.append(p1)

X, Y = np.array(X), np.array(Y)
X = np.nan_to_num(X, nan=np.nanmin(X))
Y = np.nan_to_num(Y, nan=np.nanmin(Y))

lat_trimmed = np.linspace(24, 50, X.shape[1])
lon_trimmed = np.linspace(-130, -65, X.shape[2])
Lon_trimmed, Lat_trimmed = np.meshgrid(lon_trimmed, lat_trimmed)

min_val_X, max_val_X = X.min(), X.max()
min_val_Y, max_val_Y = Y.min(), Y.max()
X = (X - min_val_X) / (max_val_X - min_val_X)
Y = (Y - min_val_Y) / (max_val_Y - min_val_Y)

X = X[:, np.newaxis, :, :]  # add channel dimension
Y = Y[:, np.newaxis, :, :]

# --- Torch Datasets ---
X_train, Y_train = X[:TRAIN_SPLIT], Y[:TRAIN_SPLIT]
X_test, Y_test = X[TRAIN_SPLIT:], Y[TRAIN_SPLIT:]
train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32))
test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test, dtype=torch.float32))
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

# --- Define Model ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3 = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.up(x)
        return self.conv3(x)

model = SimpleCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# --- Training Loop ---
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss/len(train_dl):.4f}")

# --- Predict ---
model.eval()
with torch.no_grad():
    preds = []
    for xb, _ in test_dl:
        xb = xb.to(device)
        out = model(xb).cpu().numpy()
        preds.append(out)
Y_pred = np.concatenate(preds, axis=0)

# --- Evaluation ---
flat_true = Y_test.flatten()
flat_pred = Y_pred.flatten()
mse = mean_squared_error(flat_true, flat_pred)
mae = mean_absolute_error(flat_true, flat_pred)
threshold = 0.05
y_true_bin = np.abs(flat_true - flat_pred) <= threshold
y_pred_bin = np.ones_like(y_true_bin)

precision = precision_score(y_true_bin, y_pred_bin)
recall = recall_score(y_true_bin, y_pred_bin)
f1 = f1_score(y_true_bin, y_pred_bin)

print("\n🔍 Evaluation Metrics:")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"Precision (±{threshold} norm): {precision:.3f}")
print(f"Recall    (±{threshold} norm): {recall:.3f}")
print(f"F1 Score  (±{threshold} norm): {f1:.3f}")

# --- Visualization ---
sample_idx = 0
x_sample = X_test[sample_idx:sample_idx+1]
y_true_sample = Y_test[sample_idx]
y_pred_sample = Y_pred[sample_idx]

valid_mask = (x_sample[0, 0, :, :] * (max_val_X - min_val_X) + min_val_X) < 1e5
y_true_denorm = y_true_sample[0, :, :] * (max_val_Y - min_val_Y) + min_val_Y
y_pred_denorm = y_pred_sample[0, :, :] * (max_val_Y - min_val_Y) + min_val_Y
y_true_denorm[~valid_mask] = np.nan
y_pred_denorm[~valid_mask] = np.nan
error = np.abs(y_true_denorm - y_pred_denorm)

fig, axs = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'projection': ccrs.PlateCarree()})
titles = ["True Pressure (Pa)", "Predicted Pressure (Pa)", "Absolute Error (Pa)"]
data = [y_true_denorm, y_pred_denorm, error]

for ax, title, dat in zip(axs, titles, data):
    dat = np.flipud(dat)
    cont = ax.contourf(Lon_trimmed, Lat_trimmed, dat, cmap="viridis" if title != "Absolute Error (Pa)" else "coolwarm", transform=ccrs.PlateCarree())
    ax.set_title(title)
    ax.set_extent([-130, -65, 24, 50])
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.gridlines(draw_labels=True)
    plt.colorbar(cont, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle("CNN Prediction vs Truth (Sample from Test Set)")
plt.tight_layout()
plt.show()
