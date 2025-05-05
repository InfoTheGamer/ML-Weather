import os
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score, f1_score, r2_score, root_mean_squared_error, root_mean_squared_log_error
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import glob
from datetime import datetime, timedelta


# --- Config ---
DATA_DIR = "hurricanes-24"
NUM_SAMPLES = len(os.listdir(DATA_DIR))
#NUM_SAMPLES = 256
TRAIN_SPLIT_PERCENT = 0.7
TRAIN_SPLIT = int(TRAIN_SPLIT_PERCENT * NUM_SAMPLES)
BATCH_SIZE = 64
EPOCHS = 256
LR = 1e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
START_TIME = datetime(2024, 6, 1, 00) 
TIME_STEP = timedelta(hours=1)  #Time Step Between Data pairs


# --- Utility: Pad to divisible dimensions ---
def pad_to_multiple(arr, divisor=8):
    h, w = arr.shape
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    padded = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='edge')
    return padded, (pad_top, pad_bottom, pad_left, pad_right)

def unpad(arr, pads):
    pt, pb, pl, pr = pads
    return arr[..., pt:arr.shape[-2]-pb, pl:arr.shape[-1]-pr]

# Denormalize Function
def denormalize(data, min_val, max_val):
    return data * (max_val - min_val) + min_val


# --- Load + Preprocess Data ---
# Get all matching pair_*.nc files
pair_files = sorted(glob.glob(os.path.join(DATA_DIR, "pair_*.nc")))
# Limit the number of samples if needed
pair_files = pair_files[:NUM_SAMPLES]

X, Y, pads = [], [], []
for i, file_path in enumerate(pair_files):
    with nc.Dataset(file_path) as ds:
        var0 = ds.variables["pressure_T"][:]
        var1 = ds.variables["pressure_T_plus_1"][:]

        if i == 0:
            lat = ds.variables["latitude"][:]  # Latitude
            lon = ds.variables["longitude"][:]  # Longitude

        p0 = np.ma.filled(var0, np.nan)
        p1 = np.ma.filled(var1, np.nan)

        p0 = np.nan_to_num(p0, nan=np.nanmin(p0))
        p1 = np.nan_to_num(p1, nan=np.nanmin(p1))

        p0, pad = pad_to_multiple(p0)
        p1, _ = pad_to_multiple(p1)
        pads.append(pad)

        X.append(p0)
        Y.append(p1)

X, Y = np.array(X), np.array(Y)
min_val_X, max_val_X = X.min(), X.max()
min_val_Y, max_val_Y = Y.min(), Y.max()
X = (X - min_val_X) / (max_val_X - min_val_X)
Y = (Y - min_val_Y) / (max_val_Y - min_val_Y)

X = X[:, np.newaxis, :, :]  # add channel dimension
Y = Y[:, np.newaxis, :, :]

# --- Torch Datasets ---
X_train, Y_train = X[:TRAIN_SPLIT], Y[:TRAIN_SPLIT]
X_test, Y_test = X[TRAIN_SPLIT:], Y[TRAIN_SPLIT:]
pads_test = pads[TRAIN_SPLIT:]
train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32))
test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test, dtype=torch.float32))
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)

# --- Define UNet Model ---
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc1 = self.conv_block(1, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self.conv_block(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self.conv_block(32, 64)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = self.conv_block(64, 128)

        self.up3 = self.up_conv(128, 64)
        self.dec3 = self.conv_block(128, 64)
        self.up2 = self.up_conv(64, 32)
        self.dec2 = self.conv_block(64, 32)
        self.up1 = self.up_conv(32, 16)
        self.dec1 = self.conv_block(32, 16)
        self.final = nn.Conv2d(16, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def up_conv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)

        d3 = self.up3(e4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.final(d1)

model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6)
def gradient_loss(pred, target):                        # Loss
    def get_grad(img):
        dx = img[:, :, :, 1:] - img[:, :, :, :-1]
        dy = img[:, :, 1:, :] - img[:, :, :-1, :]
        return dx, dy

    dx_p, dy_p = get_grad(pred)
    dx_t, dy_t = get_grad(target)
    return F.l1_loss(dx_p, dx_t) + F.l1_loss(dy_p, dy_t)
mse_loss = nn.MSELoss()
def combined_loss(pred, target):
    mse = mse_loss(pred, target)
    grad = gradient_loss(pred, target)
    return mse + 0.05 * grad  # tune the weight

criterion = combined_loss


# --- Training Loop ---
train_losses = []
test_losses = []

for epoch in range(EPOCHS):
    # --- Train ---
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
    avg_train_loss = running_loss / len(train_dl)
    train_losses.append(avg_train_loss)

    # --- Test Loss ---
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for xb, yb in test_dl:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            test_loss += loss.item()
    avg_test_loss = test_loss / len(test_dl)
    test_losses.append(avg_test_loss)

    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.7f} - Test Loss: {avg_test_loss:.7f}")

# --- Plot losses ---
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel("Epoch")
plt.ylabel("MSE+Gradient Loss")
plt.title("Train vs Test Loss")
plt.legend()
plt.yscale("log")
plt.grid(True, which='both', axis='x', linestyle='-', color='gray')  # Regular grid for x
plt.grid(True, which='both', axis='y', linestyle='--', color='gray')  # Log grid for y
plt.tight_layout()
plt.show()

# --- Predict ---
model.eval()
Y_pred = []
with torch.no_grad():
    for xb, _ in test_dl:
        xb = xb.to(device)
        out = model(xb).cpu().numpy()
        Y_pred.append(out)
Y_pred = np.concatenate(Y_pred, axis=0)

# --- Unpad for Evaluation ---
Y_true = Y_test
Y_pred_unpad = []
Y_true_unpad = []
for i in range(len(Y_pred)):
    Y_pred_unpad.append(unpad(Y_pred[i], pads_test[i]))
    Y_true_unpad.append(unpad(Y_true[i, 0], pads_test[i]))

Y_pred_unpad = np.array(Y_pred_unpad)
Y_true_unpad = np.array(Y_true_unpad)

# Debug
print("Y_pred_unpad shape: ", Y_pred_unpad.shape)
print("Y_true_unpad shape: ", Y_true_unpad.shape)

# --- Evaluation ---
flat_true = Y_true_unpad.flatten()
flat_pred = Y_pred_unpad.flatten()

mse = mean_squared_error(flat_true, flat_pred)
mae = mean_absolute_error(flat_true, flat_pred)

threshold = 0.05
y_true_bin = np.abs(flat_true - flat_pred) <= threshold
y_pred_bin = np.ones_like(y_true_bin)

print("Y_true_bin shape: ", y_true_bin.shape)
print("Y_pred_bin shape: ", y_pred_bin.shape)

precision = precision_score(y_true_bin, y_pred_bin)
recall = recall_score(y_true_bin, y_pred_bin)
f1 = f1_score(y_true_bin, y_pred_bin)
rmse = root_mean_squared_error(y_true_bin, y_pred_bin)
rmsle = root_mean_squared_log_error(y_true_bin, y_pred_bin)
r2 = r2_score(flat_true, flat_pred)

print("\n🔍 Evaluation Metrics:")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"RMSLE: {rmsle:.4f}")
print(f"R2: {r2:.4f}")
print(f"Precision (±{threshold} norm): {precision:.3f}")
print(f"Recall    (±{threshold} norm): {recall:.3f}")
print(f"F1 Score  (±{threshold} norm): {f1:.3f}")

# DENORMALIZED METRICS
flat_true_denorm = denormalize(flat_true, min_val_X, max_val_X)
flat_pred_denorm = denormalize(flat_pred, min_val_Y, max_val_Y)

# Recalculate metrics on denormalized data
mse_denorm = mean_squared_error(flat_true_denorm, flat_pred_denorm)
mae_denorm = mean_absolute_error(flat_true_denorm, flat_pred_denorm)
rmse_denorm = np.sqrt(mse_denorm)
r2_denorm = r2_score(flat_true_denorm, flat_pred_denorm)

print("\n📏 Denormalized Evaluation Metrics:")
print(f"MSE: {mse_denorm:.4f}")
print(f"MAE: {mae_denorm:.4f}")
print(f"RMSE: {rmse_denorm:.4f}")
print(f"R2: {r2_denorm:.4f}")

# --- Visualization of Results ---
def visualize_example(example_idx, Y_test, Y_pred, lats, lons, min_val_Y, max_val_Y):
    """
    Plot ground truth, prediction, and absolute error using Cartopy.

    Parameters:
    - example_idx: index in test set
    - Y_test: ground truth tensor (N, 1, 261, 581)
    - Y_pred: predicted tensor (N, 261, 581)
    - lats: 2D array of shape (261, 581) with latitudes
    - lons: 2D array of shape (261, 581) with longitudes
    - min_val_Y, max_val_Y: for denormalizing output
    """
    timestamp = START_TIME + example_idx * TIME_STEP
    timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M UTC')

    gt = Y_test[example_idx]
    pred = Y_pred[example_idx].squeeze()

    # Debug
    print("Y_test / gt shape: ", gt.shape)
    print("Y_pred / pred shape: ", pred.shape)

    gt_denorm = denormalize(gt, min_val_Y, max_val_Y)
    pred_denorm = denormalize(pred, min_val_Y, max_val_Y)
    error_denorm = np.abs(gt_denorm - pred_denorm)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    titles = ['Ground Truth', 'Prediction', 'Absolute Error']
    data_maps = [gt_denorm, pred_denorm, error_denorm]
    cmaps = ['viridis', 'viridis', 'Reds']

    # Find common color scale range for GT and prediction
    vmin = min(gt_denorm.min(), pred_denorm.min())
    vmax = max(gt_denorm.max(), pred_denorm.max())

    for ax, title, data, cmap in zip(axs, titles, data_maps, cmaps):
        print("Data shape:", data.shape)

        # Use shared color scale for GT and prediction, and separate for error
        if title in ['Ground Truth', 'Prediction']:
            im = ax.pcolormesh(lons, lats, data, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
        else:
            im = ax.pcolormesh(lons, lats, data, cmap=cmap, shading='auto')

        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.STATES, linewidth=0.5)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.7)

    fig.suptitle(f"Weather Prediction Example - {timestamp_str}", fontsize=16)
    plt.tight_layout()
    plt.show()

print("lons: ", lon.shape)
print("lats: ", lat.shape)


visualize_example(
    example_idx=10,
    Y_test=Y_true_unpad,
    Y_pred=Y_pred_unpad,
    lats=lat,       # shape (261, 581)
    lons=lon,       # shape (261, 581)
    min_val_Y=min_val_Y,
    max_val_Y=max_val_Y
)

# 2nd example visualization
visualize_example(
    example_idx=105,
    Y_test=Y_true_unpad,
    Y_pred=Y_pred_unpad,
    lats=lat,       # shape (261, 581)
    lons=lon,       # shape (261, 581)
    min_val_Y=min_val_Y,
    max_val_Y=max_val_Y
)

