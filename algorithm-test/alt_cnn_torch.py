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
EPOCHS = 10
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

def center_crop(tensor, target_tensor):
    # Ensure tensor has 4 dimensions by adding batch dimension if needed
    if tensor.ndim == 3:
        tensor = tensor[np.newaxis, :, :, :]
    if target_tensor.ndim == 3:
        target_tensor = target_tensor[np.newaxis, :, :, :]

    _, _, h, w = target_tensor.shape
    tensor_h, tensor_w = tensor.shape[2], tensor.shape[3]
    delta_h = (tensor_h - h) // 2
    delta_w = (tensor_w - w) // 2
    return tensor[:, :, delta_h:delta_h + h, delta_w:delta_w + w]

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Encoder
        self.enc1 = self.conv_block(1, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self.conv_block(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self.conv_block(32, 64)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = self.conv_block(64, 128)

        # Decoder
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
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)

        # Decoder
        d3 = self.up3(e4)
        e3_cropped = center_crop(e3, d3)
        d3 = self.dec3(torch.cat([d3, e3_cropped], dim=1))

        d2 = self.up2(d3)
        e2_cropped = center_crop(e2, d2)
        d2 = self.dec2(torch.cat([d2, e2_cropped], dim=1))

        d1 = self.up1(d2)
        e1_cropped = center_crop(e1, d1)
        d1 = self.dec1(torch.cat([d1, e1_cropped], dim=1))

        out = self.final(d1)
        return out

model = UNet().to(device)
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.MSELoss()

# --- Training Loop ---
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        # Crop ground truth to match prediction size
        yb_cropped = center_crop(yb, preds)

        # Compute loss
        loss = criterion(preds, yb_cropped)

        #
        #loss = criterion(preds, yb)
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
# Assuming Y_test and Y_pred are torch tensors
Y_test_cropped = center_crop(Y_test, Y_pred)  # Crop Y_test to match Y_pred's size

# Flatten the tensors for evaluation
flat_true = Y_test_cropped.flatten()
flat_pred = Y_pred.flatten()

# Compute metrics
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

# Ensure the shapes match by cropping the ground truth to match the prediction size
if Y_pred.ndim == 3:
    Y_pred = Y_pred[np.newaxis, :, :, :]
y_true_sample_cropped = center_crop(y_true_sample, y_pred_sample)

# Create the valid mask based on some condition on the input data
valid_mask = (x_sample[0, 0, :, :] * (max_val_X - min_val_X) + min_val_X) < 1e5

# Denormalize the ground truth and prediction
y_true_denorm = y_true_sample_cropped[0, :, :] * (max_val_Y - min_val_Y) + min_val_Y
y_pred_denorm = y_pred_sample[0, :, :] * (max_val_Y - min_val_Y) + min_val_Y

# Crop valid_mask to match the shape of y_true_denorm and y_pred_denorm
valid_mask_cropped = valid_mask[:y_true_denorm.shape[0], :y_true_denorm.shape[1]]
valid_mask_cropped = np.squeeze(valid_mask_cropped, axis=0)

print("y_pred_denorm shape:", y_pred_denorm.shape)
print("y_true_denorm shape:", y_true_denorm.shape)
print("valid_mask_cropped shape:", valid_mask_cropped.shape)

# Apply mask: keep only valid rows
y_true_denorm = y_true_denorm.squeeze(0)[valid_mask_cropped]
y_pred_denorm = y_pred_denorm[valid_mask_cropped]

# Calculate the error between the denormalized ground truth and prediction
error = np.abs(y_true_denorm - y_pred_denorm)

# Visualization part
fig, axs = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'projection': ccrs.PlateCarree()})
titles = ["True Pressure (Pa)", "Predicted Pressure (Pa)", "Absolute Error (Pa)"]
data = [y_true_denorm, y_pred_denorm, error]

# Fix for wrong lat/long being trimmed:
print("Original:", Lon_trimmed.shape)  # (260, 580)
Lat_trimmed = Lat_trimmed[2:-2, 2:-2]
Lon_trimmed = Lon_trimmed[2:-2, 2:-2]
print("Trimmed:", Lon_trimmed.shape)   # Should be (256, 576)

for ax, title, dat in zip(axs, titles, data):
    dat = np.flipud(dat)  # Flip the data if necessary (depends on the projection)
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
