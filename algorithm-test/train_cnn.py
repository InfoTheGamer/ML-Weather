import os
import numpy as np
import tensorflow as tf
import netCDF4 as nc
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- Config ---
DATA_DIR = "algorithm-test"
NUM_SAMPLES = 10
TRAIN_SPLIT_PERCENT = 0.7
TRAIN_SPLIT = int(TRAIN_SPLIT_PERCENT * NUM_SAMPLES)
BATCH_SIZE = 2
EPOCHS = 100

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

        data0 = np.ma.filled(var0, np.nan)
        data1 = np.ma.filled(var1, np.nan)

        p0 = trim_even(data0)
        p1 = trim_even(data1)

        p0 = np.nan_to_num(p0, nan=np.nanmin(p0))
        p1 = np.nan_to_num(p1, nan=np.nanmin(p1))

        X.append(p0)
        Y.append(p1)

X = np.array(X)
Y = np.array(Y)

# Replace any lingering NaNs
X = np.nan_to_num(X, nan=np.nanmin(X))
Y = np.nan_to_num(Y, nan=np.nanmin(Y))

# --- Create lat/lon grid for the trimmed data ---
lat_trimmed = np.linspace(24, 50, X.shape[1])
lon_trimmed = np.linspace(-130, -65, X.shape[2])
Lon_trimmed, Lat_trimmed = np.meshgrid(lon_trimmed, lat_trimmed)

# --- Normalize (min-max to [0, 1]) ---
min_val_X, max_val_X = X.min(), X.max()
min_val_Y, max_val_Y = Y.min(), Y.max()

X = (X - min_val_X) / (max_val_X - min_val_X)
Y = (Y - min_val_Y) / (max_val_Y - min_val_Y)

# --- Expand dims for CNN ---
X = np.expand_dims(X, axis=-1)
Y = np.expand_dims(Y, axis=-1)

# --- Split into train/test ---
X_train, Y_train = X[:TRAIN_SPLIT], Y[:TRAIN_SPLIT]
X_test, Y_test = X[TRAIN_SPLIT:], Y[TRAIN_SPLIT:]

# --- Define CNN Model ---
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=X.shape[1:]),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.UpSampling2D((2, 2)),
    tf.keras.layers.Conv2D(1, (3, 3), activation='linear', padding='same')
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# --- Train Model ---
history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)

# --- Predict on Test Set ---
Y_pred = model.predict(X_test)

# --- Evaluate ---
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

# --- Visualization: True vs Predicted ---
sample_idx = 0
x_sample = X_test[sample_idx:sample_idx+1]
y_true_sample = Y_test[sample_idx]
y_pred_sample = Y_pred[sample_idx]

valid_mask = (x_sample[0, :, :, 0] * (max_val_X - min_val_X) + min_val_X) < 1e5
y_true_denorm = y_true_sample[:, :, 0] * (max_val_Y - min_val_Y) + min_val_Y
y_pred_denorm = y_pred_sample[:, :, 0] * (max_val_Y - min_val_Y) + min_val_Y
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
    ax.set_extent([-130, -65, 24, 50], crs=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.gridlines(draw_labels=True)
    plt.colorbar(cont, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle("CNN Prediction vs Truth (Sample from Test Set)")
plt.tight_layout()
plt.show()
