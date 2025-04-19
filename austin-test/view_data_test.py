import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- Step 1: Load the NetCDF File ---
file_path = "9982c820b33434b32fe7377df421722a.nc"  # Change this to your actual file path
ds = xr.open_dataset(file_path)

# --- Step 2: Check Dataset Structure ---
print(ds)  # Print to see actual dimensions and variable names
print(ds.data_vars)  # Print available variables

# Extract geopotential data (Z) at 800 hPa
variable = "z"  # 'geopotential' is usually stored as 'z' in ERA5 datasets
if variable not in ds:
    raise KeyError(f"Variable '{variable}' not found! Available variables: {list(ds.data_vars)}")

# Ensure we select using the correct time dimension name
time_dim = "valid_time" if "valid_time" in ds.dims else None
pressure_dim = "pressure_level" if "pressure_level" in ds.dims else None

if time_dim and pressure_dim:
    geopotential = ds[variable].isel(**{time_dim: 0, pressure_dim: 0})  # Select first time and first pressure level
elif pressure_dim:
    geopotential = ds[variable].isel(**{pressure_dim: 0})  # Only select pressure if no time dimension
else:
    geopotential = ds[variable]  # Use the full dataset if no time/pressure dimension exists

# Convert geopotential (m²/s²) to geopotential height (meters)
g = 9.80665  # Gravity (m/s²)
geopotential_height = geopotential / g

# --- Step 3: Ensure Only Latitude & Longitude Dimensions Remain ---
geopotential_height = geopotential_height.squeeze()  # Removes any singleton dimensions

if len(geopotential_height.dims) != 2:
    raise ValueError(f"DataArray must be 2D, but it has dimensions {geopotential_height.dims}")

# --- Step 4: Plot the Data ---
plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.PlateCarree())  # Define a global map projection
geopotential_height.plot.contourf(ax=ax, cmap="viridis", transform=ccrs.PlateCarree())

# Add coastlines, borders, and gridlines
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=":")
ax.gridlines(draw_labels=True)

plt.title(f"Geopotential Height (800 hPa) - {geopotential[time_dim].values if time_dim else 'Unknown Time'}")
plt.show()
