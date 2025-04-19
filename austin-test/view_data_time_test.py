import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- Step 1: Load the NetCDF File ---
file_path = "faab6a38c35d22663235e55e847d8104.nc"  # Change this to your actual file path
ds = xr.open_dataset(file_path)

# --- Step 2: Check Dataset Structure ---
print(ds)  # Check dimensions and variables to ensure everything is correct

# Extract geopotential data (Z) for the pressure level (e.g., 800 hPa) and variable
variable = "z"  # 'geopotential' is typically 'z' in ERA5 datasets
if variable not in ds:
    raise KeyError(f"Variable '{variable}' not found! Available variables: {list(ds.data_vars)}")

# Check if we have time and pressure_level dimensions
time_dim = "valid_time" if "valid_time" in ds.dims else None
pressure_dim = "pressure_level" if "pressure_level" in ds.dims else None

# Get the geopotential data (assuming it's at pressure level 800 hPa)
if time_dim and pressure_dim:
    geopotential = ds[variable].isel(pressure_level=0)  # Select first pressure level (800 hPa)
elif pressure_dim:
    geopotential = ds[variable].isel(pressure_level=0)  # Select first pressure level
else:
    geopotential = ds[variable]  # Use the full dataset if no time/pressure dimension exists

# Convert geopotential (m²/s²) to geopotential height (meters)
g = 9.80665  # Gravity (m/s²)
geopotential_height = geopotential / g

# --- Step 3: Ensure the data is 2D for plotting ---
geopotential_height = geopotential_height.squeeze()  # Removes singleton dimensions
if len(geopotential_height.dims) != 3:
    raise ValueError(f"Expected 3D data (time, lat, lon), but got {geopotential_height.dims}")

# --- Step 4: Set Up Plot ---
fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()})

# --- Step 5: Plot Initial Data ---
geopotential_at_time = geopotential_height.isel(valid_time=0)  # Get first time slice
contour = ax.contourf(
    geopotential_at_time['longitude'], geopotential_at_time['latitude'],
    geopotential_at_time, cmap="viridis", transform=ccrs.PlateCarree()
)

# Add coastlines, borders, and gridlines
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=":")
ax.gridlines(draw_labels=True)


# --- Step 6: Create the Animation Function ---
def animate(i):
    # Update the data for the current time step
    geopotential_at_time = geopotential_height.isel(valid_time=i)  # Get data for time i

    # Clear the axis before plotting the new contour
    ax.clear()

    # Reapply map features (since ax.clear() removes everything)
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.gridlines(draw_labels=True)

    # Plot the new contour data for the current time step
    contour = ax.contourf(
        geopotential_at_time['longitude'], geopotential_at_time['latitude'],
        geopotential_at_time, cmap="viridis", transform=ccrs.PlateCarree()
    )

    # Reset title with the current time
    ax.set_title(f"Geopotential Height (800 hPa) at {str(geopotential_height['valid_time'].values[i])}")


# --- Step 7: Create the Animation ---
ani = animation.FuncAnimation(
    fig, animate, frames=len(geopotential_height['valid_time']), interval=50, repeat=True
)

# Show the animation
plt.show()
