import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature

padval = 0.1
fracval = 0.04

def plot_pair_with_cartopy(file_path):
    """Plot the surface pressure for T, T+1, and the difference (T+1 - T) with Cartopy."""
    with nc.Dataset(file_path) as ds:
        # Extract variables from the dataset
        lat = ds.variables["latitude"][:]  # Latitude
        lon = ds.variables["longitude"][:]  # Longitude
        pressure_T = ds.variables["pressure_T"][:]  # Pressure at time t

        # Testing/debug
        print(np.amax(pressure_T))

        pressure_T_plus_1 = ds.variables["pressure_T_plus_1"][:]  # Pressure at time t+1

    # Create meshgrid for lat and lon
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Calculate the difference between T+1 and T
    pressure_diff = pressure_T_plus_1 - pressure_T

    # Create a figure with subplots for T, T+1, and difference
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': ccrs.PlateCarree()})

    # Set the extent to zoom into the U.S. (CONUS)
    for ax in axes:
        ax.set_extent([-130, -65, 24, 50], crs=ccrs.PlateCarree())  # Zoom to CONUS
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.gridlines(draw_labels=True)

    # Plot T (Pressure at Time T)
    c1 = axes[0].contourf(lon_grid, lat_grid, pressure_T, cmap="viridis", transform=ccrs.PlateCarree())
    axes[0].set_title("Surface Pressure at Time T")
    # Adjust the colorbar
    fig.colorbar(c1, ax=axes[0], label="Surface Pressure (Pa)", fraction=fracval, pad=padval)

    # Plot T+1 (Pressure at Time T+1)
    c2 = axes[1].contourf(lon_grid, lat_grid, pressure_T_plus_1, cmap="viridis", transform=ccrs.PlateCarree())
    axes[1].set_title("Surface Pressure at Time T+1")
    # Adjust the colorbar
    fig.colorbar(c2, ax=axes[1], label="Surface Pressure (Pa)", fraction=fracval, pad=padval)

    # Plot the difference (T+1 - T)
    c3 = axes[2].contourf(lon_grid, lat_grid, pressure_diff, cmap="coolwarm", transform=ccrs.PlateCarree())
    axes[2].set_title("Pressure Difference (T+1 - T)")
    # Adjust the colorbar
    fig.colorbar(c3, ax=axes[2], label="Pressure Difference (Pa)", fraction=fracval, pad=padval)

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()

    # Testing/debug
    print(np.amax(pressure_T))
    print(np.amax(pressure_T_plus_1))

# Test the plotting function with a specific file
file_path = "algorithm-test/pair_008.nc"  # Update this to your file path
plot_pair_with_cartopy(file_path)


