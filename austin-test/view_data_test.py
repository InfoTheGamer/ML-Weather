import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Load data with a specified engine
file_path = "austin-test/reanalysis_era5_land_test.nc"
ds = xr.open_dataset(file_path, engine="netcdf4")

# Inspect variables
print(ds)
print(ds.data_vars)

# Use surface pressure ('sp'), convert from Pa to hPa
variable = "sp"
if variable not in ds:
    raise KeyError(f"Variable '{variable}' not found! Available variables: {list(ds.data_vars)}")

sp = ds[variable].squeeze() / 100  # hPa

# Plot
plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
sp.plot.contourf(ax=ax, cmap="viridis", transform=ccrs.PlateCarree())

ax.set_extent([-130, -65, 24, 50], crs=ccrs.PlateCarree())  # Zoom to CONUS
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=":")
ax.gridlines(draw_labels=True)

plt.title("Surface Pressure (hPa)")
plt.show()
