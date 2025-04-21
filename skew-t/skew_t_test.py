import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import numpy as np
import xarray as xr
from metpy.plots import SkewT
from metpy.calc import dewpoint_from_relative_humidity, height_to_pressure_std, pressure_to_height_std
from metpy.units import units

# Load data
ds = xr.open_dataset("skew-t/reanalysis_era5_pressure_levels_midlandtest.nc")

# Select nearest grid point to your location
lat = 32.988517
lon = -106.975062
ds_point = ds.sel(latitude=lat, longitude=lon, method="nearest")

# Extract variables at the selected point
p = ds_point.pressure_level.values * units.hPa
T = ds_point.t.squeeze().metpy.convert_units("degC")
RH = ds_point.r.squeeze()
Td = dewpoint_from_relative_humidity(T, RH / 100)

# Unit conversions
u = ds_point.u.metpy.convert_units("knots")
v = ds_point.v.metpy.convert_units("knots")

# Change T, dewpoint to deg F
#T = T.metpy.convert_units('degF')
#d = Td.metpy.convert_units('degF')


# ----------------------------------------------------------------
# PLOTTING

# Create Skew-T plot
fig = plt.figure(figsize=(9, 9))
skew = SkewT(fig, rotation=45)

# Plot temperature, dewpoint, wind barbs
skew.plot(p, T, 'r')
skew.plot(p, Td, 'g')
skew.plot_barbs(p, u, v)

# Style
skew.ax.set_ylim(1000, 100)
skew.ax.set_xlim(-30,50)
skew.ax.set_title("Skew-T Log-P from ERA5")


# Add dry adiabats, moist adiabats, and mixing ratio lines
skew.plot_dry_adiabats()
skew.plot_moist_adiabats()
skew.plot_mixing_lines()


# -------------------------------
# Adding kft labels

# Grab the visible pressure range from skew
pres_min, pres_max = skew.ax.get_ylim()  # These are hPa values

# Define pressure levels (in hPa) at which to label altitudes
pressure_vals = np.array([925, 850, 700, 500, 400, 300, 250, 200]) * units.hPa
pressure_vals = pressure_vals[(pressure_vals.magnitude >= pres_max) & (pressure_vals.magnitude <= pres_min)]

altitudes_m = pressure_to_height_std(pressure_vals)  # returns meters
altitudes_ft = (altitudes_m / units.meter * 3.28084).magnitude  # convert to ft
altitudes_kft = altitudes_ft # dividing by 1000 makes everything 0

# Create a second y-axis sharing the same x-axis (but place it on the left)
alt_ax = skew.ax.twinx()
alt_ax.set_ylim(1000, 100)

alt_ax.set_yscale('log')
alt_ax.invert_yaxis()

alt_ax.set_ylim(skew.ax.get_ylim())
alt_ax.set_yticks(pressure_vals.magnitude)
alt_ax.set_yticklabels([f"{kft:.0f} kft" for kft in altitudes_kft], color='red')
alt_ax.tick_params(axis='y', colors='red', direction='in', length=5, pad=4)
alt_ax.spines['left'].set_position(('outward', 60))
alt_ax.spines['left'].set_color('red')
alt_ax.yaxis.set_label_position('left')
alt_ax.yaxis.set_ticks_position('left')
#alt_ax.set_ylabel("Altitude (kft)", color='red', fontweight='bold')

plt.subplots_adjust(left=0.2)

#print([f"{kft:.0f} kft" for kft in altitudes_kft])
alt_ax.tick_params(axis='y', colors='red', labelleft=True)


plt.show()
