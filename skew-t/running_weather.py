
import netCDF4 as nc
from rocketpy import Environment

'''
# Open the NetCDF file
dataset = nc.Dataset("era5_spaceport.nc")

# View the available variables
print(dataset.variables.keys())

# Inspect the shape and values of the variables you are interested in
temperature = dataset.variables["t"][:]
relative_humidity = dataset.variables["r"][:]
u_wind = dataset.variables["u"][:]
v_wind = dataset.variables["v"][:]
pressure_levels = dataset.variables["pressure_level"][:]
time = dataset.variables["valid_time"][:]

# Print the shape to understand the data structure
print("Temperature shape:", temperature.shape)
print("Relative humidity shape:", relative_humidity.shape)
print("U-wind shape:", u_wind.shape)
print("V-wind shape:", v_wind.shape)
'''

EnvERA = Environment(
    date=(2025, 3, 22, 21),
    latitude=32.988517,
    longitude=-106.975062,
    elevation=1400,
)

EnvERA.set_atmospheric_model(
    type="Reanalysis",
    file="testing_weather/reanalysis_era5_pressure_levels_2025_03_22_21_00.nc",
    dictionary="ECMWF",
)
EnvERA.plots.atmospheric_model()



