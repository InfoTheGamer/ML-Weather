#
# Testing environment models in RocketPy
#

#from rocketpy import Environment, SolidMotor, Rocket, Flight, StochasticSolidMotor, AeroSurface, FreeFormFins, TrapezoidalFins, StochasticEnvironment, Motor
#from datetime import datetime, timedelta
import cdsapi
import os

'''
now = datetime.now()
now_plus_twelve = now #+ timedelta(hours=12)

env_rap = Environment(
    date=now_plus_twelve,
    latitude=32.988528,
    longitude=-106.975056,
)
env_rap.set_atmospheric_model(type="forecast", file="RAP")
env_rap.plots.atmospheric_model()
'''


dataset = "reanalysis-era5-pressure-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": [
        "geopotential",
        "relative_humidity",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind"
    ],
    "year": ["2025"],
    "month": ["03"],
    "day": ["22"],
    "time": ["12:00"],
    "pressure_level": [
        "200", "250", "300",
        "350", "400", "450",
        "500", "550", "600",
        "650", "700", "750",
        "775", "800", "825",
        "850", "875", "900",
        "925", "950", "975",
        "1000"
    ],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "area": [31, -106, 33, -102]
}

# Create the directory if it doesn't exist
folder_path = "skew-t"

# File path with the same name as the dataset (ERA5)
file_name = f"{dataset.replace('-', '_')}_midlandtest.nc"  # Create a file name based on the request
file_path = os.path.join(folder_path, file_name)

# Initialize the client
client = cdsapi.Client()

# Retrieve and download the data
client.retrieve(dataset, request).download(file_path)

print(f"File downloaded to: {file_path}")
