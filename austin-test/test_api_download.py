import cdsapi
import os

dataset = "reanalysis-era5-land"
request = {
    "variable": ["surface_pressure"],
    "year": "2025",
    "month": "03",
    "day": ["21"],
    "time": ["00:00"],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "area": [50, -125, 24, -67]
}

# Create the directory if it doesn't exist
folder_path = "algorithm-test"

# File path with the same name as the dataset (ERA5)
file_name = f"{dataset.replace('-', '_')}_test.nc"  # Create a file name based on the request
file_path = os.path.join(folder_path, file_name)

# Initialize the client
client = cdsapi.Client()

# Retrieve and download the data
client.retrieve(dataset, request).download(file_path)

print(f"File downloaded to: {file_path}")