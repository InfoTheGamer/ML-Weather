import cdsapi

dataset = "reanalysis-era5-pressure-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": ["geopotential"],
    "year": ["2025"],
    "month": ["02"],
    "day": ["28"],
    "time": [
        "07:00", "08:00", "09:00",
        "10:00", "11:00", "12:00"
    ],
    "pressure_level": ["800"],
    "data_format": "netcdf",
    "download_format": "unarchived"
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
