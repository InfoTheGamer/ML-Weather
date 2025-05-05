import cdsapi
import os
import netCDF4 as nc
import zipfile
import numpy as np
from datetime import datetime, timedelta
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_full_day(client, year, month, day, area, save_dir, dataset="reanalysis-era5-land", retries=3):
    tmp_zip_path = os.path.join(save_dir, f"full_{year}_{month:02}_{day:02}.zip").replace("\\", "/")
    tmp_nc_path = os.path.join(save_dir, f"full_{year}_{month:02}_{day:02}.nc").replace("\\", "/")
    request = {
        "variable": ["surface_pressure"],
        "product_type": "reanalysis",
        "year": f"{year}",
        "month": f"{month:02}",
        "day": f"{day:02}",
        "time": [f"{h:02}:00" for h in range(24)],
        "format": "netcdf",
        "area": area,
    }

    for attempt in range(1, retries + 1):
        try:
            client.retrieve(dataset, request).download(tmp_zip_path)
            with zipfile.ZipFile(tmp_zip_path, "r") as zip_ref:
                zip_ref.extractall(save_dir)

            extracted_path = os.path.join(save_dir, "data_0.nc")
            if not os.path.exists(extracted_path):
                raise FileNotFoundError("data_0.nc not found in ZIP archive.")

            os.rename(extracted_path, tmp_nc_path)
            os.remove(tmp_zip_path)
            print(f"✅ Saved: {tmp_nc_path}")
            return tmp_nc_path
        except Exception as e:
            print(f"⚠️ Attempt {attempt} failed for {year}-{month:02}-{day:02}: {e}")
            time.sleep(5 * attempt)
    raise RuntimeError(f"All attempts failed for {year}-{month:02}-{day:02}")

def process_day_into_hour_pairs(nc_path, save_dir):
    """Load a full-day .nc file and split into hourly pairs."""
    with nc.Dataset(nc_path) as ds:
        times = ds.variables["valid_time"][:]
        pressure = ds.variables["sp"][:]  # shape (24, lat, lon)
        lat = ds.variables["latitude"][:]
        lon = ds.variables["longitude"][:]

        for i in range(len(times) - 1):
            t0 = nc.num2date(times[i], ds.variables["valid_time"].units)
            time_str = t0.strftime("%Y_%m_%d_%H")
            output_path = os.path.join(save_dir, f"pair_{time_str}.nc")
            
            with nc.Dataset(output_path, "w", format="NETCDF4") as ds_out:
                ds_out.createDimension("lat", len(lat))
                ds_out.createDimension("lon", len(lon))

                ds_out.createVariable("latitude", "f4", ("lat",))[:] = lat
                ds_out.createVariable("longitude", "f4", ("lon",))[:] = lon
                ds_out.createVariable("pressure_T", "f4", ("lat", "lon"))[:, :] = pressure[i]
                ds_out.createVariable("pressure_T_plus_1", "f4", ("lat", "lon"))[:, :] = pressure[i + 1]

            print(f"🧩 Created: {output_path}")

    # Delete the full file following the split:
    try:
        os.remove(nc_path)
        print(f"🗑️ Deleted: {nc_path}")
    except Exception as e:
        print(f"⚠️ Could not delete {nc_path}: {e}")

def batch_download_and_process(start_date, end_date, area, save_dir, dataset="reanalysis-era5-land", max_threads=4):
    os.makedirs(save_dir, exist_ok=True)
    client = cdsapi.Client()

    date_list = []
    current = start_date
    while current <= end_date:
        date_list.append(current)
        current += timedelta(days=1)

    def task(date):
        try:
            nc_path = download_full_day(client, date.year, date.month, date.day, area, save_dir, dataset)
            process_day_into_hour_pairs(nc_path, save_dir)
        except Exception as e:
            print(f"❌ Failed for {date}: {e}")

    print(f"🚀 Starting download of {len(date_list)} days with {max_threads} threads.")
    with ThreadPoolExecutor(max_threads) as executor:
        executor.map(task, date_list)

# Example usage:
if __name__ == "__main__":
    start = datetime(2024, 6, 1)
    end = datetime(2024, 11, 18)
    area = [50, -125, 24, -67] 
    save_directory = "hurricanes-24"
    batch_download_and_process(start, end, area, save_directory)
