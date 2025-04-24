import cdsapi
import os
import zipfile
import netCDF4 as nc
import numpy as np
import calendar
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import time

def download_and_extract_nc(client, dt, base_name, save_dir, area, dataset="reanalysis-era5-land", retries=3):
    tmp_zip_path = os.path.join(save_dir, f"{base_name}.zip").replace("\\", "/")
    tmp_nc_path = os.path.join(save_dir, f"{base_name}.nc").replace("\\", "/")

    request = {
        "variable": ["surface_pressure"],
        "product_type": "reanalysis",
        "year": f"{dt.year}",
        "month": f"{dt.month:02}",
        "day": f"{dt.day:02}",
        "time": [dt.strftime("%H:%M")],
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

            with nc.Dataset(tmp_nc_path) as ds:
                pressure = ds.variables["sp"][0]
                lat = ds.variables["latitude"][:]
                lon = ds.variables["longitude"][:]
                return pressure, lat, lon
        except Exception as e:
            print(f"⚠️ Attempt {attempt} failed for {base_name}: {e}")
            time.sleep(5 * attempt)
        finally:
            if os.path.exists(tmp_zip_path):
                os.remove(tmp_zip_path)
    raise RuntimeError(f"All attempts failed for {base_name}.")

def download_pair(i, t0, area, save_dir, dataset):
    client = cdsapi.Client()
    t1 = t0 + timedelta(hours=1)

    print(f"📥 Starting pair {i}: {t0} -> {t1}")

    try:
        p0, lat, lon = download_and_extract_nc(client, t0, f"tmp0_{i}", save_dir, area, dataset)
        p1, _, _ = download_and_extract_nc(client, t1, f"tmp1_{i}", save_dir, area, dataset)

        output_path = os.path.join(save_dir, f"pair_{i:03}.nc")

        with nc.Dataset(output_path, "w", format="NETCDF4") as ds_out:
            lat_dim, lon_dim = p0.shape
            ds_out.createDimension("lat", lat_dim)
            ds_out.createDimension("lon", lon_dim)

            lat_var = ds_out.createVariable("latitude", "f4", ("lat",))
            lon_var = ds_out.createVariable("longitude", "f4", ("lon",))
            v1 = ds_out.createVariable("pressure_T", "f4", ("lat", "lon"))
            v2 = ds_out.createVariable("pressure_T_plus_1", "f4", ("lat", "lon"))

            lat_var[:] = lat
            lon_var[:] = lon
            v1[:, :] = p0
            v2[:, :] = p1

        print(f"✅ Saved: {output_path}")
        return True

    except Exception as e:
        print(f"❌ Failed to download pair {i}: {e}")
        return False

    finally:
        for name in [f"tmp0_{i}", f"tmp1_{i}"]:
            for ext in [".zip", ".nc"]:
                path = os.path.join(save_dir, name + ext)
                if os.path.exists(path):
                    os.remove(path)

def dataset_batch_download(start_time, end_time, num_pairs, area, save_dir, dataset="reanalysis-era5-land", max_threads=5):
    os.makedirs(save_dir, exist_ok=True)

    all_possible_starts = []
    current = start_time
    select = 512/(end_time.month - start_time.month)
    randomTimes = []
    for i in range(start_time.month, end_time.month):
        x = calendar.monthrange(start_time.year,i)
        days = []
        hours = []
        
        for d in range(int(select)):
            s = random.randrange(x[1])
            days.append(s)
        days.sort()
        prevDay = 0
        temp = 0
        numofDays = []
        for d in range(len(days)):
            if prevDay == days[d]:
                temp += 1
            else:
                if d + 1 < len(days):
                    prevDay = days[d+1]
                numofDays.append(temp)
                temp = 1
        if temp != 0:
            numofDays.append(temp)
            temp = 0
        for n in numofDays:
            h = random.sample(range(0,23),n)
            hours = hours + h
        for notI in range(len(hours)):
            all_possible_starts.append(datetime(2024,i,days[notI]+1,hours[notI]))
 
        
            
        
    #while current + timedelta(hours=1) <= end_time:
    #    all_possible_starts.append(current)
    #    current += timedelta(hours=1)

    #if num_pairs > len(all_possible_starts):
    #    raise ValueError("Requested more pairs than available time intervals.")

    selected_times = all_possible_starts

    print(f"🚀 Starting batch download: {num_pairs} random pairs using {max_threads} threads.")
    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {
            executor.submit(download_pair, i, t0, area, save_dir, dataset): i
            for i, t0 in enumerate(selected_times)
        }

        for future in as_completed(futures):
            i = futures[future]
            try:
                if future.result():
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                print(f"💥 Exception during download of pair {i}: {e}")
                fail_count += 1

    print(f"🏁 Batch finished: {success_count} successful, {fail_count} failed.")

# === Main execution ===
if __name__ == "__main__":
    dataset_batch_download(
        start_time=datetime(2024, 6, 1, 0),
        end_time=datetime(2024, 11, 30, 23),
        num_pairs=512,
        area=[50, -125, 24, -67],
        save_dir="512-data",
        max_threads=8
    )
