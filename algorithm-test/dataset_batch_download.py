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
        "year": ["2024"],
        "month": f"{(dt[0]+5):02d}",
        "day": [f"{dt[1]:02}"],
        "time": dt[2],
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
                times = ds.variables["valid_time"][:]
                pressure = ds.variables["sp"][:]
                lat = ds.variables["latitude"][:]
                lon = ds.variables["longitude"][:]
                return times, pressure, lat, lon
        except Exception as e:
            print(f"⚠️ Attempt {attempt} failed for {base_name}: {e}")
            time.sleep(5 * attempt)
        finally:
            if os.path.exists(tmp_zip_path):
                os.remove(tmp_zip_path)
    raise RuntimeError(f"All attempts failed for {base_name}.")

def download_pair(i, day,hour, area, save_dir, dataset):
    client = cdsapi.Client()
    hourPlusOne = []
    stringHour =[]
    for i in range(len(hour)):
        stringHour.append(str(hour[i])+":00")
        hourPlusOne.append(str(hour[i]+1)+":00")
    """
    #t1 = t0 + timedelta(hours=1)
    t1 = []
    for time in t0:
        t1.append(time + timedelta(hours=1))
    """
    print(f"📥 Starting {calendar.month_name[i+6]} {day:02} ")
    try:
        times, p0, lat, lon = download_and_extract_nc(client, [i,day,stringHour], f"tmp0_{calendar.month_name[i+6]}_{day:02}", save_dir, area, dataset)
        _, p1, _, _ = download_and_extract_nc(client, [i,day,hourPlusOne], f"tmp1_{calendar.month_name[i+6]}_{day:02}", save_dir, area, dataset)

        for i,otherWordForTime in enumerate(times):
            dateFromEpoch =datetime.fromtimestamp(otherWordForTime)
            timeFromdate = dateFromEpoch.strftime("%Y_%m_%d_%H")
            output_path = os.path.join(save_dir, f"pair_{timeFromdate}.nc")
            with nc.Dataset(output_path, "w", format="NETCDF4") as ds_out:
                lat_dim, lon_dim = p0[i].shape
                ds_out.createDimension("lat", lat_dim)
                ds_out.createDimension("lon", lon_dim)

                lat_var = ds_out.createVariable("latitude", "f4", ("lat",))
                lon_var = ds_out.createVariable("longitude", "f4", ("lon",))
                v1 = ds_out.createVariable("pressure_T", "f4", ("lat", "lon"))
                v2 = ds_out.createVariable("pressure_T_plus_1", "f4", ("lat", "lon"))

                lat_var[:] = lat
                lon_var[:] = lon
                v1[:, :] = p0[i]
                v2[:, :] = p1[i]

            print(f"✅ Saved: {output_path}")
        return True

    except Exception as e:
        print(f"❌ Failed to download pair {calendar.month_name[i+6]}_{day:02}: {e}")
        return False

    finally:
        for name in [f"tmp0_{calendar.month_name[i+6]}_{day:02}", f"tmp1_{calendar.month_name[i+6]}_{day:02}"]:
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
        months = {}
        x = calendar.monthrange(start_time.year,i)
        days = []
        hours = []
        for d in range(int(select)):
            s = random.randrange(1,x[1])
            days.append(s)
        days.sort()
        prevDay = days[0]
        temp = 0
        numofDays = []
        
        for day in days:
            if prevDay == day:
                temp += 1
            else:
                prevDay = day
                numofDays.append(temp)
                temp = 1
            if day == days[len(days)-1]:
                numofDays.append(temp)
        for n in numofDays:
            h = random.sample(range(0,23),n)
            hours.append(h)
        days = set(days)
        for day in days:
            months[day] = hours[0]
            if(len(hours) != 0):
                hours.pop(0)
        all_possible_starts.append(months)
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
            executor.submit(download_pair, i, day,hour, area, save_dir, dataset): i
            for i, t0 in enumerate(selected_times) for day,hour in t0.items()
        }

        for future in as_completed(futures):
            i = futures[future]
            try:
                if future.result():
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                print(f"💥 Exception during download of pair {i}_{day}: {e}")
                fail_count += 1

    print(f"🏁 Batch finished: {success_count} successful, {fail_count} failed.")

# === Main execution ===
if __name__ == "__main__":
    dataset_batch_download(
        start_time=datetime(2024, 6, 1, 0),
        end_time=datetime(2024, 11, 30, 23),
        num_pairs=12,
        area=[50, -125, 24, -67],
        save_dir="32gb-dataset",
        max_threads=8
    )
