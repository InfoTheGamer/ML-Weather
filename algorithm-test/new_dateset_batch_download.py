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
    x = calendar.monthrange(2024,dt)[1]
    daysInAMonth = []
    for i in range(1,x+1):
        daysInAMonth.append(f"{i:02d}")
    request = {
        "variable": ["surface_pressure"],
        "product_type": "reanalysis",
        "year": ["2024"],
        "month": f"{dt:02d}",
        "day": daysInAMonth,
        "time": [ '00:00','01:00','02:00',
            '03:00','04:00','05:00',
            '06:00','07:00','08:00',
            '09:00','10:00','11:00',
            '12:00','13:00','14:00',
            '15:00','16:00','17:00',
            '18:00','19:00','20:00',
            '21:00','22:00','23:00'],
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

def download_pair(i,selectTimes, area, save_dir, dataset):
    client = cdsapi.Client()
    # for i in range(len(hour)):
    #     stringHour.append(str(hour[i])+":00")
    #     hourPlusOne.append(str(hour[i]+1)+":00")
    """
    #t1 = t0 + timedelta(hours=1)
    t1 = []
    for time in t0:
        t1.append(time + timedelta(hours=1))
    """
    print(f"📥 Starting {calendar.month_name[i]} ")
    try:
        times, p0, lat, lon = download_and_extract_nc(client, i, f"tmp0_{calendar.month_name[i]}", save_dir, area, dataset)
        
        #_, p1, _, _ = download_and_extract_nc(client, [i,day,hourPlusOne], f"tmp1_{calendar.month_name[i+6]}_{day:02}", save_dir, area, dataset)
        for day,otherWordForTime in selectTimes[i-5].items():
            for hourLook in otherWordForTime:
                lookingFor = datetime(2024,i,day,hourLook).timestamp()
                lookingForPlusOne = datetime(2024,i,day,hourLook+1).timestamp()
                times.mask =False
                pos = np.nonzero(times  == int(lookingFor))[0][0]

                posPlusOne = np.nonzero(times  == int(lookingForPlusOne))[0][0]
                timeFromdate = datetime(2024,i,day,hourLook).strftime("%Y_%m_%d_%H")
                output_path = os.path.join(save_dir, f"pair_{timeFromdate}.nc")
                with nc.Dataset(output_path, "w", format="NETCDF4") as ds_out:
                    lat_dim, lon_dim = p0[pos].shape
                    ds_out.createDimension("lat", lat_dim)
                    ds_out.createDimension("lon", lon_dim)

                    lat_var = ds_out.createVariable("latitude", "f4", ("lat",))
                    lon_var = ds_out.createVariable("longitude", "f4", ("lon",))
                    v1 = ds_out.createVariable("pressure_T", "f4", ("lat", "lon"))
                    v2 = ds_out.createVariable("pressure_T_plus_1", "f4", ("lat", "lon"))
                    lat_var[:] = lat
                    lon_var[:] = lon
                    v1[:, :] = p0[pos]
                    v2[:, :] = p0[posPlusOne]

            print(f"✅ Saved: {output_path}")
        return True

    except Exception as e:
        print(f"❌ Failed to download pair {calendar.month_name[i]}: {e}")
        return False

    finally:
        for name in [f"tmp0_{calendar.month_name[i]}"]:
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
    for i in range(start_time.month, end_time.month+1):
        months={}
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
            executor.submit(download_pair, i+5,selected_times, area, save_dir, dataset): i
            for i in range(6)
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
