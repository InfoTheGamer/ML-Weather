import cdsapi
import os
import zipfile
import netCDF4 as nc
import numpy as np
from datetime import datetime, timedelta

def dataset_batch_download(start_time, num_pairs, area, save_dir, dataset="reanalysis-era5-land"):
    client = cdsapi.Client()
    os.makedirs(save_dir, exist_ok=True)

    def download_and_extract_nc(dt, base_name):
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

    for i in range(num_pairs):
        t0 = start_time + timedelta(hours=i)
        t1 = t0 + timedelta(hours=1)

        print(f"📥 Downloading pair {i}: {t0} -> {t1}")

        try:
            p0, lat, lon = download_and_extract_nc(t0, f"tmp0_{i}")
            p1, _, _ = download_and_extract_nc(t1, f"tmp1_{i}")

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

        except Exception as e:
            print(f"❌ Failed on pair {i}: {e}")

        finally:
            for name in [f"tmp0_{i}", f"tmp1_{i}"]:
                for ext in [".zip", ".nc"]:
                    path = os.path.join(save_dir, name + ext)
                    if os.path.exists(path): os.remove(path)

# === Main execution ===
if __name__ == "__main__":
    dataset_batch_download(
        start_time=datetime(2025, 3, 21, 0),
        num_pairs=10,
        area=[50, -125, 24, -67],
        save_dir="algorithm-test"
    )
