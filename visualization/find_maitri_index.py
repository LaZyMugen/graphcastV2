import xarray as xr
import numpy as np

ds = xr.open_dataset(
    "dataset/atmosphere/atmosphereGC.nc"
)

lats = ds.latitude.values
lons = ds.longitude.values

target_lat = -70.77
target_lon = 11.73

lat_idx = np.abs(
    lats - target_lat
).argmin()

lon_idx = np.abs(
    lons - target_lon
).argmin()

print("Latitude Index :", lat_idx)
print("Longitude Index:", lon_idx)

print(
    "Actual Lat:",
    lats[lat_idx]
)

print(
    "Actual Lon:",
    lons[lon_idx]
)