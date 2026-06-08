import xarray as xr
import numpy as np

ds = xr.open_dataset("dataset/atmosphere/atmosphereGC.nc")

z = ds["z"]

print(z.isel(valid_time=0).mean().values)
print(z.isel(valid_time=100).mean().values)

print(
    np.abs(
        z.isel(valid_time=0) -
        z.isel(valid_time=100)
    ).max().values
)