import xarray as xr
import numpy as np

ds = xr.open_dataset("dataset/ocean/SST_GC.nc")

sst = ds["sst"].values

print("Shape:", sst.shape)

print("NaNs:", np.isnan(sst).sum())

print("Total values:", sst.size)

print("NaN Percentage:",
      100*np.isnan(sst).sum()/sst.size)

print("Min:", np.nanmin(sst))
print("Max:", np.nanmax(sst))
print("Mean:", np.nanmean(sst))