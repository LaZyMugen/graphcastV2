import xarray as xr

print("=" * 60)
print("ATMOSPHERE DATASET")
print("=" * 60)

atm = xr.open_dataset("dataset/atmosphere/atmosphereGC.nc")
print(atm)

print("\n\n")

print("=" * 60)
print("SST DATASET")
print("=" * 60)

sst = xr.open_dataset("dataset/ocean/SST_GC.nc")
print(sst)