import xarray as xr
import numpy as np
from pathlib import Path

print("Loading SST dataset...")

sst_ds = xr.open_dataset(
    "dataset/ocean/SST_GC.nc"
)

sst = sst_ds["sst"].values

print("Generating SST mask...")

# Use first timestep because missing pattern is usually static
sst_mask = (~np.isnan(sst[0])).astype(np.float32)

print("Mask shape:", sst_mask.shape)

print("Ocean pixels:", np.sum(sst_mask))
print("Land/Ice pixels:", np.sum(sst_mask == 0))

Path(
    "dataset/processed"
).mkdir(
    parents=True,
    exist_ok=True
)

np.save(
    "dataset/processed/sst_mask.npy",
    sst_mask
)

print("Saved SST mask.")