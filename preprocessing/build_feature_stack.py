import xarray as xr
import numpy as np
from pathlib import Path

from src.utils.config_loader import load_yaml

print("Loading configuration...")

config = load_yaml("config/data_config.yaml")

atm_file = config["atmosphere"]["file"]
ocean_file = config["ocean"]["file"]

feature_channels = config["feature_stack"]["channels"]

print("Loading datasets...")

atm = xr.open_dataset(atm_file)
ocean_ds = xr.open_dataset(ocean_file)

channel_data = []

print("Building feature stack...")

for channel in feature_channels:

    print(f"Processing: {channel}")

    # Atmospheric variables
    if channel in atm.data_vars:

        data = atm[channel].values
        channel_data.append(data)

    # Ocean variables
    elif channel in ocean_ds.data_vars:

        data = ocean_ds[channel].values
        channel_data.append(data)

    # Derived elevation
    elif channel == "elevation":

        z = atm["z"].isel(valid_time=0).values

        elevation = z / 9.80665

        num_timesteps = atm.sizes["valid_time"]

        elevation = np.repeat(
            elevation[np.newaxis, :, :],
            num_timesteps,
            axis=0
        )

        channel_data.append(elevation)

    else:

        raise ValueError(
            f"Unknown channel: {channel}"
        )

print("Stacking channels...")

features = np.stack(
    channel_data,
    axis=1
)

print("Final shape:", features.shape)

output_path = config["paths"]["features"]

Path(output_path).parent.mkdir(
    parents=True,
    exist_ok=True
)

np.save(
    output_path,
    features.astype(np.float32)
)

print(f"Saved: {output_path}")