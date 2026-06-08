import numpy as np
from pathlib import Path

print("Loading features...")

features = np.load(
    "dataset/processed/features.npy"
)

print("Shape:", features.shape)

num_channels = features.shape[1]

means = []
stds = []
mins = []
maxs = []

channel_names = [
    "t2m",
    "msl",
    "u10",
    "v10",
    "sst",
    "elevation"
]

for i in range(num_channels):

    channel = features[:, i]

    mean = np.nanmean(channel)
    std = np.nanstd(channel)
    minimum = np.nanmin(channel)
    maximum = np.nanmax(channel)

    means.append(mean)
    stds.append(std)
    mins.append(minimum)
    maxs.append(maximum)

    print(f"\n{channel_names[i]}")
    print(f"Mean : {mean}")
    print(f"Std  : {std}")
    print(f"Min  : {minimum}")
    print(f"Max  : {maximum}")

means = np.array(means, dtype=np.float32)
stds = np.array(stds, dtype=np.float32)
mins = np.array(mins, dtype=np.float32)
maxs = np.array(maxs, dtype=np.float32)

stats_dir = Path(
    "dataset/metadata/statistics"
)

stats_dir.mkdir(
    parents=True,
    exist_ok=True
)

np.save(stats_dir / "mean.npy", means)
np.save(stats_dir / "std.npy", stds)
np.save(stats_dir / "min.npy", mins)
np.save(stats_dir / "max.npy", maxs)

print("\nStatistics saved successfully.")