import numpy as np

x = np.load(
    "dataset/processed/normalized_features.npy"
)

for i in range(x.shape[1]):

    channel = x[:, i]

    print(
        f"Channel {i}",
        np.nanmean(channel),
        np.nanstd(channel)
    )