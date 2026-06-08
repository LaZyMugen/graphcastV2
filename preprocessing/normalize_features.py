import numpy as np
from pathlib import Path

print("Loading features...")

features = np.load(
    "dataset/processed/features.npy"
)

means = np.load(
    "dataset/metadata/statistics/mean.npy"
)

stds = np.load(
    "dataset/metadata/statistics/std.npy"
)

print("Features shape:", features.shape)

normalized = np.empty_like(
    features,
    dtype=np.float32
)

num_channels = features.shape[1]

for channel_idx in range(num_channels):

    channel = features[:, channel_idx]

    normalized_channel = (
        channel - means[channel_idx]
    ) / (stds[channel_idx] + 1e-8)

    normalized_channel = np.nan_to_num(
        normalized_channel,
        nan=0.0
    )

    normalized[:, channel_idx] = normalized_channel
    
print("Saving normalized features...")

Path(
    "dataset/processed"
).mkdir(
    parents=True,
    exist_ok=True
)

np.save(
    "dataset/processed/normalized_features.npy",
    normalized
)

print("Done.")
print("Shape:", normalized.shape)