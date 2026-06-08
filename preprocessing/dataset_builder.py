import numpy as np
from pathlib import Path

from src.utils.config_loader import load_yaml

print("Loading configuration...")

data_cfg = load_yaml("config/data_config.yaml")
model_cfg = load_yaml("config/model_config.yaml")

input_steps = model_cfg["dataset"]["input_steps"]
forecast_steps = model_cfg["dataset"]["forecast_steps"]

features_path = data_cfg["paths"]["normalized_features"]

print("Loading normalized features...")

features = np.load(features_path)

num_timesteps = features.shape[0]

X = []
Y = []

print("Building samples...")

for t in range(
    input_steps,
    num_timesteps - forecast_steps + 1
):

    x = features[
        t - input_steps:t
    ]

    y = features[
        t + forecast_steps - 1
    ]

    X.append(x)
    Y.append(y)

X = np.asarray(
    X,
    dtype=np.float32
)

Y = np.asarray(
    Y,
    dtype=np.float32
)

print("X shape:", X.shape)
print("Y shape:", Y.shape)

x_path = data_cfg["paths"]["x_train"]
y_path = data_cfg["paths"]["y_train"]

Path(x_path).parent.mkdir(
    parents=True,
    exist_ok=True
)

np.save(
    x_path,
    X
)

np.save(
    y_path,
    Y
)

print("Saved training dataset.")