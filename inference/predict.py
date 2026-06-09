import numpy as np
import torch
from pathlib import Path

from src.datasets.graph_dataset import GraphWeatherDataset
from src.models.mini_graphcast import MiniGraphCast

device = torch.device("cpu")

dataset = GraphWeatherDataset()

sample = dataset[0]

model = MiniGraphCast()

model.load_state_dict(
    torch.load(
        "mini_graphcast.pth",
        map_location=device
    )
)

model.eval()

with torch.no_grad():

    prediction = model(
        sample.x,
        sample.edge_index
    )

prediction = prediction.numpy()
truth = sample.y.numpy()

# 9801 -> 81 x 121

prediction = prediction.reshape(
    81,
    121,
    6
)

truth = truth.reshape(
    81,
    121,
    6
)

prediction = np.transpose(
    prediction,
    (2, 0, 1)
)

truth = np.transpose(
    truth,
    (2, 0, 1)
)

Path("results").mkdir(
    exist_ok=True
)

np.save(
    "results/prediction.npy",
    prediction
)

np.save(
    "results/ground_truth.npy",
    truth
)

print("Prediction shape:", prediction.shape)
print("Truth shape:", truth.shape)

print("Saved prediction outputs.")