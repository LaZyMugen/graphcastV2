import numpy as np
import torch

from src.datasets.graph_dataset import GraphWeatherDataset
from src.models.mini_graphcast import MiniGraphCast

LAT_IDX = 43
LON_IDX = 47

dataset = GraphWeatherDataset()

model = MiniGraphCast()

model.load_state_dict(
    torch.load(
        "mini_graphcast.pth",
        map_location="cpu"
    )
)

model.eval()

truth_series = []
pred_series = []

node_id = (
    LAT_IDX * 121
    + LON_IDX
)

with torch.no_grad():

    for i in range(100):

        sample = dataset[i]

        pred = model(
            sample.x,
            sample.edge_index
        )

        pred = pred.numpy()

        truth = sample.y.numpy()

        pred_series.append(
            pred[node_id, 0]
        )

        truth_series.append(
            truth[node_id, 0]
        )

np.save(
    "results/t2m_pred_series.npy",
    np.array(pred_series)
)

np.save(
    "results/t2m_truth_series.npy",
    np.array(truth_series)
)

print("Saved timeseries.")