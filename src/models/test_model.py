from src.datasets.graph_dataset import (
    GraphWeatherDataset
)

from src.models.mini_graphcast import (
    MiniGraphCast
)

dataset = GraphWeatherDataset()

sample = dataset[0]

model = MiniGraphCast()

output = model(
    sample.x,
    sample.edge_index
)

print()

print(
    "Input:",
    sample.x.shape
)

print(
    "Output:",
    output.shape
)

print(
    "Target:",
    sample.y.shape
)