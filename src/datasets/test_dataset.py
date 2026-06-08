from src.datasets.graph_dataset import (
    GraphWeatherDataset
)

dataset = GraphWeatherDataset()

sample = dataset[0]

print()

print(
    "Node Features:",
    sample["x"].shape
)

print(
    "Target:",
    sample["y"].shape
)

print(
    "Edges:",
    sample["edge_index"].shape
)