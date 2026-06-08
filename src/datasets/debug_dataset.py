from src.datasets.graph_dataset import GraphWeatherDataset
import torch

dataset = GraphWeatherDataset()

sample = dataset[0]

print("x NaNs:", torch.isnan(sample.x).sum())
print("y NaNs:", torch.isnan(sample.y).sum())

print("x shape:", sample.x.shape)
print("y shape:", sample.y.shape)