import numpy as np
import xarray as xr
from pathlib import Path

from src.utils.config_loader import load_yaml

print("Loading configuration...")

data_cfg = load_yaml("config/data_config.yaml")
model_cfg = load_yaml("config/model_config.yaml")

atm_file = data_cfg["atmosphere"]["file"]

connectivity = model_cfg["graph"]["connectivity"]

print("Loading coordinates...")

ds = xr.open_dataset(atm_file)

latitudes = ds["latitude"].values
longitudes = ds["longitude"].values

num_lat = len(latitudes)
num_lon = len(longitudes)

print(f"Latitudes : {num_lat}")
print(f"Longitudes: {num_lon}")

# --------------------------------------------------
# Build Nodes
# --------------------------------------------------

nodes = []

node_id = 0

for i in range(num_lat):

    for j in range(num_lon):

        nodes.append([
            node_id,
            latitudes[i],
            longitudes[j]
        ])

        node_id += 1

nodes = np.asarray(
    nodes,
    dtype=np.float32
)

print("Nodes shape:", nodes.shape)

# --------------------------------------------------
# Build Edges
# --------------------------------------------------

print("Building edges...")

edge_list = []

def grid_to_node(i, j):
    return i * num_lon + j

for i in range(num_lat):

    for j in range(num_lon):

        current = grid_to_node(i, j)

        # North
        if i > 0:
            edge_list.append(
                [current, grid_to_node(i - 1, j)]
            )

        # South
        if i < num_lat - 1:
            edge_list.append(
                [current, grid_to_node(i + 1, j)]
            )

        # West
        if j > 0:
            edge_list.append(
                [current, grid_to_node(i, j - 1)]
            )

        # East
        if j < num_lon - 1:
            edge_list.append(
                [current, grid_to_node(i, j + 1)]
            )

        if connectivity == 8:

            # NW
            if i > 0 and j > 0:
                edge_list.append(
                    [current, grid_to_node(i - 1, j - 1)]
                )

            # NE
            if i > 0 and j < num_lon - 1:
                edge_list.append(
                    [current, grid_to_node(i - 1, j + 1)]
                )

            # SW
            if i < num_lat - 1 and j > 0:
                edge_list.append(
                    [current, grid_to_node(i + 1, j - 1)]
                )

            # SE
            if i < num_lat - 1 and j < num_lon - 1:
                edge_list.append(
                    [current, grid_to_node(i + 1, j + 1)]
                )

edge_index = np.asarray(
    edge_list,
    dtype=np.int64
).T

print("Edge shape:", edge_index.shape)

# --------------------------------------------------
# Save
# --------------------------------------------------

graph_dir = Path("graph")

graph_dir.mkdir(
    parents=True,
    exist_ok=True
)

np.save(
    graph_dir / "nodes.npy",
    nodes
)

np.save(
    graph_dir / "edge_index.npy",
    edge_index
)

print("Graph saved.")