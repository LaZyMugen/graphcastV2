import numpy as np
import torch

from torch.utils.data import Dataset
from torch_geometric.data import Data

from src.utils.config_loader import load_yaml


class GraphWeatherDataset(Dataset):

    def __init__(self):

        data_cfg = load_yaml(
            "config/data_config.yaml"
        )

        self.X = np.load(
            data_cfg["paths"]["x_train"],
            mmap_mode="r"
        )

        self.Y = np.load(
            data_cfg["paths"]["y_train"],
            mmap_mode="r"
        )

        self.edge_index = np.load(
            "graph/edge_index.npy"
        )

        print(
            f"Loaded {len(self.X)} samples"
        )

    def __len__(self):

        return len(self.X)
    

    def __getitem__(self, idx):

        x = self.X[idx]
        y = self.Y[idx]

        # ----------------------------------
        # INPUT
        # (T,C,H,W)
        # -> (H,W,T,C)
        # ----------------------------------

        x = np.transpose(
            x,
            (2, 3, 0, 1)
        )

        H, W, T, C = x.shape

        x = x.reshape(
            H * W,
            T * C
        )

        # ----------------------------------
        # TARGET
        # (C,H,W)
        # -> (H,W,C)
        # ----------------------------------

        y = np.transpose(
            y,
            (1, 2, 0)
        )

        y = y.reshape(
            H * W,
            C
        )

        return Data(
        x=torch.tensor(
            x,
            dtype=torch.float32
        ),
        edge_index=torch.tensor(
            self.edge_index,
            dtype=torch.long
        ),
        y=torch.tensor(
            y,
            dtype=torch.float32
        )
    )