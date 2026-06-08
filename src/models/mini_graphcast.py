import torch
import torch.nn as nn

from torch_geometric.nn import GraphConv


class MiniGraphCast(nn.Module):

    def __init__(
        self,
        input_dim=24,
        hidden_dim=64,
        output_dim=6
    ):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim
            ),
            nn.ReLU()
        )

        self.conv1 = GraphConv(
            hidden_dim,
            hidden_dim
        )

        self.conv2 = GraphConv(
            hidden_dim,
            hidden_dim
        )

        self.conv3 = GraphConv(
            hidden_dim,
            hidden_dim
        )

        self.decoder = nn.Linear(
            hidden_dim,
            output_dim
        )

    def forward(
        self,
        x,
        edge_index
    ):

        x = self.encoder(x)

        x = self.conv1(
            x,
            edge_index
        )
        x = torch.relu(x)

        x = self.conv2(
            x,
            edge_index
        )
        x = torch.relu(x)

        x = self.conv3(
            x,
            edge_index
        )
        x = torch.relu(x)

        x = self.decoder(x)

        return x