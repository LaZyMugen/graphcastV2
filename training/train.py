import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from torch_geometric.loader import DataLoader

from src.datasets.graph_dataset import (
    GraphWeatherDataset
)

from src.models.mini_graphcast import (
    MiniGraphCast
)

from src.utils.config_loader import (
    load_yaml
)


def main():

    model_cfg = load_yaml(
        "config/model_config.yaml"
    )

    epochs = model_cfg["training"]["epochs"]

    batch_size = model_cfg["training"]["batch_size"]

    learning_rate = model_cfg["training"]["learning_rate"]

    weight_decay = model_cfg["training"]["weight_decay"]

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("Device:", device)

    dataset = GraphWeatherDataset()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    model = MiniGraphCast()

    model = model.to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    print()
    print("Starting training...")

    losses = []

    Path("results").mkdir(
        parents=True,
        exist_ok=True
    )

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0

        for batch in loader:

            batch = batch.to(device)

            optimizer.zero_grad()

            prediction = model(
                batch.x,
                batch.edge_index
            )

            loss = criterion(
                prediction,
                batch.y
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        epoch_loss = (
            running_loss
            / len(loader)
        )

        losses.append(epoch_loss)

        print(
            f"Epoch {epoch+1}/{epochs}"
            f"  Loss: {epoch_loss:.6f}"
        )

    print()
    print("Training complete.")


    np.save(
    "results/training_losses.npy",
    np.array(losses)
)

    print("Loss history saved.")

    torch.save(
        model.state_dict(),
        "mini_graphcast.pth"
    )

    print(
        "Model saved."
    )


if __name__ == "__main__":

    main()