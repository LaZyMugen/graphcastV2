import numpy as np
import matplotlib.pyplot as plt

losses = np.load(
    "results/training_losses.npy"
)

plt.figure(figsize=(8,5))

plt.plot(
    range(1, len(losses) + 1),
    losses,
    marker="o"
)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Mini GraphCast Training Loss")

plt.grid(True)

plt.savefig(
    "results/training_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "Saved results/training_curve.png"
)