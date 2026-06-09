import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

print("Loading results...")

pred = np.load("results/prediction.npy")
truth = np.load("results/ground_truth.npy")

# Channel order
# 0 t2m
# 1 msl
# 2 u10
# 3 v10
# 4 sst
# 5 elevation

CHANNEL = 0  # t2m

pred_field = pred[CHANNEL]
truth_field = truth[CHANNEL]

error_field = pred_field - truth_field

Path("results").mkdir(
    exist_ok=True
)

plt.figure(figsize=(18,6))

plt.subplot(1,3,1)

plt.imshow(
    truth_field,
    origin="lower",
    aspect="auto"
)

plt.title("Ground Truth T2M")
plt.colorbar()

plt.subplot(1,3,2)

plt.imshow(
    pred_field,
    origin="lower",
    aspect="auto"
)

plt.title("Predicted T2M")
plt.colorbar()

plt.subplot(1,3,3)

plt.imshow(
    error_field,
    origin="lower",
    aspect="auto"
)

plt.title("Prediction Error")
plt.colorbar()

plt.tight_layout()

plt.savefig(
    "results/t2m_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "Saved results/t2m_comparison.png"
)