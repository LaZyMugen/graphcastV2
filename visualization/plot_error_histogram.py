import numpy as np
import matplotlib.pyplot as plt

print("Loading results...")

pred = np.load(
    "results/prediction.npy"
)

truth = np.load(
    "results/ground_truth.npy"
)

error = pred - truth

print("Mean Error :", np.mean(error))
print("Std Error  :", np.std(error))
print("MAE        :", np.mean(np.abs(error)))
print("RMSE       :", np.sqrt(np.mean(error**2)))

plt.figure(figsize=(8,5))

plt.hist(
    error.flatten(),
    bins=50
)

plt.xlabel("Prediction Error")
plt.ylabel("Frequency")

plt.title(
    "Forecast Error Distribution"
)

plt.grid(True)

plt.axvline(
    0,
    linestyle="--"
)

plt.savefig(
    "results/error_histogram.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "Saved results/error_histogram.png"
)