import numpy as np
import matplotlib.pyplot as plt

truth = np.load(
    "results/t2m_truth_series.npy"
)

pred = np.load(
    "results/t2m_pred_series.npy"
)

plt.figure(
    figsize=(12,6)
)

plt.plot(
    truth,
    label="Ground Truth"
)

plt.plot(
    pred,
    label="Prediction"
)

plt.xlabel("Forecast Sample")
plt.ylabel("Normalized T2M")

plt.title(
    "Maitri Temperature Forecast"
)

plt.legend()

plt.grid(True)

plt.savefig(
    "results/t2m_timeseries.png",
    dpi=300
)

plt.show()