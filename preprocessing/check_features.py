import numpy as np

x = np.load(
    "dataset/processed/features.npy"
)

print("Shape:", x.shape)

print("Channels:")
print("0 -> t2m")
print("1 -> msl")
print("2 -> u10")
print("3 -> v10")
print("4 -> sst")
print("5 -> elevation")