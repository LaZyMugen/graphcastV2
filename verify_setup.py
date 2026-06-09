import importlib
from pathlib import Path

print("=" * 60)
print("GRAPHCAST V2 SETUP VERIFICATION")
print("=" * 60)

# --------------------------------------------------
# PACKAGE CHECK
# --------------------------------------------------

packages = [
    "numpy",
    "torch",
    "torch_geometric",
    "xarray",
    "yaml",
    "matplotlib",
    "cfgrib",
    "pandas",
    "scipy"
]

print("\n[1] Checking Python Packages\n")

all_packages_ok = True

for pkg in packages:

    try:
        importlib.import_module(pkg)
        print(f"[OK] {pkg}")

    except Exception:
        print(f"[FAIL] {pkg}")
        all_packages_ok = False


# --------------------------------------------------
# PYTORCH CHECK
# --------------------------------------------------

print("\n[2] Checking PyTorch\n")

try:

    import torch

    print(f"Torch Version : {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

except Exception as e:

    print("[FAIL] Torch check failed")
    print(e)


# --------------------------------------------------
# CONFIG FILE CHECK
# --------------------------------------------------

required_configs = [

    "config/data_config.yaml",
    "config/model_config.yaml",
    "config/region_config.yaml"

]

print("\n[3] Checking Config Files\n")

for file in required_configs:

    if Path(file).exists():

        print(f"[OK] {file}")

    else:

        print(f"[MISSING] {file}")


# --------------------------------------------------
# DATASET CHECK
# --------------------------------------------------

required_data = [

    "dataset/atmosphere/atmosphereGC.nc",
    "dataset/ocean/SST_GC.nc",

    "dataset/processed/features.npy",
    "dataset/processed/normalized_features.npy",

    "dataset/processed/X.npy",
    "dataset/processed/Y.npy",

]

print("\n[4] Checking Dataset Files\n")

for file in required_data:

    if Path(file).exists():

        print(f"[OK] {file}")

    else:

        print(f"[MISSING] {file}")


# --------------------------------------------------
# GRAPH CHECK
# --------------------------------------------------

required_graph = [

    "graph/nodes.npy",
    "graph/edge_index.npy"

]

print("\n[5] Checking Graph Files\n")

for file in required_graph:

    if Path(file).exists():

        print(f"[OK] {file}")

    else:

        print(f"[MISSING] {file}")


# --------------------------------------------------
# MODEL CHECK
# --------------------------------------------------

print("\n[6] Checking Trained Model\n")

if Path("mini_graphcast.pth").exists():

    print("[OK] mini_graphcast.pth")

else:

    print("[MISSING] mini_graphcast.pth")


# --------------------------------------------------
# RESULTS CHECK
# --------------------------------------------------

required_results = [

    "results/prediction.npy",
    "results/ground_truth.npy"

]

print("\n[7] Checking Results\n")

for file in required_results:

    if Path(file).exists():

        print(f"[OK] {file}")

    else:

        print(f"[MISSING] {file}")


print("\n" + "=" * 60)
print("Verification Complete")
print("=" * 60)