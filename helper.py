
import sys
import subprocess
import os
import faiss


def setup_environment():
    os.environ["OMP_NUM_THREADS"] = "1"

    def in_kaggle():
        return "KAGGLE_KERNEL_RUN_TYPE" in os.environ

    def install(pkg):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", pkg])

    standard_pkgs = ["faker", "matplotlib", "seaborn", "scikit-learn", "pandas", "numpy"]
    
    print(f"Installiere transformers==4.57.6...")
    install("transformers==4.57.6")

    for pkg in standard_pkgs:
        install(pkg)

    try:
        import torch
    except ImportError:
        install("torch")

    try:
        import faiss
    except ImportError:
        if in_kaggle():
            subprocess.check_call(["conda", "install", "-y", "-c", "pytorch", "-c", "nvidia", "faiss-gpu"])
        else:
            install("faiss-cpu")

    return in_kaggle

def setup_device():
    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device


def make_faiss_index(dim: int, device) -> "faiss.Index":
    import faiss

    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 64

    if device.type == "cuda":
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    return index

def print_environment_info(in_kaggle_fn, device):
    import transformers
    import torch

    print("Python:      ", sys.executable)
    print("transformers:", transformers.__version__)
    print(f"Environment : {'Kaggle' if in_kaggle_fn() else 'Local'}")
    print(f"Device      : {device}")
    if device.type == "cuda":
        print(f"GPU         : {torch.cuda.get_device_name(0)}")
