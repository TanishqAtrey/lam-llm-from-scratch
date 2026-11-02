import os

# Prevent OpenMP runtime crash (OMP: Error #15) on macOS when PyTorch and FAISS/numpy are both loaded
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
