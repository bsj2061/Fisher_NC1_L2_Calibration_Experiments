"""
Shared environment setup. Import this BEFORE importing torch/numpy.

Handles the OpenMP duplicate-library error common on Windows + Anaconda:
    OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll
    already initialized.

The workaround is documented (and discouraged) by Intel, but is the standard
solution for PyTorch + numpy/scipy installs on Windows. In practice it does
not cause issues for the operations used in this codebase.
"""
import os

# Allow duplicate OpenMP libraries (PyTorch's libiomp5md.dll vs numpy's).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Cap OpenMP threads to avoid CPU oversubscription when multiple libs each
# spawn their own thread pools.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
