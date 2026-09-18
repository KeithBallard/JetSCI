import os
import sys
import ctypes
import site
import importlib.util


def _preload_nvidia_shared_libraries():
    """Discover and preload pip-installed NVIDIA shared libraries (.so) programmatically.

    Finds the 'nvidia' package directory dynamically via importlib and site-packages
    without making any assumptions about virtualenv location, Python version, or paths.
    Loaded with RTLD_GLOBAL so PETSc and petsc4py find CUDA symbols in process memory.
    """
    candidate_dirs = set()

    # 1. Discover via importlib module spec
    try:
        spec = importlib.util.find_spec("nvidia")
        if spec and spec.submodule_search_locations:
            for loc in spec.submodule_search_locations:
                if os.path.isdir(loc):
                    candidate_dirs.add(os.path.abspath(loc))
    except Exception:
        pass

    # 2. Discover via Python's active site-packages / user-site paths
    try:
        site_dirs = []
        if hasattr(site, "getsitepackages"):
            site_dirs.extend(site.getsitepackages())
        if hasattr(site, "getusersitepackages"):
            user_site = site.getusersitepackages()
            if isinstance(user_site, str):
                site_dirs.append(user_site)
        for s_dir in site_dirs:
            n_dir = os.path.join(s_dir, "nvidia")
            if os.path.isdir(n_dir):
                candidate_dirs.add(os.path.abspath(n_dir))
    except Exception:
        pass

    # Preload all discovered shared libraries
    for nvidia_dir in candidate_dirs:
        for root, _, files in os.walk(nvidia_dir):
            for f in files:
                if ".so" in f:
                    try:
                        ctypes.CDLL(os.path.join(root, f), mode=ctypes.RTLD_GLOBAL)
                    except OSError:
                        pass


_preload_nvidia_shared_libraries()
