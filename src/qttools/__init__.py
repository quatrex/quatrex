# Copyright 2023-2024 ETH Zurich and Quantum Transport Toolbox authors.

import os
from warnings import warn

from qttools.__about__ import __version__

# Allows user to specify the array module via an environment variable.
ARRAY_MODULE = os.environ.get("ARRAY_MODULE")
if ARRAY_MODULE is not None:
    if ARRAY_MODULE == "numpy":
        import numpy as xp
        from scipy import sparse

    elif ARRAY_MODULE == "cupy":
        try:
            import cupy as xp
            from cupyx.scipy import sparse

        except ImportError as e:
            warn(f"'cupy' is unavailable, defaulting to 'numpy'. ({e})")
            import numpy as xp
            from scipy import sparse

    else:
        raise ValueError(f"Unrecognized ARRAY_MODULE '{ARRAY_MODULE}'")

else:
    # If the user does not specify the array module, prioritize cupy but
    # default to numpy if cupy is not available or not working.
    try:
        import cupy as xp
        from cupyx.scipy import sparse

        try:
            # Check if cupy is actually working. This could still raise
            # a cudaErrorInsufficientDriver error or something.
            xp.abs(1)

        except Exception as e:
            warn(f"'cupy' is unavailable, defaulting to 'numpy'. ({e})")
            import numpy as xp
            from scipy import sparse

    except ImportError as e:
        warn(f"'cupy' is unavailable, defaulting to 'numpy'. ({e})")
        import numpy as xp
        from scipy import sparse


__all__ = ["__version__", "xp", "sparse"]
