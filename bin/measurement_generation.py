import numpy as np

def measurement_random(n_modes=1, num_ops=None):
    """Return a list of random rank-1 measurements for 2*n_modes system."""
    size = 4 * n_modes
    if num_ops is None:
        num_ops = 2 * n_modes * (4 * n_modes + 1)  # default same as argparse

    M_list = []
    for _ in range(num_ops):
        u = np.random.randn(size)
        u /= (np.linalg.norm(u) + 1e-16)
        M_list.append(np.outer(u, u))

    return M_list

def measurement_homogeneous(n_modes=1, num_ops=None):
    """Generate a well-conditioned set of measurement operators.
    Uses a structured basis (diagonal + off-diagonal) for best conditioning.
    If num_ops < full basis size, returns first num_ops operators.
    If num_ops > full basis size, pads with random operators.
    """
    size = 4 * n_modes
    M_list = []

    # Diagonal basis: e_i e_i^T
    for i in range(size):
        u = np.zeros(size)
        u[i] = 1.0
        M_list.append(np.outer(u, u))

    # Off-diagonal basis: (e_i + e_j)(e_i + e_j)^T / 2
    for i in range(size):
        for j in range(i + 1, size):
            u = np.zeros(size)
            u[i] = 1.0 / np.sqrt(2)
            u[j] = 1.0 / np.sqrt(2)
            M_list.append(np.outer(u, u))

    if num_ops is None:
        return M_list
    elif num_ops <= len(M_list):
        return M_list[:num_ops]
    else:
        # pad with random operators
        for _ in range(num_ops - len(M_list)):
            u = np.random.randn(size)
            u /= (np.linalg.norm(u) + 1e-16)
            M_list.append(np.outer(u, u))
        return M_list

def measurement_fixed(n_modes=1, num_ops=None):
    """
    For testing only
    Deterministic measurement operators.
    Uses a fixed orthonormal basis of symmetric matrices.
    """
    size = 4 * n_modes
    M_list = []

    # --- 1. Diagonal elements ---
    for i in range(size):
        M = np.zeros((size, size))
        M[i, i] = 1.0
        M_list.append(M)

    # --- 2. Symmetric off-diagonal elements ---
    for i in range(size):
        for j in range(i + 1, size):
            M = np.zeros((size, size))
            M[i, j] = 1.0 / np.sqrt(2)
            M[j, i] = 1.0 / np.sqrt(2)
            M_list.append(M)

    # Total possible operators
    full_basis_size = len(M_list)

    if num_ops is None:
        return M_list

    if num_ops <= full_basis_size:
        return M_list[:num_ops]

    # --- 3. If more requested, pad deterministically ---
    # Use fixed pseudo-random but deterministic vectors
    rng = np.random.default_rng(12345)

    while len(M_list) < num_ops:
        u = rng.standard_normal(size)
        u /= np.linalg.norm(u) + 1e-16
        M_list.append(np.outer(u, u))

    return M_list