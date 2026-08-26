import numpy as np


def rand_rsymp(n,c):
    """Generate random symplectic matrix with specified symplectic eigenvalues.
    Parameters:
        n: dimension (2nx2n)
        c: symplectic eigenvalue(s) - either a scalar or n-vector
    Returns:
        A symplectic matrix
    """
    assert (len(c)==1 or len(c)==n),  (f"Expected vector component length for "
                                       f"symplectic matrix generation either 1 or {n}, got: {len(c)}")
    for scalar in c:
        assert scalar >= 1, f"Expected vector component for symplectic matrix generation >=, got: {scalar}"

    s = np.zeros(2*n)
    if(len(c)==1):
        s[0]=np.sqrt(c[0])
        s[1:n] = np.sort(1 + (s[0] - 1) * np.random.rand(n-1))[::-1]
        s[n:] = 1.0 / s[:n]
    else:
        c = np.sort(c)
        s[0:n] = c[::-1]
        s[n:2*n] = 1.0 / s[0:n]

    U=symp_orth(n)
    V=symp_orth(n)

    return U @ np.diag(s) @ V

def symp_orth(n):
    """Generates a symplectic orthogonal matrix of size n"""
    B = qmult_unit(n)
    Re = np.real(B)
    Im = np.imag(B)
    return np.block([[Re, -Im],
                    [Im,  Re]])

def qmult_unit(n):
    """Generates a Haar-random unitary matrix of size n"""
    F = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    Q, R = np.linalg.qr(F)
    return Q @ np.diag(np.exp(1j * np.random.rand(n) * 2 * np.pi))

def symplectic_form(total_modes):
    """Symplectic form for interleaved (x1,p1,x2,p2,...) ordering."""
    Omega = np.zeros((2 * total_modes, 2 * total_modes))
    for i in range(total_modes):
        Omega[2*i, 2*i+1] = 1
        Omega[2*i+1, 2*i] = -1
    return Omega


def symplectic_eigenvalues(cm, total_modes):
    """Symplectic eigenvalues of a covariance matrix via i*Omega*cm."""
    Omega = symplectic_form(total_modes)
    M = 1j * Omega @ cm
    eigs = np.linalg.eigvals(M)
    nu = np.sort(np.abs(eigs))[::2]  # keep one from each +/- pair
    return np.real(nu)


def partial_transpose_B(cm, n_modes):
    """Partial transpose on Bob's modes: flip sign of each p_B."""
    T = np.eye(cm.shape[0])
    for i in range(n_modes):
        T[4*i + 3, 4*i + 3] = -1
    return T @ cm @ T


def log_negativity(cm, n_modes):
    """Logarithmic negativity across the Alice|Bob bipartition."""
    total_modes = 2 * n_modes
    cm_pt = partial_transpose_B(cm, n_modes)
    nu_tilde = symplectic_eigenvalues(cm_pt, total_modes)
    nu_tilde = np.maximum(nu_tilde, 1e-10)
    below = nu_tilde[nu_tilde < 1.0]
    if below.size == 0:
        return 0.0
    return -np.sum(np.log2(below))


def randCM(entg=1, n_modes=1, num=100000, int_factor=5, tol=0.05, cond_thresh=1e4):
    """Searches for a random state whose entanglement is close to `entg` for 2*n_modes.
    Parameters:
        entg: desired entanglement level (target = entg / int_factor)
        n_modes: number of modes per party
        num: number of random trials before giving up
        int_factor: scaling divisor applied to entg
        tol: acceptance half-width; a state is accepted if
             |EN - target| < tol. Larger tol -> more diversity in
             the returned states' EN, smaller tol -> tighter clustering
             around the target.
        cond_thresh: reject accepted states that are too ill-conditioned
    Returns:
        A (4*n_modes x 4*n_modes) covariance matrix in interleaved convention
        [x1_A, p1_A, x1_B, p1_B, x2_A, p2_A, x2_B, p2_B, ...] whose
        log-negativity is within `tol` of the target, or None if no such
        state was found within `num` trials.
    """
    total_modes = 2 * n_modes
    size = 4 * n_modes
    target = entg / int_factor

    # Build interleaving permutation matrix:
    # split convention [x1_A, x1_B, ..., p1_A, p1_B, ...]
    # -> interleaved convention [x1_A, p1_A, x1_B, p1_B, ...]
    P = np.zeros((size, size))
    for i in range(n_modes):
        P[4*i,   2*i]              = 1  # Alice x_i
        P[4*i+1, 2*n_modes+2*i]    = 1  # Alice p_i
        P[4*i+2, 2*i+1]            = 1  # Bob x_i
        P[4*i+3, 2*n_modes+2*i+1]  = 1  # Bob p_i

    for _ in range(num):
        # Generate thermal state CM in split convention
        d_vals = 1.0 + np.random.rand(total_modes) * 0.5
        g_diag = np.diag(np.repeat(d_vals, 2))

        # Apply random symplectic
        S = rand_rsymp(total_modes, 1.0 + np.random.rand(total_modes) * 5)
        g3 = S.T @ g_diag @ S

        # Apply permutation to get interleaved convention
        cm = P.T @ g3 @ P

        # General log-negativity across the Alice/Bob bipartition
        EN = log_negativity(cm, n_modes)

        if abs(EN - target) < tol:
            if np.linalg.cond(cm) > cond_thresh:
                continue
            return cm

    return None


def randCM_fixed(entg=1, n_modes=1, seed=0):
    """
    Deterministic version of randCM. Returns the same covariance matrix
    for a given (entg, n_modes, seed) combination.
    """
    rng = np.random.default_rng(seed)

    total_modes = 2 * n_modes
    size = 4 * n_modes

    # Same permutation matrix as randCM
    P = np.zeros((size, size))
    for i in range(n_modes):
        P[4*i,    2*i]             = 1
        P[4*i+1,  2*n_modes+2*i]  = 1
        P[4*i+2,  2*i+1]          = 1
        P[4*i+3,  2*n_modes+2*i+1]= 1

    num = 100000
    rot = 1000
    int_factor = 5

    for _ in range(num):
        d_vals = 1.0 + rng.random(total_modes) * 0.5
        g_diag = np.diag(np.repeat(d_vals, 2))

        symp_eigs = 1.0 + rng.random(total_modes) * 5

        # symp_orth and rand_rsymp use global np.random, so we patch via seed
        # by temporarily seeding the global state
        local_seed = int(rng.integers(0, 2**31))
        np.random.seed(local_seed)
        S = rand_rsymp(total_modes, symp_eigs)

        g3 = S.T @ g_diag @ S
        cm = P.T @ g3 @ P

        det_11 = np.linalg.det(cm[0:2, 0:2])
        det_22 = np.linalg.det(cm[2:4, 2:4])
        det_12 = np.linalg.det(cm[0:2, 2:4])
        det_cm = np.linalg.det(cm)

        f = (0.5 * (det_11 + det_22) - det_12
             - np.sqrt(np.maximum(0, (0.5 * (det_11 + det_22) - det_12)**2 - det_cm)))

        EN = -0.5 * np.log2(np.maximum(1e-10, f))
        EN_rounded = np.round(EN * rot) / rot


        if n_modes == 1:
            if EN_rounded == entg / int_factor:
                if np.linalg.cond(cm) > 1e4:
                    continue
                return cm
        else:
            return cm

    return None

def separableCM(n_modes=1):
    """Generate a separable (non-entangled) Gaussian covariance matrix.
    A product state: no correlations between Alice and Bob.
    Each party gets an independent thermal state with random symplectic transformation.
    """
    total_modes = 2 * n_modes
    size = 4 * n_modes

    # Independent thermal states for Alice and Bob
    d_alice = 1.0 + np.random.rand(n_modes) * 0.5
    d_bob   = 1.0 + np.random.rand(n_modes) * 0.5

    g_alice = np.diag(np.repeat(d_alice, 2))
    g_bob   = np.diag(np.repeat(d_bob, 2))

    # Random symplectic on each party independently
    S_alice = rand_rsymp(n_modes, 1.0 + np.random.rand(n_modes) * 5)
    S_bob   = rand_rsymp(n_modes, 1.0 + np.random.rand(n_modes) * 5)

    cm_alice = S_alice.T @ g_alice @ S_alice
    cm_bob   = S_bob.T @ g_bob   @ S_bob

    # Block diagonal — no Alice-Bob correlations
    cm = block_diag(cm_alice, cm_bob)

    return cm

def random_orthonormal_directions(dim):
    d1 = np.random.randn(dim)
    d1 /= np.linalg.norm(d1) + 1e-12

    d2 = np.random.randn(dim)
    d2 -= np.dot(d2, d1) * d1  # Gram-Schmidt
    d2 /= np.linalg.norm(d2) + 1e-12

    return d1, d2