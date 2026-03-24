import traceback
import numpy as np
import nlopt
import math
import random
import matplotlib.pyplot as plt

from IPython.lib.deepreload import found_now
from scipy.linalg import block_diag, expm
from numpy import *
import argparse

def get_S(Z):
    """
    Find the symplectic transformation S such that S Z S^T = D,
    where D = diag(d1, d1, d2, d2, ..., dn, dn).
    """
    size = len(Z)
    n = size // 2

    # 1. Standard interleaved Omega for [x1, y1, x2, y2, ...]
    J = np.array([[0, 1], [-1, 0]])
    Omega = block_diag(*([J] * n))

    # 2. Ensure Z is symmetric and regularized (robust to singular matrices)
    # Optional, but it helps sometimes
    # Z = (Z + Z.T) / 2 + 1e-12 * np.eye(size)

    # 3. Williamson decomposition:
    # Symplectic eigenvalues are eigenvalues of A = i * Omega @ Z
    A = 1j * Omega @ Z
    vals, vecs = np.linalg.eig(A)

    # Positive eigenvalues (d_k) and their eigenvectors
    # Sort by real parts descending. For PSD Z, they should be +/- d_k.
    # We take the n largest (one from each pair).
    sort_idx = np.argsort(vals.real)[::-1]
    d_ks = vals[sort_idx[:n]].real
    vecs_pos = vecs[:, sort_idx[:n]]

    # 4. Construct S such that S Z S^T = D
    # Normalize v_k such that v_k^H (i Omega) v_k = 2
    # Then the rows of S are interleaved: Im(v_k)^T and Re(v_k)^T
    rows = []
    for i in range(n):
        v = vecs_pos[:, i]
        # Normalize: v^H (i Omega) v = 2
        norm = np.real(v.conj().T @ (1j * Omega) @ v)
        # Numerical safety: ensure norm is not zero before division
        v = v * np.sqrt(2.0 / (np.abs(norm) + 1e-16))
        
        # Phase fixing for numerical stability: force first non-zero element to be real-positive
        abs_v = np.abs(v)
        if np.max(abs_v) > 1e-12:
            first_idx = np.argmax(abs_v > 1e-12)
            first_nonzero = v[first_idx]
            v = v * (np.conj(first_nonzero) / (np.abs(first_nonzero) + 1e-18))
        
        rows.append(np.imag(v).T)
        rows.append(np.real(v).T)
        
    S = np.vstack(rows)
    
    return S

def sTr(Z, n_modes=1):
    """Compute symplectic trace sTr(Z) = sum of symplectic eigenvalues.
    For n_modes=1: sTr(Z) = sqrt(det(Z))
    For n_modes>1: uses Williamson decomposition via get_S.
    Also returns the gradient function dsTr/dZ for use in optimization.
    """
    Z_r = (Z + Z.T) / 2.0

    # Check if Z is PD to send back a signal of infesability to the objective
    eigvals = np.linalg.eigvalsh(Z_r)
    if np.any(eigvals <= 0):
        return -100.0, None

    if n_modes == 1:
        det = np.linalg.det(Z_r)
        val = np.sqrt(np.maximum(1e-18, det))
        try:
            inv = np.linalg.inv(Z_r)
            grad_Z = 0.5 * val * inv  # d(sqrt(det(Z)))/dZ = 0.5 * sqrt(det(Z)) * Z^{-1}
        except:
            grad_Z = None
    else:
        S = get_S(Z_r)
        val = np.sum(np.diag(S @ Z_r @ S.T)[::2])
        grad_Z = S.T[::2, :].T @ S[::2, :]  # d(sTr)/dZ = S^T D S where D selects even rows
        grad_Z = None  # placeholder — multi-mode gradient needs more careful derivation

    return val, grad_Z


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

def qmult_unit(n):
    """Generates a Haar-random unitary matrix of size n"""
    F = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    Q, R = np.linalg.qr(F)
    return Q @ np.diag(np.exp(1j * np.random.rand(n) * 2 * np.pi))

def symp_orth(n):
    """Generates a symplectic orthogonal matrix of size n"""
    B = qmult_unit(n)
    Re = np.real(B)
    Im = np.imag(B)
    return np.block([[Re, -Im],
                    [Im,  Re]])

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

def randCM(entg=1, n_modes=1):
    """Searches for a random state with a specific entanglement level for 2*n_modes.
    Parameters:
        entg: entanglement level
        n_modes: number of modes per party
    Returns:
        A (4*n_modes x 4*n_modes) covariance matrix in interleaved convention
        [x1_A, p1_A, x1_B, p1_B, x2_A, p2_A, x2_B, p2_B, ...]
    """
    total_modes = 2 * n_modes
    size = 4 * n_modes
    num = 100000
    rot = 10
    int_factor = 5

    # Build interleaving permutation matrix
    # Converts split convention [x1_A, x1_B, ..., p1_A, p1_B, ...]
    # to interleaved convention [x1_A, p1_A, x1_B, p1_B, ...]
    P = np.zeros((size, size))
    for i in range(n_modes):
        P[4*i,   2*i]            = 1  # Alice x_i
        P[4*i+1, 2*n_modes+2*i]  = 1  # Alice p_i
        P[4*i+2, 2*i+1]          = 1  # Bob x_i
        P[4*i+3, 2*n_modes+2*i+1]= 1  # Bob p_i

    for _ in range(num):
        # Generate thermal state CM in split convention
        d_vals = 1.0 + np.random.rand(total_modes) * 0.5
        g_diag = np.diag(np.repeat(d_vals, 2))

        # Apply random symplectic
        S = rand_rsymp(total_modes, 1.0 + np.random.rand(total_modes) * 5)
        g3 = S.T @ g_diag @ S

        # Apply permutation to get interleaved convention
        cm = P.T @ g3 @ P

        # Logarithmic negativity check
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
                return cm
        else:
            # For n_modes > 1, log negativity across the Alice/Bob bipartition
            # needs a proper PPT check — for now just return a valid CM
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

def check_constraints(w_opt, M_list, min_val, num_ops=10, n_modes=1, verbose=False):
    """
    Verify all constraints are satisfied.
    """
    W = np.sum([w_opt[k] * M_list[k] for k in range(num_ops)], axis=0)
    size = 2 * n_modes

    # Check 1: W ≥ 0 (PSD)
    eigvals = np.linalg.eigvalsh(W)
    min_eigval = np.min(np.real(eigvals))
    W_psd = min_eigval >= -1e-9

    # Check 2: sTr(Z1) + sTr(Z2) ≥ 0.5
    Z1 = W[0:size, 0:size]
    Z2 = W[size:2*size, size:2*size]

    sTr_Z1, _ = sTr(Z1, n_modes=n_modes)
    sTr_Z2, _ = sTr(Z2, n_modes=n_modes)

    sTr_sum = sTr_Z1 + sTr_Z2
    sTr_satisfied = sTr_sum >= 0.5 - 1e-7

    steering_ok = min_val < 1.0 - 1e-7

    results = {
        'min_eigval_W': min_eigval,
        'W_is_PSD': W_psd,
        'sTr_sum': sTr_sum,
        'sTr_satisfied': sTr_satisfied,
        'steering_satisfied': steering_ok,
        'objective_value': min_val,
        'all_constraints_ok': W_psd and sTr_satisfied and steering_ok
    }

    if verbose:
        print("\n" + "="*50)
        print(f"{'Condition':30} | {'Value':12} | {'Result'}")
        print("-" * 50)
        print(f"{'PSD (Min eigval >= 0)':30} | {min_eigval:12.8f} | {'[OK]' if W_psd else '[FAIL]'}")
        if sTr_sum is not None:
            print(f"{'sTr (Sum traces >= 0.5)':30} | {sTr_sum:12.8f} | {'[OK]' if sTr_satisfied else '[FAIL]'}")
            print(f"{'sTr(Z1)':30} | {sTr_Z1:12.8f} | ")
            print(f"{'sTr(Z2)':30} | {sTr_Z2:12.8f} | ")

        else:
            print(f"{'sTr Condition':30} | {'ERROR':12} | [FAIL]")
        print(f"{'Steering (w·m < 1)':30} | {min_val:12.8f} | {'[OK]' if steering_ok else '[FAIL]'}")
        print("-" * 50)
        print(f"OVERALL RESULT: {'FEASIBLE [OK]' if results['all_constraints_ok'] else 'UNFEASIBLE [FAIL]'}")
        print("="*50 + "\n")

    return results

def random_orthonormal_directions(dim):
    d1 = np.random.randn(dim)
    d1 /= np.linalg.norm(d1) + 1e-12

    d2 = np.random.randn(dim)
    d2 -= np.dot(d2, d1) * d1  # Gram-Schmidt
    d2 /= np.linalg.norm(d2) + 1e-12

    return d1, d2

def evaluate_2d_slice(w0, d1, d2, M_list, m_list, num_ops, n_modes,
                      grid_size=80, span=1.0):
    size = 2 * n_modes
    alphas = np.linspace(-span, span, grid_size)
    betas  = np.linspace(-span, span, grid_size)

    F = np.zeros((grid_size, grid_size))
    PSD = np.zeros_like(F)
    STR = np.zeros_like(F)
    STEER = np.zeros_like(F)

    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            w = w0 + a*d1 + b*d2

            # Objective
            f = np.dot(w, m_list)
            F[i, j] = f

            # Build W
            W = np.sum([w[k] * M_list[k] for k in range(num_ops)], axis=0)
            W = (W + W.T) / 2.0

            # PSD constraint
            min_eig = np.min(np.linalg.eigvalsh(W))
            PSD[i, j] = min_eig

            # Symplectic trace
            Z1 = W[0:size, 0:size]
            Z2 = W[size:2*size, size:2*size]
            sTr1, _ = sTr(Z1, n_modes)
            sTr2, _ = sTr(Z2, n_modes)
            STR[i, j] = sTr1 + sTr2 - 0.5

            # Steering
            STEER[i, j] = 0.999 - f

    return alphas, betas, F, PSD, STR, STEER


def plot_2d_slice(alphas, betas, F, PSD, STR, STEER):
    A, B = np.meshgrid(alphas, betas, indexing='ij')

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ============================================================
    # LEFT: OBJECTIVE LANDSCAPE
    # ============================================================
    ax = axes[0]

    obj = ax.contourf(A, B, F, levels=50)
    ax.scatter(0, 0, s=80, marker='x')
    fig.colorbar(obj, ax=ax, label="Objective f(w)")

    ax.set_title("Objective Landscape")
    ax.set_xlabel("alpha")
    ax.set_ylabel("beta")
    ax.grid(True, alpha=0.3)

    # ============================================================
    # RIGHT: CONSTRAINT GEOMETRY
    # ============================================================
    ax = axes[1]
    ax.scatter(0, 0, s=80, marker='x')

    # Feasible regions (binary masks)
    psd_region = PSD >= 0
    str_region = STR >= 0
    steer_region = STEER >= 0

    # Plot shaded regions separately (different layers)
    ax.contourf(A, B, psd_region, levels=[0.5, 1], alpha=0.2)
    ax.contourf(A, B, str_region, levels=[0.5, 1], alpha=0.2)
    ax.contourf(A, B, steer_region, levels=[0.5, 1], alpha=0.2)

    # Boundaries
    c_psd = ax.contour(A, B, PSD, levels=[0], linewidths=2)
    c_str = ax.contour(A, B, STR, levels=[0], linestyles='--', linewidths=2)
    c_steer = ax.contour(A, B, STEER, levels=[0], linestyles=':', linewidths=2)

    # Labels
    ax.clabel(c_psd, fmt={0: 'PSD'}, inline=True)
    ax.clabel(c_str, fmt={0: 'sTr'}, inline=True)
    ax.clabel(c_steer, fmt={0: 'Steering'}, inline=True)

    ax.set_title("Constraint Feasible Regions")
    ax.set_xlabel("alpha")
    ax.set_ylabel("beta")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def visualiser_wrapper(opt, w_res):
    result_code = opt.last_optimize_result()
    if w_res is not None:
        print("\nVisualizing local geometry")

        d1, d2 = random_orthonormal_directions(num_ops)

        alphas, betas, F, PSD, STR, STEER = evaluate_2d_slice(
            w_res, d1, d2,
            M_list, m_list, num_ops, n_modes,
            grid_size=160,
            span=1
        )

        plot_2d_slice(alphas, betas, F, PSD, STR, STEER)
def find_good_seeds(M_list, m_list, num_ops, n_modes, n_candidates=10000, n_best=4):
    size = 2 * n_modes
    candidates = []

    for _ in range(n_candidates):
        w = np.random.randn(num_ops) * 0.5
        f_k = np.dot(w, m_list)

        W = np.sum([w[idx] * M_list[idx] for idx in range(num_ops)], axis=0)
        Z1 = W[0:size, 0:size]
        Z2 = W[size:2 * size, size:2 * size]

        min_eig = np.min(np.linalg.eigvalsh(W))
        sTr1, _ = sTr(Z1, n_modes)
        sTr2, _ = sTr(Z2, n_modes)
        sTr_sum = sTr1 + sTr2

        # score: want high sTr, close to steering boundary, close to PSD
        steering_penalty = abs(f_k - 0.999)  # close to boundary from either side
        psd_penalty = abs(min(min_eig, 0)) * 10  # penalize negative eigenvalues
        str_penalty = abs(min(sTr_sum - 0.5, 0)) * 10

        score = steering_penalty + psd_penalty + str_penalty

        candidates.append((score, w))

    candidates.sort(key=lambda x: x[0])
    seeds = [w for _, w in candidates[:n_best]]
    w0 = np.ones(num_ops) * (1.0 / num_ops)
    seeds.append(w0)
    return seeds

def steering_detection(M_list, m_list, num_ops=14, n_modes=1):
    """
    Detect steering using a robust direct optimizer.
    Feasibility problem: find c >= 0 such that:
        f_k(c) = Σ c_j m_j < 1
        Z = Σ c_j M_j >= 0
        sTr(Z1) + sTr(Z2) >= 0.5
    Objective: maximize sTr(Z1) + sTr(Z2) to drive into feasible region.
    """
    size = 2 * n_modes
    stats = {'eval': 0}

    def compute_sTr_sum(w):
        """Helper: compute Z, Z1, Z2, sTr1, sTr2, g1, g2 from weights."""
        W = np.sum([w[idx] * M_list[idx] for idx in range(num_ops)], axis=0)
        Z1 = W[0:size, 0:size]
        Z2 = W[size:2*size, size:2*size]
        sTr1, g1 = sTr(Z1, n_modes)
        sTr2, g2 = sTr(Z2, n_modes)
        return W, Z1, Z2, sTr1, sTr2, g1, g2

    def objective(w, grad):
        w = np.nan_to_num(w, nan=0.0)
        try:
            obj = np.dot(w, m_list)
            if grad.size > 0:
                grad[:] = m_list
        except:
            obj = 0.0
            if grad.size > 0:
                grad[:] = 0

        stats['eval'] += 1
        output_frequency = 1
        if stats['eval'] % output_frequency == 0:
            W, Z1, Z2, sTr1, sTr2, g1, g2 = compute_sTr_sum(w)
            val = sTr1 + sTr2
            f = np.dot(w, m_list)
            W = np.sum([w[idx] * M_list[idx] for idx in range(num_ops)], axis=0)
            min_eig = np.min(np.linalg.eigvalsh(W))
            print(f"\r    Eval: {stats['eval']:7d} | sTr: {val:10.6f} | f: {f:10.6f} | min_eig: {min_eig:+.6f}", end="", flush=True)
        return float(obj)

    def constraint_W_psd(w, grad):
        w = np.nan_to_num(w, nan=0.0)
        W = np.sum([w[idx] * M_list[idx] for idx in range(num_ops)], axis=0)
        W = (W + W.T) / 2.0
        eigvals, eigvecs = np.linalg.eigh(W)
        min_idx = np.argmin(eigvals)
        min_eigval = eigvals[min_idx]
        v_min = eigvecs[:, min_idx]
        if grad.size > 0:
            for k in range(num_ops):
                grad[k] = -float(np.real(v_min.conj().T @ M_list[k] @ v_min))
        return float(1e-9 - min_eigval)

    def constraint_symplectic_trace(w, grad):
        w = np.nan_to_num(w, nan=0.0)
        try:
            W, Z1, Z2, sTr1, sTr2, g1, g2 = compute_sTr_sum(w)
            val = 0.5 - (sTr1 + sTr2)
            if grad.size > 0 and g1 is not None and g2 is not None:
                for k in range(num_ops):
                    dk1 = np.trace(g1 @ M_list[k][0:size, 0:size])
                    dk2 = np.trace(g2 @ M_list[k][size:2*size, size:2*size])
                    grad[k] = -float(dk1 + dk2)
            elif grad.size > 0:
                grad[:] = 0
            if np.isnan(val) or np.isinf(val): return 10.0
            return float(val)
        except:
            if grad.size > 0: grad[:] = 0
            return 10.0

    def constraint_steering(w, grad):
        val = np.dot(w, m_list) - 0.999
        if grad.size > 0:
            grad[:] = m_list
        return float(val)


    # opt = nlopt.opt(nlopt.LD_MMA, num_ops)
    # opt = nlopt.opt(nlopt.LD_CCSAQ, num_ops)
    opt = nlopt.opt(nlopt.LD_SLSQP, num_ops)
    opt.set_min_objective(objective)
    opt.add_inequality_constraint(constraint_W_psd, 1e-10)
    opt.add_inequality_constraint(constraint_symplectic_trace, 1e-10)
    opt.add_inequality_constraint(constraint_steering, 1e-10)
    opt.set_lower_bounds(-100 * np.ones(num_ops))
    opt.set_upper_bounds(100 * np.ones(num_ops))
    opt.set_xtol_rel(1e-10)
    opt.set_ftol_rel(1e-10)  # add function tolerance too
    opt.set_maxeval(20000)

    best_w = None
    best_obj = np.inf

    # seeds = find_good_seeds(M_list, m_list, num_ops, n_modes)
    # if not seeds:
    #     print("\n" + "="*50)
    #     print("\nFailed to find good seeds")
    #     print("\n" + "="*50)
    #     return best_obj, best_w

    w0 = np.ones(num_ops) * (1.0 / num_ops)
    seeds = [
        w0
    ]

    for seed_idx, seed in enumerate(seeds):
        try:
            w_res = opt.optimize(seed)
            f_k = np.dot(w_res, m_list)
            if f_k < best_obj:
                res = check_constraints(w_res, M_list, f_k, num_ops, n_modes, verbose=False)
                if res['W_is_PSD'] and res['sTr_satisfied']:
                    best_obj = f_k
                    best_w = w_res

        except Exception as e:
            print(f"\n  Seed {seed_idx} failed: {e}")
            continue

    # visualiser_wrapper(opt, w_res)
    return best_obj, best_w

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum Steering Detection")
    parser.add_argument("-nm", "--n_modes", type=int, default=1, help="Number of modes per block (default: 1)")
    parser.add_argument("-e", "--entanglement", type=int, default=1, help="Target entanglement level (default: 1)")
    parser.add_argument("-mo", "--max_ops", type=int, default=10, help="Number of maximum measurement operators")
    parser.add_argument("-ma", "--max_attempts", type=int, default=1, help="Max optimization attempts per state (default: 1)")

    args = parser.parse_args()
    n_modes = args.n_modes
    entanglement_target = args.entanglement
    max_ops = args.max_ops
    max_attempts = args.max_attempts

    print(f"Searching for steering witness for {n_modes}-mode state with entanglement level {entanglement_target}...")

    found = False
    successes = 0
    failures = 0

    # Generate new state
    state_g = None
    while state_g is None:
        state_g = randCM(entanglement_target, n_modes)
    print(f"\nState generated.")

    filename = f"output/single_state_f_values_ent{entanglement_target}.csv"
    with open(filename, "w") as f:
        f.write("entanglement, measurements, f\n")

        for num_ops in range(1, max_ops+1):
            # Generate measurements for this state
            M_list = measurement_random(n_modes, num_ops)
            m_list = [np.real(np.trace(M @ state_g)) for M in M_list]
            num_ops = len(M_list)

            for attempt in range(max_attempts):
                print(f"  Attempt {attempt+1}/{max_attempts} ({num_ops} operators)...")
                min_val, w_opt = steering_detection(M_list, m_list, num_ops, n_modes)
                if min_val is not inf:
                    print(f"\nSteering detected for {num_ops} measurements using entanglement level {entanglement_target}.")
                else:
                    print(f"\nOptimization failed for {num_ops} measurements using entanglement level {entanglement_target}.")
                f.write(
                    f"{entanglement_target},{num_ops},{min_val}\n")



