import traceback
import numpy as np
import nlopt
import math
import random
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

def symplectic_values(Z):
    """
    Return the sum of the symplectic eigenvalues for Z.
    This is calculated exactly using the Williamson decomposition from get_S(Z).
    """
    S = get_S(Z)
    result = S @ Z @ S.T
    # For interleaved diagonal D = diag(d1, d1, d2, d2, ...), 
    # we sum d_k once for each pair.
    values = np.diag(result)[::2]
    return np.sum(values)


def measurement(n_modes=1, num_ops=None):
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


def measurement_basis(n_modes=1):
    """Generate a complete, well-conditioned basis of measurement operators."""
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

    return M_list

def check_str_feasibility(M_list, state_g, num_ops, n_modes, upper_bound=100):
    """Check if sTr >= 0.5 is achievable with current measurements."""
    size = 2 * n_modes

    # Maximum possible sTr with all weights at upper bound
    W_max = np.sum([upper_bound * M_list[k] for k in range(num_ops)], axis=0)
    Z1_max = W_max[0:size, 0:size]
    Z2_max = W_max[size:2*size, size:2*size]
    max_sTr = symplectic_values(Z1_max) + symplectic_values(Z2_max)

    # Minimum possible sTr with all weights at zero
    W_min = np.sum([0 * M_list[k] for k in range(num_ops)], axis=0)
    Z1_min = W_min[0:size, 0:size]
    Z2_min = W_min[size:2*size, size:2*size]
    min_sTr = symplectic_values(Z1_min) + symplectic_values(Z2_min)

    # sTr of target state itself
    Z1_g = state_g[0:size, 0:size]
    Z2_g = state_g[size:2*size, size:2*size]
    sTr_g = symplectic_values(Z1_g) + symplectic_values(Z2_g)

    print("=" * 50)
    print("sTr Feasibility Check")
    print("=" * 50)
    print(f"sTr of target state g:        {sTr_g:.6f}")
    print(f"sTr range with w in [0, {upper_bound}]:  [{min_sTr:.6f}, {max_sTr:.6f}]")
    print(f"Required sTr >= 0.5:          {'[OK]' if max_sTr >= 0.5 else '[INFEASIBLE]'}")
    print("=" * 50)

    return max_sTr >= 0.5

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

def check_constraints(w_opt, M_list, min_val, num_ops=14, n_modes=1, verbose=False):
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

    sTr_Z1 = symplectic_values(Z1)
    sTr_Z2 = symplectic_values(Z2)

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
        else:
            print(f"{'sTr Condition':30} | {'ERROR':12} | [FAIL]")
        print(f"{'Steering (w·m < 1)':30} | {min_val:12.8f} | {'[OK]' if steering_ok else '[FAIL]'}")
        print("-" * 50)
        print(f"OVERALL RESULT: {'FEASIBLE [OK]' if results['all_constraints_ok'] else 'UNFEASIBLE [FAIL]'}")
        print("="*50 + "\n")

    return results

def steering_detection(M_list, m_list, num_ops=14, n_modes=1):
    """
    Detect steering using a robust direct optimizer.
    """
    size = 2 * n_modes
    stats = {'eval': 0}

    def objective(w, grad):
        w = np.nan_to_num(w, nan=0.0)
        obj = np.dot(w, m_list)
        if grad.size > 0:
            grad[:] = m_list
        stats['eval'] += 1
        if stats['eval'] % 500 == 0:
            print(f"\r    Eval: {stats['eval']:7d} | Obj: {obj:10.6f}", end="", flush=True)
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

        # print(f"  PSD constraint: {float(1e-9 - min_eigval):.6f}")
        return float(1e-9 - min_eigval)

    def constraint_symplectic_trace(w, grad):
        w = np.nan_to_num(w, nan=0.0)
        W = np.sum([w[idx] * M_list[idx] for idx in range(num_ops)], axis=0)
        Z1 = W[0:size, 0:size]
        Z2 = W[size:2*size, size:2*size]
        
        # Buffer to ensure Z is definitely PSD
        Z1_r = (Z1 + Z1.T) / 2.0
        Z2_r = (Z2 + Z2.T) / 2.0
        
        try:
            if n_modes == 1:
                det1, det2 = np.linalg.det(Z1_r), np.linalg.det(Z2_r)
                sTr1, sTr2 = np.sqrt(np.maximum(1e-18, det1)), np.sqrt(np.maximum(1e-18, det2))
                val = 0.5 - (sTr1 + sTr2)
                if grad.size > 0:
                    inv1, inv2 = np.linalg.inv(Z1_r), np.linalg.inv(Z2_r)
                    g1, g2 = 0.5 * sTr1 * inv1, 0.5 * sTr2 * inv2
                    for k in range(num_ops):
                        dk1 = np.trace(g1 @ M_list[k][0:2, 0:2])
                        dk2 = np.trace(g2 @ M_list[k][2:4, 2:4])
                        grad[k] = -float(dk1 + dk2)
                        # print(f"  grad norm: {np.linalg.norm(grad):.6e}")

            else:
                S1, S2 = get_S(Z1_r), get_S(Z2_r)
                sTr1 = np.sum(np.diag(S1 @ Z1_r @ S1.T)[::2])
                sTr2 = np.sum(np.diag(S2 @ Z2_r @ S2.T)[::2])
                val = 0.5 - (sTr1 + sTr2)
                if grad.size > 0:
                    for k in range(num_ops):
                        dk1 = np.sum(np.diag(S1 @ M_list[k][0:size, 0:size] @ S1.T)[::2])
                        dk2 = np.sum(np.diag(S2 @ M_list[k][size:2*size, size:2*size] @ S2.T)[::2])
                        grad[k] = -float(dk1 + dk2)
            
            if np.isnan(val) or np.isinf(val): return 10.0
            # print(f"  sTr constraint: {val:.6f}")
            return float(val)
        except:
            if grad.size > 0: grad[:] = 0
            return 10.0

    def constraint_steering(w, grad):
        val = np.dot(w, m_list) - 0.999
        if grad.size > 0:
            grad[:] = m_list
        # print(f"  steering constraint: {val:.6f}")
        return float(val)

    opt = nlopt.opt(nlopt.LD_SLSQP, num_ops)  # sequential quadratic programming
    # opt = nlopt.opt(nlopt.LD_MMA, num_ops)  # method of moving asymptotes
    # opt = nlopt.opt(nlopt.LD_CCSAQ, num_ops)  # conservative convex separable approximation
    # opt = nlopt.opt(nlopt.LD_LBFGS, num_ops)  # limited memory BFGS
    # opt = nlopt.opt(nlopt.LD_TNEWTON, num_ops)  # truncated Newton
    # opt = nlopt.opt(nlopt.LD_TNEWTON_RESTART, num_ops)  # truncated Newton with restarts
    # opt = nlopt.opt(nlopt.LD_VAR1, num_ops)  # shifted limited memory variable metric rank 1
    # opt = nlopt.opt(nlopt.LD_VAR2, num_ops)  # shifted limited memory variable metric rank 2
    # opt = nlopt.opt(nlopt.LD_AUGLAG, num_ops)  # augmented Lagrangian (needs sub-optimizer)
    # sub_opt = nlopt.opt(nlopt.LD_LBFGS, num_ops)
    # opt.set_local_optimizer(sub_opt)
    # opt = nlopt.opt(nlopt.LN_COBYLA, num_ops)  # linear approximation, good for constraints
    # opt = nlopt.opt(nlopt.LN_BOBYQA, num_ops)  # quadratic approximation, bound constraints
    # opt = nlopt.opt(nlopt.LN_NEWUOA, num_ops)  # quadratic approximation, unconstrained
    # opt = nlopt.opt(nlopt.LN_PRAXIS, num_ops)  # principal axis method
    # opt = nlopt.opt(nlopt.LN_NELDERMEAD, num_ops)  # Nelder-Mead simplex
    # opt = nlopt.opt(nlopt.LN_SBPLX, num_ops)  # subplex, more robust than Nelder-Mead

    opt.set_min_objective(objective)
    # opt.set_stopval(0.999)
    opt.add_inequality_constraint(constraint_W_psd, 1e-10)
    opt.add_inequality_constraint(constraint_symplectic_trace, 1e-10)
    opt.add_inequality_constraint(constraint_steering, 1e-6)
    opt.set_lower_bounds(np.zeros(num_ops))
    opt.set_upper_bounds(100 * np.ones(num_ops))
    opt.set_xtol_rel(1e-8)
    opt.set_maxeval(20000)

    # Multi-start logic for robustness: Zeros, then Random, then 'Safe Start' (identity-like)
    best_w = None
    best_obj = np.inf
    
    # Construct diverse seeds
    w0 = np.ones(num_ops) * (1.0 / num_ops)

    seeds = [
        w0,  # feasible, uniform
        w0 * 2,  # feasible, larger
        np.abs(np.random.randn(num_ops)) * 0.1,  # feasible, random positive
        np.abs(np.random.randn(num_ops)) * 0.5,
    ]
    
    for w0 in seeds:
        try:
            w_res = opt.optimize(w0)
            obj = opt.last_optimum_value()
            if obj < best_obj:
                # Check if feasible before accepting
                res = check_constraints(w_res, M_list, obj, num_ops, n_modes)
                if res['W_is_PSD'] and res['sTr_satisfied']:
                    best_obj = obj
                    best_w = w_res
            # if best_obj < 0.999: break # Found a solid witness
        except:
            continue
            
    if stats['eval'] >= 500: print()
    return best_obj, best_w

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum Steering Detection")
    parser.add_argument("-nm", "--n_modes", type=int, default=1, help="Number of modes per block (default: 1)")
    parser.add_argument("-e", "--entanglement", type=int, default=1, help="Target entanglement level (default: 1)")
    args, _ = parser.parse_known_args()
    n_modes = args.n_modes
    # parser.add_argument("-no", "--num_ops", type=int, default=2*n_modes*(4*n_modes+1), help="Number of measurement operators (default: 2n(2n+1); n=modes)")
    parser.add_argument("-no", "--num_ops", type=int, default=10, help="Number of measurement operators (default: 2n(2n+1); n=modes)")

    args = parser.parse_args()

    entanglement_target = args.entanglement
    num_ops = args.num_ops
    print(f"Searching for steering witness for {n_modes}-mode state with entanglement level {entanglement_target}...")

    # Generate target state once
    while True:
        state_g = randCM(entanglement_target, n_modes)
        if state_g is not None:
            break
        print("  ...generating new state")

    M_list = measurement(n_modes)  # generate basis once outside the attempt loop
    m_list = [np.real(np.trace(M @ state_g)) for M in M_list]
    num_ops = len(M_list)

    # M_matrix = np.array(M_list).reshape(num_ops, -1)
    # print(f"Condition number of M: {np.linalg.cond(M_matrix):.2e}")
    # input()

    # check_str_feasibility(M_list, state_g, num_ops, n_modes, upper_bound=100)
    max_attempts = 10
    for attempt in range(max_attempts):
        # Run optimization
        print(f"Attempt {attempt+1} ({num_ops} operators): Running optimization with new random measurements...")
        min_val, w_opt = steering_detection(M_list, m_list, num_ops, n_modes)
        
        if w_opt is not None:
            # Check constraints carefully
            results = check_constraints(w_opt, M_list, min_val, num_ops, n_modes, True)
            
            # Check if steering detected
            if results['all_constraints_ok']:
                print(f"\nSUCCESS! Steering detected on attempt {attempt+1}!")
                print(f"Final value: {min_val:.6f}")
                
                # Calculate and print diagonalized matrices for verification
                W_opt = np.sum([w_opt[k] * M_list[k] for k in range(num_ops)], axis=0)
                size = 2 * n_modes
                Z1 = W_opt[0:size, 0:size]
                Z2 = W_opt[size:2*size, size:2*size]
                S1 = get_S(Z1)
                S2 = get_S(Z2)
                D1 = S1 @ Z1 @ S1.T
                D2 = S2 @ Z2 @ S2.T
                
                # print("\nVerification of Williamson Decomposition for optimal W:")
                # print(f"Diagonalized Z1 ({size}x{size}):")
                # print(np.array2string(D1, precision=4, suppress_small=True))
                # print(f"\nDiagonalized Z2 ({size}x{size}):")
                # print(np.array2string(D2, precision=4, suppress_small=True))
                
                # Export f function to CSV
                f_values = w_opt * np.array(m_list)  # shape (num_ops,)
                filename_f = f"output/f_values_nm{n_modes}_ent{entanglement_target}_ops{num_ops}.csv"
                header = "k,w_k,m_k,f_k"
                rows = np.column_stack([
                    np.arange(num_ops),
                    w_opt,
                    np.array(m_list),
                    f_values
                ])
                np.savetxt(filename_f, rows, delimiter=",", header=header, comments="")
                print(f"f(c) values saved to: {filename_f}")

                # Export Z matrices
                filename_z = f"output/z_matrix_nm{n_modes}_ent{entanglement_target}_ops{num_ops}.csv"
                np.savetxt(filename_z, W_opt, delimiter=",",
                           header=f"Z ({2 * size}x{2 * size})", comments="")
                print(f"Z matrix saved to: {filename_z}")

                eigvals_z = np.linalg.eigvalsh(W_opt)
                print(f"Z min eigenvalue: {np.min(eigvals_z):.6e}")
                print(f"Z is PD: {np.all(eigvals_z > 0)}")

                break
        
        print(f"  -> Current Result: {min_val:.6f} (No valid witness found yet)")
