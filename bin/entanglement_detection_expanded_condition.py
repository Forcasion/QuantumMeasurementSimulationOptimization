import nlopt
import numpy as np
import time
from scipy.linalg import block_diag


# Tolerances

tolerance_psd = 1e-10
tolerance_sTr = 1e-10
tolerance_steering = 1e-10

tolerance_x = 1e-20
tolerance_f = 1e-20


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

    if n_modes == 1:
        det = np.linalg.det(Z_r)

        # val = np.sqrt(np.maximum(1e-18, det))
        if det <= 0:
            return None, None

        val = np.sqrt(det)

        try:
            inv = np.linalg.inv(Z_r)
            grad_Z = 0.5 * val * inv  # d(sqrt(det(Z)))/dZ = 0.5 * sqrt(det(Z)) * Z^{-1}
        except:
            grad_Z = None
    else:
        S = get_S(Z_r)
        val = np.sum(np.diag(S @ Z_r @ S.T)[::2])
        grad_Z = S.T[::2, :].T @ S[::2, :]  # d(sTr)/dZ = S^T D S where D selects even rows
        grad_Z = None  # Ignore this. n_modes is always 1 for now.

    return val, grad_Z

def check_constraints(w_opt, M_list, min_val, num_ops=10, n_modes=1, verbose=False):
    """
    Verify all constraints are satisfied.
    """
    W = np.sum([w_opt[k] * M_list[k] for k in range(num_ops)], axis=0)
    size = 2 * n_modes

    # Check 1: W ≥ 0 (PSD)
    eigvals = np.linalg.eigvalsh(W)
    min_eigval = np.min(np.real(eigvals))
    W_psd = min_eigval >= -tolerance_psd

    # Check 2: sTr(Z1) + sTr(Z2) ≥ 0.5
    Z1 = W[0:size, 0:size]
    Z2 = W[size:2*size, size:2*size]

    sTr_Z1, _ = sTr(Z1, n_modes=n_modes)
    sTr_Z2, _ = sTr(Z2, n_modes=n_modes)

    if sTr_Z1 is None or sTr_Z2 is None:
        sTr_satisfied = False
    else:
        sTr_sum = sTr_Z1 + sTr_Z2
        sTr_satisfied = sTr_sum >= 0.5 - tolerance_sTr

    steering_ok = min_val < 1.0 - tolerance_steering

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
        print(f"{'PSD (Min eigval >= 0)':30} | {min_eigval:12.20f} | {'[OK]' if W_psd else '[FAIL]'}")
        if sTr_sum is not None:
            print(f"{'sTr (Sum traces >= 0.5)':30} | {sTr_sum:12.20f} | {'[OK]' if sTr_satisfied else '[FAIL]'}")
            print(f"{'sTr(Z1)':30} | {sTr_Z1:12.8f} | ")
            print(f"{'sTr(Z2)':30} | {sTr_Z2:12.8f} | ")

        else:
            print(f"{'sTr Condition':30} | {'ERROR':12} | [FAIL]")
        print(f"{'Steering (w·m < 1)':30} | {min_val:12.20f} | {'[OK]' if steering_ok else '[FAIL]'}")
        print("-" * 50)
        print(f"OVERALL RESULT: {'FEASIBLE [OK]' if results['all_constraints_ok'] else 'UNFEASIBLE [FAIL]'}")
        print("="*50 + "\n")


    return results

def find_good_seeds(M_list, m_list, num_ops, n_modes, n_candidates=1000, n_best=4):
    # For good results n_candidates = 1000000 or more
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

        if sTr1 is None or sTr2 is None:
            continue
        else:
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

def find_fixed_seeds(M_list, m_list, num_ops, n_modes, n_candidates=10000, n_best=4):
    """deprecated"""
    w0 = np.ones(num_ops) * (1.0 / num_ops)
    seeds = [
        w0,
        w0 * 2]
    return seeds

def entanglement_detection(M_list, m_list, num_ops=14, n_modes=1):
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
        output_frequency = 1000
        if stats['eval'] % output_frequency == 0:
            W, Z1, Z2, sTr1, sTr2, g1, g2 = compute_sTr_sum(w)
            if sTr1 is None or sTr2 is None:
                val = -1*np.inf
            else:
                val = sTr1 + sTr2
            f = np.dot(w, m_list)
            W = np.sum([w[idx] * M_list[idx] for idx in range(num_ops)], axis=0)
            min_eig = np.min(np.linalg.eigvalsh(W))
            print(f"\n    Eval: {stats['eval']:7d} | sTr: {val:10.20f} | f: {f:10.20f} | min_eig: {min_eig:+.20f}", end="", flush=True)
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
        return float(tolerance_psd - min_eigval)

    def constraint_symplectic_trace(w, grad):
        w = np.nan_to_num(w, nan=0.0)
        try:
            W, Z1, Z2, sTr1, sTr2, g1, g2 = compute_sTr_sum(w)

            if sTr1 is None or sTr2 is None:
                if grad.size > 0:
                    grad[:] = 0
                return 10.0

            val = 0.5 - (sTr1 * sTr2) # HERE!!!!

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

    # Optimization
    # opt = nlopt.opt(nlopt.LD_MMA, num_ops)
    # opt = nlopt.opt(nlopt.LD_CCSAQ, num_ops)
    opt = nlopt.opt(nlopt.LD_SLSQP, num_ops)
    opt.set_min_objective(objective)

    opt.add_inequality_constraint(constraint_W_psd, tolerance_psd)
    opt.add_inequality_constraint(constraint_symplectic_trace, tolerance_sTr)
    opt.add_inequality_constraint(constraint_steering, tolerance_steering)

    opt.set_lower_bounds(-50 * np.ones(num_ops))
    opt.set_upper_bounds(50 * np.ones(num_ops))

    opt.set_xtol_rel(tolerance_x)
    opt.set_ftol_rel(tolerance_f)
    opt.set_maxeval(20000)

    best_w = None
    best_obj = np.inf

    # #Seeds for global optimization
    # seeds = [w_global]
    # for _ in range(4):
    #     noise = np.random.randn(num_ops) * 0.05
    #     seeds.append(w_global + noise)


    print("\nGenerating seeds")
    # seeds = find_fixed_seeds(M_list, m_list, num_ops, n_modes)
    start_time_seeds = time.time()
    seeds = find_good_seeds(M_list, m_list, num_ops, n_modes, n_candidates = 10000)
    end_time_seeds = time.time()
    print(f"\nSeeds generated. time {end_time_seeds - start_time_seeds}s")


    if not seeds:
        print("\n" + "="*50)
        print("\nFailed to find good seeds")
        print("\n" + "="*50)
        return best_obj, best_w

    print("\nStarting optimization")
    start_time_opt = time.time()
    for seed_idx, seed in enumerate(seeds):
        try:
            w_res = opt.optimize(seed)
            f_k = np.dot(w_res, m_list)
            if f_k < best_obj:
                res = check_constraints(w_res, M_list, f_k, num_ops, n_modes, verbose=False)
                if res['W_is_PSD'] and res['sTr_satisfied']:
                    best_obj = f_k
                    best_w = w_res
                # elif best_w is None:  # fallback: keep best even if marginally infeasible
                #     best_obj = f_k
                #     best_w = w_res

        except Exception as e:
            print(f"\n  Seed {seed_idx} failed: {e}")
            continue
    end_time_opt = time.time()
    print(f"\nOptimization time {end_time_opt - start_time_opt}s")
    # visualiser_wrapper(opt, w_res)
    return best_obj, best_w
